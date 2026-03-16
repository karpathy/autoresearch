"""
Autoresearch fine-tuning script. Multi-GPU DDP + LoRA anti-forgetting safeguards.
Usage 6x GPUs: torchrun --nproc_per_node=6 train.py

Optimizes val_bpb_domain while monitoring val_bpb_general for catastrophic forgetting.
ANTI-FORGETTING safeguards are mandatory — do not remove them.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# ANTI-FORGETTING — do not remove: PCIe consumer GPUs hang without these NCCL vars
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import gc
import json
import math
import time
import datetime
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

# ---------------------------------------------------------------------------
# Hyperparameters — agent tunes these
# ---------------------------------------------------------------------------

MODEL_NAME = "ADD_MODEL_NAME"

# ANTI-FORGETTING — do not remove: LoRA keeps base model weights frozen
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0          # 0 required with gradient checkpointing (re-entrant fwd)
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ANTI-FORGETTING — do not remove: low LR prevents catastrophic weight drift
LR = 2e-4
WARMUP_RATIO = 0.05       # linear warmup over first 5% of steps
WEIGHT_DECAY = 0.01

# ANTI-FORGETTING — do not remove: REPLAY_RATIO must never be 0.0
REPLAY_RATIO = 0.2        # fraction of each batch from general replay corpus

MAX_SEQ_LEN = 1024        # reduce if still OOM; 8B model needs headroom for activations
TIME_BUDGET = 600         # seconds — 10 min for PCIe (vs 5 min for NVLink)
DEVICE_BATCH_SIZE = 1     # per-GPU domain samples; increase only if VRAM allows
GRAD_ACCUM_STEPS = 16     # keep effective batch size reasonable despite small per-GPU batch

# Reject experiment if general BPB degrades more than this vs baseline
MAX_GENERAL_BPB_DEGRADATION = 0.15   # 15% relative limit

# Agent: update this string to describe what you changed before each run
CHANGES_SUMMARY = "baseline fine-tuning run"

# ---------------------------------------------------------------------------
# Distributed process group
# ---------------------------------------------------------------------------

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
torch.manual_seed(42 + local_rank)
torch.cuda.manual_seed(42 + local_rank)


def is_main_process():
    return local_rank == 0


def print_main(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs, flush=True)


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

def _normalize_to_messages(record):
    """
    Normalize a dataset record to the standard messages list format:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

    Handles: messages list, ShareGPT conversations, ChatML text,
    Llama3 text, Alpaca, prompt/response, input/output.
    """
    import re

    # Already in messages list format
    if "messages" in record:
        return record["messages"]

    # ShareGPT format
    if "conversations" in record:
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        return [
            {"role": role_map.get(turn.get("from", ""), turn.get("from", "")),
             "content": turn.get("value", "")}
            for turn in record["conversations"]
        ]

    # text export — pre-rendered chat template string, parse back to messages
    if "text" in record:
        text = record["text"]

        # ChatML format: <|im_start|>role\ncontent<|im_end|>
        if "<|im_start|>" in text:
            parts = re.split(r"<\|im_start\|>(\w+)\n", text)
            messages = []
            for i in range(1, len(parts) - 1, 2):
                role = parts[i]
                content = parts[i + 1].replace("<|im_end|>", "").strip()
                if content and role in ("system", "user", "assistant"):
                    messages.append({"role": role, "content": content})
            if messages:
                return messages

        # Llama-3 format: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
        elif "<|start_header_id|>" in text:
            text = text.replace("<|begin_of_text|>", "")
            text = re.sub(
                r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*$", "", text
            )
            parts = re.split(r"<\|start_header_id\|>(\w+)<\|end_header_id\|>\n\n", text)
            messages = []
            for i in range(1, len(parts) - 1, 2):
                role = parts[i]
                content = parts[i + 1].replace("<|eot_id|>", "").strip()
                if content:
                    messages.append({"role": role, "content": content})
            if messages:
                return messages

        raise KeyError(
            f"Record has 'text' field but could not detect ChatML or Llama3 format. "
            f"First 200 chars: {text[:200]!r}"
        )

    # Alpaca format
    if "instruction" in record and "output" in record:
        user_content = record["instruction"]
        if record.get("input"):
            user_content = f"{user_content}\n\n{record['input']}"
        return [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": record["output"]},
        ]

    # Simple prompt/response pair
    if "prompt" in record and "response" in record:
        return [
            {"role": "user",      "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]},
        ]

    # Simple input/output pair
    if "input" in record and "output" in record:
        return [
            {"role": "user",      "content": record["input"]},
            {"role": "assistant", "content": record["output"]},
        ]

    raise KeyError(
        f"Unrecognized dataset format. Keys found: {list(record.keys())}. "
        "Expected one of: messages, conversations, text (ChatML/Llama3), "
        "instruction+output, prompt+response, input+output."
    )


class DomainDataset(Dataset):
    """Instruction-tuning dataset. Handles multiple JSONL formats automatically."""

    def __init__(self, path, tokenizer):
        self.tokenizer = tokenizer
        self.samples = []
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        self.samples.append(_normalize_to_messages(json.loads(line)))
                    except KeyError:
                        if i == 0:
                            raise  # fail fast on the first line so the error is obvious
                        # skip malformed lines silently after the first

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages = self.samples[idx]

        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"][0]
        labels = input_ids.clone()

        # Mask prompt (user) tokens — only train on assistant responses
        prompt_messages = []
        for m in messages:
            if m["role"] != "assistant":
                prompt_messages.append(m)
            else:
                break  # stop at first assistant turn

        if prompt_messages:
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_enc = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_len = min(prompt_enc["input_ids"].shape[1], input_ids.shape[0])
            labels[:prompt_len] = -100

        return {"input_ids": input_ids, "labels": labels}


class ReplayDataset(Dataset):
    """General-domain text for experience replay. No prompt masking (full LM loss)."""

    def __init__(self, path, tokenizer):
        # Encode in 100K-char paragraphs to avoid the "sequence longer than max" warning
        # that fires when the entire file is tokenized at once.
        chunk_chars = 100_000
        tokens = []
        with open(path, encoding="utf-8") as f:
            while True:
                block = f.read(chunk_chars)
                if not block:
                    break
                tokens.extend(tokenizer.encode(block, add_special_tokens=False))
        self.chunks = [
            tokens[i : i + MAX_SEQ_LEN + 1]
            for i in range(0, len(tokens) - MAX_SEQ_LEN, MAX_SEQ_LEN)
        ]
        if not self.chunks:
            raise ValueError(f"Replay file too short: {path}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        input_ids = torch.tensor(chunk[:MAX_SEQ_LEN], dtype=torch.long)
        labels = torch.tensor(chunk[1 : MAX_SEQ_LEN + 1], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch], batch_first=True, padding_value=-100
    )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": (input_ids != 0).long(),
    }


# ---------------------------------------------------------------------------
# Infinite dataloader
# ---------------------------------------------------------------------------

def infinite_loader(dataset, sampler, batch_size):
    """Infinitely yield batches, reshuffling each epoch."""
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=0,
        )
        yield from loader
        epoch += 1


# ---------------------------------------------------------------------------
# BPB evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, data_path, max_batches=50):
    """
    Compute bits-per-byte on a dataset, aggregated across all DDP ranks.
    data_path: .txt for replay (no masking) or .jsonl for domain (prompt masking).
    """
    is_txt = str(data_path).endswith(".txt")
    dataset = (
        ReplayDataset(data_path, tokenizer)
        if is_txt
        else DomainDataset(data_path, tokenizer)
    )
    loader = DataLoader(
        dataset,
        batch_size=DEVICE_BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0,
    )

    total_nats = 0.0
    total_bytes = 0.0
    steps = 0

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in loader:
            if steps >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = outputs.logits.float()

            # Next-token prediction: shift logits and labels by 1
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss_flat = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            )

            valid_mask = shift_labels.view(-1) != -100
            valid_labels = shift_labels.view(-1)[valid_mask]
            valid_nats = loss_flat[valid_mask]

            # Count UTF-8 bytes per predicted token for BPB denominator
            decoded = [tokenizer.decode([t.item()]) for t in valid_labels]
            byte_counts = torch.tensor(
                [len(s.encode("utf-8")) for s in decoded],
                dtype=torch.float32,
                device=device,
            )

            total_nats += valid_nats.sum().item()
            total_bytes += byte_counts.sum().item()
            steps += 1

    # Aggregate across all DDP ranks
    stats = torch.tensor([total_nats, total_bytes], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_nats_agg, total_bytes_agg = stats[0].item(), stats[1].item()

    return (
        total_nats_agg / (math.log(2) * total_bytes_agg)
        if total_bytes_agg > 0
        else float("inf")
    )


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(model, run_id, keep_last=3):
    """Save LoRA adapter weights only (not the full base model)."""
    checkpoint_dir = Path("checkpoints") / f"run_{run_id}" / "adapter_model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.module.save_pretrained(str(checkpoint_dir))  # model.module = PeftModel
    print_main(f"Checkpoint saved: {checkpoint_dir}")

    # Prune old checkpoints, keep only the most recent `keep_last`
    checkpoints = sorted(
        Path("checkpoints").iterdir(), key=lambda p: p.stat().st_mtime
    )
    for old in checkpoints[:-keep_last]:
        shutil.rmtree(old, ignore_errors=True)
        print_main(f"Removed old checkpoint: {old}")


# ---------------------------------------------------------------------------
# Model + LoRA initialization
# ---------------------------------------------------------------------------

t_start = time.time()
print_main(f"Loading model: {MODEL_NAME}")

# QLoRA: load base model in 4-bit (NF4) to fit in 24 GB alongside DDP buffers.
# 8B model in bfloat16 ≈ 16 GB; in 4-bit ≈ 5-6 GB — leaves headroom for activations.
# LoRA adapters are kept in bfloat16; only adapter grads flow through DDP all-reduce.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": device},
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ANTI-FORGETTING — do not remove: LoRA freezes base weights, only adapters train
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=LORA_TARGET_MODULES,
)
model = get_peft_model(base_model, lora_config)

# Gradient checkpointing: recomputes activations during backward instead of storing them.
# Cuts activation memory ~10x — essential for fitting an 8B model in 24 GB alongside
# DDP gradient buffers. use_reentrant=False is required for PEFT compatibility.
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()  # required when using gradient checkpointing with PEFT

if is_main_process():
    model.print_trainable_parameters()

model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DOMAIN_TRAIN = Path("data/domain/train.jsonl")
DOMAIN_VAL = Path("data/domain/val.jsonl")
REPLAY_TRAIN = Path("data/replay/train.txt")

assert DOMAIN_TRAIN.exists(), (
    f"Missing {DOMAIN_TRAIN}. Run: uv run prepare.py --mode finetune"
)
assert REPLAY_TRAIN.exists(), (
    f"Missing {REPLAY_TRAIN}. Run: uv run prepare.py --mode finetune"
)

domain_dataset = DomainDataset(DOMAIN_TRAIN, tokenizer)
replay_dataset = ReplayDataset(REPLAY_TRAIN, tokenizer)  # ANTI-FORGETTING — do not remove

# ANTI-FORGETTING — do not remove: REPLAY_RATIO controls anti-forgetting mix
replay_batch_size = max(1, round(DEVICE_BATCH_SIZE * REPLAY_RATIO / (1 - REPLAY_RATIO)))

print_main(f"Domain dataset: {len(domain_dataset)} samples")
print_main(f"Replay dataset: {len(replay_dataset)} chunks")
print_main(
    f"Batch per GPU per micro-step: {DEVICE_BATCH_SIZE} domain + {replay_batch_size} replay"
)

domain_sampler = DistributedSampler(
    domain_dataset, num_replicas=world_size, rank=local_rank, shuffle=True, seed=42
)
replay_sampler = DistributedSampler(
    replay_dataset, num_replicas=world_size, rank=local_rank, shuffle=True, seed=42
)

domain_iter = infinite_loader(domain_dataset, domain_sampler, DEVICE_BATCH_SIZE)
replay_iter = infinite_loader(replay_dataset, replay_sampler, replay_batch_size)  # ANTI-FORGETTING — do not remove

# ---------------------------------------------------------------------------
# Optimizer + LR schedule
# ---------------------------------------------------------------------------

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.95),
)


def get_lr_scale(step, total_steps):
    """Linear warmup then cosine decay to 10% of peak LR."""
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cosine * 0.9 + 0.1  # decays from 1.0 → 0.1


# Rough initial estimate for LR scheduler; refines from actual step times
estimated_step_time = 5.0
total_steps_estimate = max(100, int(TIME_BUDGET / estimated_step_time))

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

print_main(
    f"TIME_BUDGET={TIME_BUDGET}s | GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS} | world_size={world_size}"
)
print_main("Starting training...")

t_start_training = None
step = 0
total_training_time = 0.0
smooth_loss = 0.0
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

gc.collect()
gc.freeze()
gc.disable()

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)
    step_loss = 0.0

    for _ in range(GRAD_ACCUM_STEPS):
        domain_batch = next(domain_iter)
        replay_batch = next(replay_iter)  # ANTI-FORGETTING — do not remove

        # Pad domain batch to MAX_SEQ_LEN so it matches the fixed-length replay batch
        # (collate_fn pads domain sequences to the longest in the mini-batch, which is
        # typically shorter than MAX_SEQ_LEN; replay chunks are always exactly MAX_SEQ_LEN)
        d_ids = domain_batch["input_ids"].to(device)
        d_labels = domain_batch["labels"].to(device)
        d_mask = domain_batch["attention_mask"].to(device)
        pad_len = MAX_SEQ_LEN - d_ids.shape[1]
        if pad_len > 0:
            d_ids    = F.pad(d_ids,    (0, pad_len), value=0)
            d_labels = F.pad(d_labels, (0, pad_len), value=-100)
            d_mask   = F.pad(d_mask,   (0, pad_len), value=0)

        # Concatenate domain + replay into a single forward pass
        input_ids = torch.cat([d_ids, replay_batch["input_ids"].to(device)], dim=0)
        labels = torch.cat([d_labels, replay_batch["labels"].to(device)], dim=0)
        attention_mask = torch.cat([d_mask, replay_batch["attention_mask"].to(device)], dim=0)

        with autocast_ctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / GRAD_ACCUM_STEPS

        loss.backward()
        step_loss += loss.item()

    # ANTI-FORGETTING — do not remove: clip gradients to prevent large weight updates
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    lr_scale = get_lr_scale(step, total_steps_estimate)
    for group in optimizer.param_groups:
        group["lr"] = LR * lr_scale

    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step == 0:
        t_start_training = t1
    else:
        total_training_time += dt
        # Refine step-time estimate for LR scheduler
        estimated_step_time = total_training_time / step
        total_steps_estimate = max(100, int(TIME_BUDGET / estimated_step_time))

    ema = 0.9
    smooth_loss = ema * smooth_loss + (1 - ema) * step_loss
    debiased = smooth_loss / (1 - ema ** (step + 1))

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    remaining = max(0, TIME_BUDGET - total_training_time)

    if is_main_process():
        print(
            f"\rstep {step:04d} ({100*progress:.1f}%) | "
            f"loss: {debiased:.4f} | lr: {LR*lr_scale:.2e} | "
            f"dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ",
            end="",
            flush=True,
        )

    step += 1
    # Synchronize the stop decision so ALL ranks exit the loop at the same step.
    # Per-rank wall-clock times diverge slightly on PCIe GPUs; without this,
    # one rank can enter evaluate_bpb() while others are still in the gradient
    # all-reduce, causing a 600-second NCCL timeout and crash.
    should_stop = torch.tensor(
        [1 if (step > 1 and total_training_time >= TIME_BUDGET) else 0],
        device=device,
    )
    dist.all_reduce(should_stop, op=dist.ReduceOp.MAX)
    if should_stop.item():
        break

print_main()  # newline after \r output

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

print_main("Evaluating...")
model.eval()

domain_eval_path = str(DOMAIN_VAL) if DOMAIN_VAL.exists() else str(DOMAIN_TRAIN)
val_bpb_domain = evaluate_bpb(model, tokenizer, domain_eval_path, max_batches=50)

# ANTI-FORGETTING — do not remove: always measure general capability
val_bpb_general = evaluate_bpb(model, tokenizer, str(REPLAY_TRAIN), max_batches=50)

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

# ---------------------------------------------------------------------------
# Anti-forgetting guard
# ---------------------------------------------------------------------------

status = "ACCEPTED"
baseline_general_path = Path("baseline_general_bpb.txt")

if is_main_process():
    # ANTI-FORGETTING — do not remove: enforce general BPB degradation limit
    if not baseline_general_path.exists():
        baseline_general_path.write_text(str(val_bpb_general))
        print_main(f"Baseline general BPB saved: {val_bpb_general:.6f}")
        baseline_general_bpb = val_bpb_general
    else:
        baseline_general_bpb = float(baseline_general_path.read_text().strip())
        degradation = (val_bpb_general - baseline_general_bpb) / max(
            baseline_general_bpb, 1e-9
        )
        if degradation > MAX_GENERAL_BPB_DEGRADATION:
            status = "REJECTED"
            print_main(
                f"REJECTED: general BPB {val_bpb_general:.6f} vs baseline "
                f"{baseline_general_bpb:.6f} ({degradation*100:.1f}% > "
                f"{MAX_GENERAL_BPB_DEGRADATION*100:.0f}% limit)"
            )
        else:
            print_main(
                f"General BPB: {val_bpb_general:.6f} "
                f"(degradation: {degradation*100:+.1f}%)"
            )

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if status == "ACCEPTED" and is_main_process():
    save_checkpoint(model, run_id)

# ---------------------------------------------------------------------------
# Summary + logging
# ---------------------------------------------------------------------------

print_main("---")
print_main(f"val_bpb_domain:   {val_bpb_domain:.6f}")
print_main(f"val_bpb_general:  {val_bpb_general:.6f}")
print_main(f"status:           {status}")
print_main(f"training_seconds: {total_training_time:.1f}")
print_main(f"total_seconds:    {t_end - t_start:.1f}")
print_main(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print_main(f"num_steps:        {step}")

if is_main_process():
    Path("experiments").mkdir(exist_ok=True)
    log_entry = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "val_bpb_domain": round(val_bpb_domain, 6),
        "val_bpb_general": round(val_bpb_general, 6),
        "status": status,
        "changes_summary": CHANGES_SUMMARY,
        "lora_rank": LORA_RANK,
        "lr": LR,
        "replay_ratio": REPLAY_RATIO,
    }
    with open("experiments/log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print_main("Logged to experiments/log.jsonl")

dist.destroy_process_group()
