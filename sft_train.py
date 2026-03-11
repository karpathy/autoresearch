"""
SFT Autoresearch: Otonom hiperparametre arama döngüsü.
Her tur 5 dakika eğitim yapar, val_loss'u kaydeder, en iyiyi tutar.

Model:   vngrs-ai/Kumru-2B-Base
Dataset: oztrkoguz/Open_Math_Instruct_Turkish

Kullanım: uv run sft_train.py
Çıkış:   sft_results.tsv  (tüm deneylerin logu)
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import csv
import gc
import math
import random
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Sabitler — değiştirme
# ---------------------------------------------------------------------------

MODEL_NAME    = "vngrs-ai/Kumru-2B-Base"
DATASET_NAME  = "oztrkoguz/Open_Math_Instruct_Turkish"
RESULTS_FILE  = "sft_results.tsv"

TIME_BUDGET         = 300    # her HP deney turu için eğitim süresi (saniye)
MAX_ROUNDS          = 9999   # sonsuz döngü için büyük bir sayı; Ctrl+C ile dur
FINAL_TRAIN_BUDGET  = 1800   # en iyi HP ile yapılacak final eğitim süresi (saniye, 30 dk)
FINAL_SAVE_DIR      = "sft_best_model"  # final modelin kaydedileceği klasör

# ---------------------------------------------------------------------------
# Varsayılan hiperparametreler (ilk deneyde kullanılır)
# ---------------------------------------------------------------------------

@dataclass
class HParams:
    # --- 16 GB VRAM için ayarlanmış varsayılanlar ---
    lr: float           = 2e-5
    weight_decay: float = 0.01
    total_batch_size: int  = 16   # grad_accum ile karşılanır, VRAM'i etkilemez
    device_batch_size: int = 2    # 16 GB için güvenli mikro-batch
    max_seq_len: int    = 512     # 1024'e çıkabilir ama 512 daha güvenli
    warmup_ratio: float  = 0.05
    warmdown_ratio: float = 0.5
    use_lora: bool      = True    # 16 GB'da full FT sığmaz — LoRA zorunlu
    lora_rank: int      = 16      # ~32 M ek parametre, minimal VRAM
    lora_alpha: int     = 32
    mask_prompt: bool   = True
    grad_checkpoint: bool = True  # aktivasyon yeniden hesaplama — ~%30 VRAM tasarrufu

# Arama uzayı — 16 GB sınırı gözetilerek daraltılmış
SEARCH_SPACE = {
    "lr":                [5e-6, 1e-5, 2e-5, 5e-5, 1e-4],
    "weight_decay":      [0.0, 0.01, 0.05, 0.1],
    "total_batch_size":  [8, 16, 32],        # hepsi grad_accum ile çözülür
    "warmup_ratio":      [0.0, 0.03, 0.05, 0.1],
    "warmdown_ratio":    [0.3, 0.5, 0.7],
    "mask_prompt":       [True, False],
    "lora_rank":         [8, 16, 32, 64],    # 64 hâlâ 16 GB'a sığar
}

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

BF16          = True
ADAM_BETAS    = (0.9, 0.95)
ADAM_EPS      = 1e-8
SYSTEM_PROMPT = "Sen yardımcı bir matematik asistanısın. Türkçe matematik sorularını adım adım çöz."

# ---------------------------------------------------------------------------
# Prompt formatlama
# ---------------------------------------------------------------------------

def format_conversation(example, tokenizer):
    question = example["question"]
    answer   = example["answer"]
    if tokenizer.chat_template is not None:
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    return f"<s>[SYSTEM] {SYSTEM_PROMPT}\n[USER] {question}\n[ASSISTANT] {answer}</s>"


def format_prompt_only(example, tokenizer):
    question = example["question"]
    if tokenizer.chat_template is not None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"<s>[SYSTEM] {SYSTEM_PROMPT}\n[USER] {question}\n[ASSISTANT] "

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MathSFTDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_seq_len, mask_prompt=True):
        self.data = []
        skipped = 0
        for example in hf_dataset:
            full_ids = tokenizer.encode(format_conversation(example, tokenizer), add_special_tokens=False)
            if len(full_ids) < 4:
                skipped += 1
                continue
            full_ids = full_ids[:max_seq_len]
            input_ids = torch.tensor(full_ids, dtype=torch.long)
            labels    = input_ids.clone()
            if mask_prompt:
                prompt_ids = tokenizer.encode(format_prompt_only(example, tokenizer), add_special_tokens=False)
                prompt_len = min(len(prompt_ids), len(full_ids) - 1)
                labels[:prompt_len] = -1
            self.data.append((input_ids, labels))
        print(f"  Dataset: {len(self.data)} örnek ({skipped} atlandı)")

    def __len__(self):  return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


def collate_fn(batch, pad_id):
    inputs, labels = zip(*batch)
    max_len = max(x.size(0) for x in inputs)
    inp_pad = torch.full((len(batch), max_len), pad_id,  dtype=torch.long)
    lbl_pad = torch.full((len(batch), max_len), -1,      dtype=torch.long)
    for i, (inp, lbl) in enumerate(zip(inputs, labels)):
        n = inp.size(0)
        inp_pad[i, :n] = inp
        lbl_pad[i, :n] = lbl
    return inp_pad, lbl_pad

# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def get_lr_multiplier(step, total_steps, warmup_ratio, warmdown_ratio):
    wu = max(1, int(warmup_ratio * total_steps))
    wd = max(1, int(warmdown_ratio * total_steps))
    if step < wu:
        return step / wu
    elif step >= total_steps - wd:
        progress = (total_steps - step) / wd
        return 0.5 * (1 + math.cos(math.pi * (1 - progress)))
    return 1.0


def apply_lora(model, rank, alpha, target_modules):
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank, lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.05, bias="none",
        )
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
        return model
    except ImportError:
        print("  UYARI: peft kurulu değil, LoRA atlandı.")
        return model


@torch.no_grad()
def evaluate(model, loader, device, autocast_ctx, max_batches=80):
    model.eval()
    total_loss, total_tokens, n = 0.0, 0, 0
    for inp, lbl in loader:
        if n >= max_batches:
            break
        inp = inp.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)
        with autocast_ctx:
            logits = model(input_ids=inp).logits
            sl = logits[:, :-1, :].contiguous()
            tl = lbl[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), tl.view(-1), ignore_index=-1, reduction='sum')
        total_loss   += loss.item()
        total_tokens += (tl != -1).sum().item()
        n += 1
    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def mutate(hp: HParams, rng: random.Random) -> HParams:
    """Mevcut en iyi hiperparametreleri biraz değiştir."""
    new_hp = copy.copy(hp)
    k = rng.choice([1, 1, 2])
    keys = rng.sample(list(SEARCH_SPACE.keys()), k)
    for key in keys:
        setattr(new_hp, key, rng.choice(SEARCH_SPACE[key]))
    # 16 GB sınırı: use_lora her zaman True kalmalı
    new_hp.use_lora = True
    # device_batch_size, total_batch_size'ı geçemez
    if new_hp.device_batch_size > new_hp.total_batch_size:
        new_hp.device_batch_size = new_hp.total_batch_size
    # total_batch_size, device_batch_size'ın katı olmalı
    if new_hp.total_batch_size % new_hp.device_batch_size != 0:
        new_hp.device_batch_size = 1
    return new_hp


def log_result(round_idx, hp, val_loss, num_steps, peak_vram_mb, training_sec, status, description):
    row = {
        "round":           round_idx,
        "val_loss":        f"{val_loss:.6f}" if not math.isinf(val_loss) else "inf",
        "status":          status,
        "training_sec":    f"{training_sec:.1f}",
        "num_steps":       num_steps,
        "peak_vram_mb":    f"{peak_vram_mb:.0f}",
        "lr":              hp.lr,
        "weight_decay":    hp.weight_decay,
        "total_batch":     hp.total_batch_size,
        "max_seq_len":     hp.max_seq_len,
        "warmup_ratio":    hp.warmup_ratio,
        "warmdown_ratio":  hp.warmdown_ratio,
        "use_lora":        hp.use_lora,
        "lora_rank":       hp.lora_rank,
        "lora_alpha":      hp.lora_alpha,
        "grad_checkpoint": hp.grad_checkpoint,
        "mask_prompt":     hp.mask_prompt,
        "description":     description,
    }
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ---------------------------------------------------------------------------
# Tek deney: modeli yükle, eğit, değerlendir
# ---------------------------------------------------------------------------

def run_experiment(hp: HParams, tokenizer, train_dataset, val_dataset, device, base_model_name):
    """Verilen HP ile 5 dakika SFT eğitimi yapar. (val_loss, num_steps, peak_vram_mb, training_sec) döner."""

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if BF16 \
                   else torch.amp.autocast(device_type="cuda", enabled=False)

    # Her deney için modeli sıfırdan yükle (ağırlıklar base modelden)
    print(f"  Model yükleniyor: {base_model_name}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if BF16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  Model yüklendi ({time.time()-t0:.1f}s)")

    # Gradient checkpointing: aktivasyonları forward'da saklamak yerine
    # backward'da yeniden hesaplar → ~%30 VRAM tasarrufu, ~%20 yavaşlama
    if hp.grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("  Gradient checkpointing: aktif")

    if hp.use_lora:
        model = apply_lora(model, hp.lora_rank, hp.lora_alpha,
                           ["c_q", "c_k", "c_v", "c_proj", "c_fc"])

    # DataLoader
    pad_id   = tokenizer.pad_token_id
    _collate = lambda b: collate_fn(b, pad_id)
    train_loader = DataLoader(train_dataset, batch_size=hp.device_batch_size,
                              shuffle=True,  collate_fn=_collate, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=hp.device_batch_size,
                              shuffle=False, collate_fn=_collate, pin_memory=True, drop_last=False)

    grad_accum = max(1, hp.total_batch_size // hp.device_batch_size)

    params     = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.AdamW(params, lr=hp.lr, betas=ADAM_BETAS,
                                   eps=ADAM_EPS, weight_decay=hp.weight_decay)
    for g in optimizer.param_groups:
        g["initial_lr"] = g["lr"]

    # Eğitim döngüsü
    model.train()
    train_iter = iter(train_loader)

    def next_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    total_train_time = 0.0
    smooth_loss      = 0.0
    steps_per_sec    = 0.0
    total_steps_est  = 100
    step             = 0

    torch.cuda.reset_peak_memory_stats(device)
    x, y = next_batch()

    while True:
        torch.cuda.synchronize()
        t_step = time.time()

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(grad_accum):
            x_gpu = x.to(device, non_blocking=True)
            y_gpu = y.to(device, non_blocking=True)
            with autocast_ctx:
                logits = model(input_ids=x_gpu).logits
                sl     = logits[:, :-1, :].contiguous()
                tl     = y_gpu[:, 1:].contiguous()
                loss   = F.cross_entropy(sl.view(-1, sl.size(-1)), tl.view(-1), ignore_index=-1)
            accum_loss += loss.detach().item() / grad_accum
            (loss / grad_accum).backward()
            x, y = next_batch()

        # LR güncelle
        progress = min(total_train_time / TIME_BUDGET, 1.0)
        if step > 5 and steps_per_sec > 0:
            total_steps_est = max(step + 1, int(steps_per_sec * TIME_BUDGET))
        lrm = get_lr_multiplier(step, total_steps_est, hp.warmup_ratio, hp.warmdown_ratio)
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * lrm

        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # NaN / patlama kontrolü
        if math.isnan(accum_loss) or accum_loss > 100:
            print(f"\n  FAIL: loss={accum_loss:.4f}")
            del model
            torch.cuda.empty_cache()
            return float("inf"), step, 0.0, total_train_time

        torch.cuda.synchronize()
        dt = time.time() - t_step

        if step > 5:
            total_train_time += dt
            steps_per_sec = (step - 5) / total_train_time if total_train_time > 0 else 0

        ema_b       = 0.9
        smooth_loss = ema_b * smooth_loss + (1 - ema_b) * accum_loss
        debiased    = smooth_loss / (1 - ema_b ** (step + 1))
        remaining   = max(0, TIME_BUDGET - total_train_time)

        print(
            f"\r  step {step:04d} ({100*progress:.1f}%) | "
            f"loss {debiased:.4f} | lr {optimizer.param_groups[0]['lr']:.1e} | "
            f"dt {dt*1000:.0f}ms | kalan {remaining:.0f}s    ",
            end="", flush=True,
        )

        if step == 0:
            gc.collect(); gc.freeze(); gc.disable()

        step += 1
        if step > 10 and total_train_time >= TIME_BUDGET:
            break

    print()

    # Değerlendirme
    with autocast_ctx:
        val_loss = evaluate(model, val_loader, device, autocast_ctx)

    peak_vram = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    del model
    gc.enable()
    gc.collect()
    torch.cuda.empty_cache()
    gc.disable()

    return val_loss, step, peak_vram, total_train_time

# ---------------------------------------------------------------------------
# Final eğitim: en iyi HP ile uzun süre eğit, modeli kaydet
# ---------------------------------------------------------------------------

def run_final_training(hp: HParams, tokenizer, train_dataset, val_dataset, device,
                       base_model_name, time_budget, save_dir):
    """
    En iyi HP ile uzun süreli eğitim yapar ve modeli diske kaydeder.
    Seçenek A: HP arama bittikten sonra tek seferlik çalışır.
    """
    print(f"\n{'=' * 60}")
    print(f"FINAL EĞİTİM — {time_budget // 60} dakika")
    print(f"HP: lr={hp.lr}, lora_rank={hp.lora_rank}, batch={hp.total_batch_size}")
    print(f"Kayıt: {save_dir}")
    print('=' * 60)

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if BF16 \
                   else torch.amp.autocast(device_type="cuda", enabled=False)

    print(f"\n  Model yükleniyor: {base_model_name}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if BF16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  Model yüklendi ({time.time()-t0:.1f}s)")

    if hp.grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if hp.use_lora:
        model = apply_lora(model, hp.lora_rank, hp.lora_alpha,
                           ["c_q", "c_k", "c_v", "c_proj", "c_fc"])

    pad_id   = tokenizer.pad_token_id
    _collate = lambda b: collate_fn(b, pad_id)
    train_loader = DataLoader(train_dataset, batch_size=hp.device_batch_size,
                              shuffle=True,  collate_fn=_collate, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=hp.device_batch_size,
                              shuffle=False, collate_fn=_collate, pin_memory=True, drop_last=False)

    grad_accum = max(1, hp.total_batch_size // hp.device_batch_size)
    params     = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.AdamW(params, lr=hp.lr, betas=ADAM_BETAS,
                                   eps=ADAM_EPS, weight_decay=hp.weight_decay)
    for g in optimizer.param_groups:
        g["initial_lr"] = g["lr"]

    model.train()
    train_iter = iter(train_loader)

    def next_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    total_train_time = 0.0
    smooth_loss      = 0.0
    steps_per_sec    = 0.0
    total_steps_est  = 500
    best_val_loss    = float("inf")
    step             = 0

    # Checkpoint kaydetme — her N dakikada bir
    checkpoint_interval = 300  # 5 dakikada bir ara kayıt
    last_checkpoint_time = 0.0

    torch.cuda.reset_peak_memory_stats(device)
    x, y = next_batch()

    print(f"  Eğitim başlıyor ({time_budget}s)...\n")
    while True:
        torch.cuda.synchronize()
        t_step = time.time()

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(grad_accum):
            x_gpu = x.to(device, non_blocking=True)
            y_gpu = y.to(device, non_blocking=True)
            with autocast_ctx:
                logits = model(input_ids=x_gpu).logits
                sl = logits[:, :-1, :].contiguous()
                tl = y_gpu[:, 1:].contiguous()
                loss = F.cross_entropy(sl.view(-1, sl.size(-1)), tl.view(-1), ignore_index=-1)
            accum_loss += loss.detach().item() / grad_accum
            (loss / grad_accum).backward()
            x, y = next_batch()

        # LR güncelle
        progress = min(total_train_time / time_budget, 1.0)
        if step > 5 and steps_per_sec > 0:
            total_steps_est = max(step + 1, int(steps_per_sec * time_budget))
        lrm = get_lr_multiplier(step, total_steps_est, hp.warmup_ratio, hp.warmdown_ratio)
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * lrm

        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        if math.isnan(accum_loss) or accum_loss > 100:
            print(f"\n  FAIL: loss={accum_loss:.4f}, eğitim durduruluyor.")
            break

        torch.cuda.synchronize()
        dt = time.time() - t_step

        if step > 5:
            total_train_time += dt
            steps_per_sec = (step - 5) / total_train_time if total_train_time > 0 else 0

        ema_b       = 0.9
        smooth_loss = ema_b * smooth_loss + (1 - ema_b) * accum_loss
        debiased    = smooth_loss / (1 - ema_b ** (step + 1))
        remaining   = max(0, time_budget - total_train_time)
        elapsed_min = total_train_time / 60

        print(
            f"\r  step {step:05d} | {elapsed_min:.1f}/{time_budget//60}dk | "
            f"loss {debiased:.4f} | lr {optimizer.param_groups[0]['lr']:.1e} | "
            f"kalan {remaining:.0f}s    ",
            end="", flush=True,
        )

        if step == 0:
            gc.collect(); gc.freeze(); gc.disable()

        # Ara değerlendirme + checkpoint (her 5 dakikada bir)
        if step > 10 and total_train_time - last_checkpoint_time >= checkpoint_interval:
            print()
            with autocast_ctx:
                val_loss = evaluate(model, val_loader, device, autocast_ctx, max_batches=50)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            print(f"  [Ara değerlendirme] val_loss={val_loss:.6f} "
                  f"{'✓ en iyi, kaydediliyor...' if is_best else f'(en iyi: {best_val_loss:.6f})'}")
            if is_best:
                _save_model(model, tokenizer, save_dir + "_best", hp)
            last_checkpoint_time = total_train_time
            model.train()

        step += 1
        if step > 10 and total_train_time >= time_budget:
            break

    print()

    # Final değerlendirme
    print("  Final değerlendirme...")
    with autocast_ctx:
        final_val_loss = evaluate(model, val_loader, device, autocast_ctx, max_batches=100)

    peak_vram = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    # Final modeli kaydet
    _save_model(model, tokenizer, save_dir, hp)

    print(f"\n{'=' * 60}")
    print("FINAL EĞİTİM TAMAMLANDI")
    print(f"  val_loss:      {final_val_loss:.6f}")
    print(f"  val_ppl:       {math.exp(min(final_val_loss, 20)):.4f}")
    print(f"  num_steps:     {step}")
    print(f"  train_sec:     {total_train_time:.1f}s")
    print(f"  peak_vram_mb:  {peak_vram:.0f}")
    print(f"  model_dir:     {save_dir}")
    print(f"  best_ckpt_dir: {save_dir}_best")
    print('=' * 60)

    del model
    gc.enable(); gc.collect(); torch.cuda.empty_cache(); gc.disable()

    return final_val_loss


def _save_model(model, tokenizer, save_dir, hp):
    """Modeli ve tokenizer'ı diske yazar, HP'yi de not olarak ekler."""
    import json
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "training_hp.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(hp), f, indent=2)
    print(f"  → Kaydedildi: {save_dir}")


# ---------------------------------------------------------------------------
# Bir kez yüklenen kaynaklar (tokenizer + dataset)
# ---------------------------------------------------------------------------

print("=" * 60)
print("SFT Autoresearch — Otonom hiperparametre arama")
print("=" * 60)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

print(f"\nTokenizer yükleniyor: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token or "<|pad|>"
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0
print(f"  Vocab: {tokenizer.vocab_size:,} | pad: '{tokenizer.pad_token}'")
if tokenizer.chat_template:
    print("  Chat template: mevcut")

print(f"\nDataset yükleniyor: {DATASET_NAME}")
raw_ds     = load_dataset(DATASET_NAME, split="train")
split_ds   = raw_ds.train_test_split(test_size=0.05, seed=42)
train_raw  = split_ds["train"]
val_raw    = split_ds["test"]
print(f"  Train: {len(train_raw):,} | Val: {len(val_raw):,}")

# Varsayılan HP ile dataset'i bir kez tokenize et
default_hp = HParams()
print("\nDataset tokenize ediliyor (başlangıç HP ile)...")
train_dataset = MathSFTDataset(train_raw, tokenizer, default_hp.max_seq_len, mask_prompt=default_hp.mask_prompt)
val_dataset   = MathSFTDataset(val_raw,   tokenizer, default_hp.max_seq_len, mask_prompt=False)

# ---------------------------------------------------------------------------
# Otonom arama döngüsü
# ---------------------------------------------------------------------------

rng         = random.Random(42)
best_loss   = float("inf")
best_hp     = default_hp
current_hp  = default_hp

print(f"\nSonuçlar: {RESULTS_FILE}")
print(f"Dur için: Ctrl+C\n")
print("=" * 60)

for round_idx in range(1, MAX_ROUNDS + 1):
    print(f"\n[Tur {round_idx}] Hiperparametreler:")
    for k, v in asdict(current_hp).items():
        marker = " ← değişti" if round_idx > 1 and getattr(best_hp, k) != v else ""
        print(f"  {k:20s} = {v}{marker}")
    print()

    t_round_start = time.time()
    try:
        val_loss, num_steps, peak_vram, train_sec = run_experiment(
            current_hp, tokenizer, train_dataset, val_dataset, device, MODEL_NAME
        )
        status = "ok"
    except torch.cuda.OutOfMemoryError:
        print("\n  OOM hatası!")
        val_loss, num_steps, peak_vram, train_sec = float("inf"), 0, 0.0, 0.0
        status = "oom"
        torch.cuda.empty_cache()
        gc.collect()
    except KeyboardInterrupt:
        print("\nDurduruldu.")
        break
    except Exception as e:
        print(f"\n  Hata: {e}")
        val_loss, num_steps, peak_vram, train_sec = float("inf"), 0, 0.0, 0.0
        status = f"error"

    improved = val_loss < best_loss
    if improved:
        best_loss = val_loss
        best_hp   = current_hp

    # Açıklama
    if round_idx == 1:
        description = "baseline"
    elif improved:
        description = "improved"
    else:
        description = "reverted"

    log_result(round_idx, current_hp, val_loss, num_steps, peak_vram, train_sec, status, description)

    print(f"\n[Tur {round_idx}] Sonuç:")
    print(f"  val_loss:    {val_loss:.6f}  {'✓ YENİ EN İYİ' if improved else f'(en iyi: {best_loss:.6f})'}")
    print(f"  num_steps:   {num_steps}")
    print(f"  peak_vram:   {peak_vram:.0f} MB")
    print(f"  train_sec:   {train_sec:.1f}s")
    print(f"  toplam_süre: {time.time()-t_round_start:.1f}s (yükleme dahil)")
    print(f"  durum:       {status}")

    # Sonraki HP: başarılıysa mutasyona uğrat, başarısızsa en iyiden mutasyon
    if status == "oom":
        # OOM durumunda batch boyutunu küçült
        next_hp = copy.copy(best_hp)
        next_hp.device_batch_size = max(1, best_hp.device_batch_size // 2)
        next_hp.total_batch_size  = max(next_hp.device_batch_size, best_hp.total_batch_size // 2)
        print(f"  → OOM: batch boyutu düşürüldü ({next_hp.device_batch_size}/{next_hp.total_batch_size})")
    else:
        next_hp = mutate(best_hp, rng)
        changed = [k for k in asdict(next_hp) if getattr(next_hp, k) != getattr(best_hp, k)]
        print(f"  → Sonraki tur: {', '.join(changed) if changed else 'aynı HP (yeni rastgele tohum)'} değiştirildi")

    current_hp = next_hp

    # Dataset'i yeni seq_len ile yeniden oluştur (gerekirse)
    if current_hp.max_seq_len != train_dataset.data[0][0].size(0) if train_dataset.data else True:
        if current_hp.max_seq_len != default_hp.max_seq_len:
            print(f"\n  max_seq_len değişti ({default_hp.max_seq_len} → {current_hp.max_seq_len}), dataset yeniden tokenize ediliyor...")
            train_dataset = MathSFTDataset(train_raw, tokenizer, current_hp.max_seq_len, mask_prompt=current_hp.mask_prompt)
            val_dataset   = MathSFTDataset(val_raw,   tokenizer, current_hp.max_seq_len, mask_prompt=False)
            default_hp    = current_hp  # referans seq_len'i güncelle

    print("-" * 60)

print(f"\n{'=' * 60}")
print(f"HP ARAMASI TAMAMLANDI")
print(f"  Toplam tur:      {round_idx}")
print(f"  En iyi val_loss: {best_loss:.6f}")
print(f"  En iyi HP:")
for k, v in asdict(best_hp).items():
    print(f"    {k:20s} = {v}")
print(f"  Sonuçlar:        {RESULTS_FILE}")
print('=' * 60)

# ---------------------------------------------------------------------------
# Seçenek A: En iyi HP ile uzun süreli final eğitim
# ---------------------------------------------------------------------------

# Dataset'i en iyi HP'nin seq_len'ine göre yeniden oluştur
if best_hp.max_seq_len != train_dataset.data[0][0].size(0) if train_dataset.data else True:
    print(f"\nFinal eğitim için dataset yeniden tokenize ediliyor (seq_len={best_hp.max_seq_len})...")
    train_dataset = MathSFTDataset(train_raw, tokenizer, best_hp.max_seq_len, mask_prompt=best_hp.mask_prompt)
    val_dataset   = MathSFTDataset(val_raw,   tokenizer, best_hp.max_seq_len, mask_prompt=False)

gc.enable(); gc.collect(); torch.cuda.empty_cache()

run_final_training(
    hp             = best_hp,
    tokenizer      = tokenizer,
    train_dataset  = train_dataset,
    val_dataset    = val_dataset,
    device         = device,
    base_model_name= MODEL_NAME,
    time_budget    = FINAL_TRAIN_BUDGET,
    save_dir       = FINAL_SAVE_DIR,
)
