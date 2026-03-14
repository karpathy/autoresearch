import os
import math
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# argument parser
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--block_size", type=int, default=1024)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--max_iters", type=int, default=10000)
parser.add_argument("--eval_interval", type=int, default=500)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--dataset", type=str, default="data")
parser.add_argument("--compile", action="store_true")

args = parser.parse_args()

# -----------------------------------------------------------------------------
# deterministic seeding (NEW)
# -----------------------------------------------------------------------------

random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# -----------------------------------------------------------------------------
# safe device initialization (FIX)
# -----------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# dataset loader
# -----------------------------------------------------------------------------

data_dir = args.dataset

train_data = torch.from_numpy(
    torch.load(os.path.join(data_dir, "train.bin"), map_location="cpu")
)
val_data = torch.from_numpy(
    torch.load(os.path.join(data_dir, "val.bin"), map_location="cpu")
)

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([data[i:i+args.block_size] for i in ix])
    y = torch.stack([data[i+1:i+args.block_size+1] for i in ix])
    return x.to(device), y.to(device)

# -----------------------------------------------------------------------------
# rotary embeddings (FIX: avoid overwriting buffers)
# -----------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):

    def __init__(self, dim, max_position_embeddings=2048):
        super().__init__()

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2).float() / dim)
        )

        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_position_embeddings).float()
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x):
        seq_len = x.shape[1]

        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        return cos, sin

# -----------------------------------------------------------------------------
# simple transformer block
# -----------------------------------------------------------------------------

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd)
        )

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out

        h = self.ln2(x)
        x = x + self.ff(h)

        return x

# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------

class GPT(nn.Module):

    def __init__(self, vocab_size, n_layer=6, n_head=6, n_embd=384):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(args.block_size, n_embd)

        self.blocks = nn.ModuleList([
            Block(n_embd, n_head) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))

        x = tok + pos

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss


# -----------------------------------------------------------------------------
# vocab detection
# -----------------------------------------------------------------------------

meta_path = os.path.join(data_dir, "meta.pkl")

if os.path.exists(meta_path):
    import pickle
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]
else:
    vocab_size = 50304

print(f"vocab size: {vocab_size}")

model = GPT(vocab_size)
model = model.to(device)

if args.compile:
    model = torch.compile(model)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate
)

# -----------------------------------------------------------------------------
# evaluation
# -----------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss():

    model.eval()
    out = {}

    for split in ["train", "val"]:

        losses = torch.zeros(10)

        for k in range(10):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out

# -----------------------------------------------------------------------------
# training loop
# -----------------------------------------------------------------------------

print("Starting training...")

t0 = time.time()
tokens = 0

for iter in range(args.max_iters):

    if iter % args.eval_interval == 0:

        losses = estimate_loss()

        print(
            f"step {iter} | "
            f"train loss {losses['train']:.4f} | "
            f"val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    # -----------------------------------------------------------------------------
    # loss stability check (FIX)
    # -----------------------------------------------------------------------------

    if not math.isfinite(loss.item()):
        print("Loss became NaN or inf. Stopping.")
        break

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    tokens += xb.numel()

    # -----------------------------------------------------------------------------
    # logging stability (FIX)
    # -----------------------------------------------------------------------------

    dt = time.time() - t0
    t0 = time.time()

    tok_per_sec = tokens / dt if dt > 0 else 0
    tokens = 0

    if iter % 100 == 0:
        print(
            f"iter {iter} | "
            f"loss {loss.item():.4f} | "
            f"tok/s {tok_per_sec:.2f}"
        )

print("Training finished.")
