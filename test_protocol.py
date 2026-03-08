"""
DOJO adversarial test protocol.
This is the file the agent modifies in DOJO mode.
Each test probes the trained model for a specific failure mode.

Usage: imported by run_dojo.py — not run directly.
"""

import math
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from prepare import MAX_SEQ_LEN, get_token_bytes, make_dataloader


@dataclass
class TestResult:
    test_name: str
    adversarial_bpb: float
    baseline_bpb: float
    robustness_gap: float
    description: str
    num_tokens_tested: int


def compute_bpb_on_sequences(model, input_ids, target_ids):
    """
    Compute bits-per-byte on arbitrary token sequences.
    Mirrors evaluate_bpb() from prepare.py but on provided inputs.
    Returns (bpb, num_tokens_tested) or (float('inf'), 0) if no valid tokens.
    """
    token_bytes = get_token_bytes()
    loss_flat = model(input_ids, target_ids, reduction="none").reshape(-1)
    y_flat = target_ids.reshape(-1)
    nbytes = mx.take(token_bytes, y_flat, axis=0)
    mask = nbytes > 0
    total_nats = mx.sum(loss_flat * mask).item()
    total_bytes = int(mx.sum(nbytes).item())
    num_tokens = int(mx.sum(mask).item())
    if total_bytes == 0:
        return float("inf"), 0
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb, num_tokens


# ---------------------------------------------------------------------------
# Test 1: Memorization leakage
# ---------------------------------------------------------------------------

def test_memorization_leakage(model, tokenizer, baseline_bpb, time_budget):
    """
    Measures whether the model memorized training data.
    Feeds real training prefixes and checks if the model predicts continuations
    with lower loss than on validation data. A negative gap (train << val) means
    the model memorized. A large positive gap is also interesting — it means
    the model generalizes better than it memorizes.

    We compare training BPB vs baseline (val) BPB.
    """
    t0 = time.time()
    train_loader = make_dataloader(tokenizer, 32, MAX_SEQ_LEN, "train")
    val_loader = make_dataloader(tokenizer, 32, MAX_SEQ_LEN, "val")

    train_nats, train_bytes, val_nats, val_bytes = 0.0, 0, 0.0, 0
    token_bytes = get_token_bytes()
    steps = 0

    while time.time() - t0 < time_budget:
        # Training data
        x_train, y_train, _ = next(train_loader)
        loss_t = model(x_train, y_train, reduction="none").reshape(-1)
        yt_flat = y_train.reshape(-1)
        nb_t = mx.take(token_bytes, yt_flat, axis=0)
        mask_t = nb_t > 0
        train_nats += mx.sum(loss_t * mask_t).item()
        train_bytes += int(mx.sum(nb_t).item())

        # Validation data
        x_val, y_val, _ = next(val_loader)
        loss_v = model(x_val, y_val, reduction="none").reshape(-1)
        yv_flat = y_val.reshape(-1)
        nb_v = mx.take(token_bytes, yv_flat, axis=0)
        mask_v = nb_v > 0
        val_nats += mx.sum(loss_v * mask_v).item()
        val_bytes += int(mx.sum(nb_v).item())

        mx.eval()
        steps += 1

    if train_bytes == 0 or val_bytes == 0:
        return TestResult("memorization_leakage", baseline_bpb, baseline_bpb, 0.0,
                          "insufficient data", 0)

    train_bpb = train_nats / (math.log(2) * train_bytes)
    val_bpb = val_nats / (math.log(2) * val_bytes)

    # Memorization gap: how much better the model is on train vs val
    # If train_bpb << val_bpb, the model memorized
    memorization_gap = val_bpb - train_bpb
    # For robustness_gap metric, we report val_bpb as adversarial (it's the worse case)
    adversarial_bpb = max(val_bpb, train_bpb)

    return TestResult(
        test_name="memorization_leakage",
        adversarial_bpb=adversarial_bpb,
        baseline_bpb=baseline_bpb,
        robustness_gap=adversarial_bpb - baseline_bpb,
        description=f"train_bpb={train_bpb:.3f} val_bpb={val_bpb:.3f} memorization_gap={memorization_gap:.3f}",
        num_tokens_tested=steps * 32 * MAX_SEQ_LEN,
    )


# ---------------------------------------------------------------------------
# Test 2: Subgroup disparity (text LM equivalent of demographic bias)
# ---------------------------------------------------------------------------

def test_subgroup_disparity(model, tokenizer, baseline_bpb, time_budget):
    """
    Partitions validation data by token characteristics and finds the subgroup
    where the model performs worst. val_bpb is an average — this finds the
    populations that average hides.

    Subgroups: high-digit-density sequences, low-entropy (repetitive) sequences,
    high-entropy (diverse/chaotic) sequences.
    """
    t0 = time.time()
    val_loader = make_dataloader(tokenizer, 16, MAX_SEQ_LEN, "val")
    token_bytes = get_token_bytes()
    vocab_size = tokenizer.get_vocab_size()

    # Track per-subgroup stats
    subgroups = {
        "high_digit_density": {"nats": 0.0, "bytes": 0, "count": 0},
        "low_entropy": {"nats": 0.0, "bytes": 0, "count": 0},
        "high_entropy": {"nats": 0.0, "bytes": 0, "count": 0},
        "normal": {"nats": 0.0, "bytes": 0, "count": 0},
    }

    steps = 0
    while time.time() - t0 < time_budget:
        x, y, _ = next(val_loader)
        loss = model(x, y, reduction="none")  # (batch, seq_len)
        mx.eval(loss)

        # Classify each sequence in the batch
        for i in range(x.shape[0]):
            seq = x[i]  # (seq_len,)
            seq_np = np.array(seq)
            y_i = y[i:i+1]
            loss_i = loss[i:i+1].reshape(-1)
            y_flat = y_i.reshape(-1)
            nb = mx.take(token_bytes, y_flat, axis=0)
            mask = nb > 0
            nats = mx.sum(loss_i * mask).item()
            nbytes = int(mx.sum(nb).item())

            # Classify by token statistics
            unique_ratio = len(np.unique(seq_np)) / len(seq_np)
            # Rough digit detection: tokens in lower vocab range tend to be
            # single chars/digits. Check how many are < 256 (byte-level tokens)
            byte_token_ratio = np.mean(seq_np < 256)

            if byte_token_ratio > 0.3:
                group = "high_digit_density"
            elif unique_ratio < 0.3:
                group = "low_entropy"
            elif unique_ratio > 0.7:
                group = "high_entropy"
            else:
                group = "normal"

            subgroups[group]["nats"] += nats
            subgroups[group]["bytes"] += nbytes
            subgroups[group]["count"] += 1

        steps += 1

    # Find worst subgroup
    worst_group = None
    worst_bpb = 0.0
    group_details = []
    for name, stats in subgroups.items():
        if stats["bytes"] > 0:
            bpb = stats["nats"] / (math.log(2) * stats["bytes"])
            group_details.append(f"{name}={bpb:.3f}(n={stats['count']})")
            if bpb > worst_bpb:
                worst_bpb = bpb
                worst_group = name

    if worst_group is None:
        return TestResult("subgroup_disparity", baseline_bpb, baseline_bpb, 0.0,
                          "no valid subgroups", 0)

    return TestResult(
        test_name="subgroup_disparity",
        adversarial_bpb=worst_bpb,
        baseline_bpb=baseline_bpb,
        robustness_gap=worst_bpb - baseline_bpb,
        description=f"worst={worst_group} {' '.join(group_details)}",
        num_tokens_tested=steps * 16 * MAX_SEQ_LEN,
    )


# ---------------------------------------------------------------------------
# Test 3: Window boundary exploit
# ---------------------------------------------------------------------------

def test_window_boundary_exploit(model, tokenizer, baseline_bpb, time_budget):
    """
    Tests whether the SSSL sliding window attention creates blind spots.
    Creates sequences where a distinctive pattern appears early, then checks
    if the model can use that information at positions beyond the window size.

    The model uses window_size = seq_len // 2 for 'S' layers. We place
    information just beyond that boundary and measure prediction quality.
    """
    t0 = time.time()
    token_bytes = get_token_bytes()
    vocab_size = tokenizer.get_vocab_size()
    window_size = MAX_SEQ_LEN // 2  # Short window size from SSSL pattern

    # Strategy: Create sequences where a repeated marker token appears at the start,
    # then fill with random tokens, and check if loss is higher at positions
    # beyond the window boundary vs. within it.

    within_window_nats, within_window_bytes = 0.0, 0
    beyond_window_nats, beyond_window_bytes = 0.0, 0
    batch_size = 16
    steps = 0

    while time.time() - t0 < time_budget:
        # Create random sequences
        seq = mx.random.randint(4, vocab_size, shape=(batch_size, MAX_SEQ_LEN + 1))
        # Place a distinctive repeated pattern in positions 0-50
        marker = mx.random.randint(4, vocab_size, shape=(1,)).item()
        marker_pattern = mx.full((batch_size, 50), marker, dtype=mx.int32)
        seq_list = []
        for b in range(batch_size):
            row = mx.concatenate([marker_pattern[b], seq[b, 50:]])
            seq_list.append(row)
        seq = mx.stack(seq_list)

        inputs = seq[:, :-1]
        targets = seq[:, 1:]

        loss = model(inputs, targets, reduction="none")  # (batch, seq_len)
        mx.eval(loss)

        # Split loss by position relative to window boundary
        y_flat_within = targets[:, :window_size].reshape(-1)
        y_flat_beyond = targets[:, window_size:].reshape(-1)
        loss_within = loss[:, :window_size].reshape(-1)
        loss_beyond = loss[:, window_size:].reshape(-1)

        nb_within = mx.take(token_bytes, y_flat_within, axis=0)
        nb_beyond = mx.take(token_bytes, y_flat_beyond, axis=0)
        mask_w = nb_within > 0
        mask_b = nb_beyond > 0

        within_window_nats += mx.sum(loss_within * mask_w).item()
        within_window_bytes += int(mx.sum(nb_within).item())
        beyond_window_nats += mx.sum(loss_beyond * mask_b).item()
        beyond_window_bytes += int(mx.sum(nb_beyond).item())

        steps += 1

    if within_window_bytes == 0 or beyond_window_bytes == 0:
        return TestResult("window_boundary_exploit", baseline_bpb, baseline_bpb, 0.0,
                          "insufficient data", 0)

    within_bpb = within_window_nats / (math.log(2) * within_window_bytes)
    beyond_bpb = beyond_window_nats / (math.log(2) * beyond_window_bytes)
    adversarial_bpb = beyond_bpb  # Beyond window is the adversarial case

    return TestResult(
        test_name="window_boundary_exploit",
        adversarial_bpb=adversarial_bpb,
        baseline_bpb=baseline_bpb,
        robustness_gap=adversarial_bpb - baseline_bpb,
        description=f"within_window={within_bpb:.3f} beyond_window={beyond_bpb:.3f} boundary_gap={beyond_bpb - within_bpb:.3f}",
        num_tokens_tested=steps * batch_size * MAX_SEQ_LEN,
    )


# ---------------------------------------------------------------------------
# Test 4: Noise robustness
# ---------------------------------------------------------------------------

def test_noise_robustness(model, tokenizer, baseline_bpb, time_budget):
    """
    Measures how fragile the model is to minor input perturbations.
    Takes validation sequences and applies token swaps at various rates,
    then measures BPB degradation. A robust model degrades gracefully;
    a fragile model collapses.
    """
    t0 = time.time()
    val_loader = make_dataloader(tokenizer, 16, MAX_SEQ_LEN, "val")
    token_bytes = get_token_bytes()
    vocab_size = tokenizer.get_vocab_size()

    # Test multiple noise levels
    noise_rates = [0.01, 0.05, 0.10, 0.20]
    noise_stats = {rate: {"nats": 0.0, "bytes": 0} for rate in noise_rates}
    clean_nats, clean_bytes = 0.0, 0
    steps = 0

    while time.time() - t0 < time_budget:
        x, y, _ = next(val_loader)
        mx.eval(x, y)

        # Clean baseline for this batch
        loss_clean = model(x, y, reduction="none").reshape(-1)
        y_flat = y.reshape(-1)
        nb = mx.take(token_bytes, y_flat, axis=0)
        mask = nb > 0
        clean_nats += mx.sum(loss_clean * mask).item()
        clean_bytes += int(mx.sum(nb).item())

        # Noisy versions
        for rate in noise_rates:
            # Randomly replace tokens in input
            noise_mask = mx.random.uniform(shape=x.shape) < rate
            random_tokens = mx.random.randint(4, vocab_size, shape=x.shape)
            x_noisy = mx.where(noise_mask, random_tokens, x)

            loss_noisy = model(x_noisy, y, reduction="none").reshape(-1)
            mx.eval(loss_noisy)
            noise_stats[rate]["nats"] += mx.sum(loss_noisy * mask).item()
            noise_stats[rate]["bytes"] += int(mx.sum(nb).item())

        steps += 1

    if clean_bytes == 0:
        return TestResult("noise_robustness", baseline_bpb, baseline_bpb, 0.0,
                          "insufficient data", 0)

    clean_bpb = clean_nats / (math.log(2) * clean_bytes)

    # Find worst noise level (highest BPB)
    worst_rate = 0.0
    worst_bpb = 0.0
    rate_details = []
    for rate in noise_rates:
        stats = noise_stats[rate]
        if stats["bytes"] > 0:
            bpb = stats["nats"] / (math.log(2) * stats["bytes"])
            rate_details.append(f"{rate:.0%}={bpb:.3f}")
            if bpb > worst_bpb:
                worst_bpb = bpb
                worst_rate = rate

    return TestResult(
        test_name="noise_robustness",
        adversarial_bpb=worst_bpb,
        baseline_bpb=baseline_bpb,
        robustness_gap=worst_bpb - baseline_bpb,
        description=f"clean={clean_bpb:.3f} worst_rate={worst_rate:.0%} {' '.join(rate_details)}",
        num_tokens_tested=steps * 16 * MAX_SEQ_LEN * (1 + len(noise_rates)),
    )


# ---------------------------------------------------------------------------
# Test 5: Adversarial token search (gradient-free worst-case input)
# ---------------------------------------------------------------------------

def test_adversarial_search(model, tokenizer, baseline_bpb, time_budget):
    """
    Gradient-free search for the worst-case input sequence.
    Starts from real validation sequences (not random tokens) and iteratively
    mutates them to maximize per-sequence BPB. Uses a simple evolutionary
    strategy: keep mutations that increase loss.

    This finds the model's absolute worst case on semi-natural inputs.
    """
    t0 = time.time()
    val_loader = make_dataloader(tokenizer, 8, MAX_SEQ_LEN, "val")
    token_bytes = get_token_bytes()
    vocab_size = tokenizer.get_vocab_size()

    # Seed with real validation sequences — use 2 batches for more diversity
    candidates = []
    for _ in range(2):
        x_seed, y_seed, _ = next(val_loader)
        mx.eval(x_seed, y_seed)
        for i in range(x_seed.shape[0]):
            seq = x_seed[i:i+1]
            tgt = y_seed[i:i+1]
            loss = model(seq, tgt, reduction="none").reshape(-1)
            nb = mx.take(token_bytes, tgt.reshape(-1), axis=0)
            mask = nb > 0
            nats = mx.sum(loss * mask).item()
            nbytes = int(mx.sum(nb).item())
            mx.eval()
            bpb = nats / (math.log(2) * nbytes) if nbytes > 0 else 0.0
            candidates.append((np.array(seq.reshape(-1)), bpb))

    generations = 0

    while time.time() - t0 < time_budget:
        # Sort by BPB descending — focus effort on most promising candidates
        candidates.sort(key=lambda c: c[1], reverse=True)
        # Focus on top half
        active = min(len(candidates), 8)

        for idx in range(active):
            if time.time() - t0 >= time_budget:
                break

            seq_np, current_bpb = candidates[idx]

            # Adaptive mutation rate: higher for lower-BPB candidates
            base_rate = 0.02 + 0.18 * (1.0 - idx / active)  # 2-20%
            mutant = seq_np.copy()
            n_mutations = max(1, int(len(mutant) * base_rate))
            positions = np.random.choice(len(mutant), n_mutations, replace=False)
            mutant[positions] = np.random.randint(4, vocab_size, size=n_mutations)

            # Evaluate mutant
            seq_mx = mx.array(mutant.reshape(1, -1), dtype=mx.int32)
            inputs = seq_mx[:, :-1]
            targets = seq_mx[:, 1:]
            loss = model(inputs, targets, reduction="none").reshape(-1)
            nb = mx.take(token_bytes, targets.reshape(-1), axis=0)
            mask = nb > 0
            nats = mx.sum(loss * mask).item()
            nbytes = int(mx.sum(nb).item())
            mx.eval()

            if nbytes > 0:
                mutant_bpb = nats / (math.log(2) * nbytes)
                if mutant_bpb > current_bpb:
                    candidates[idx] = (mutant, mutant_bpb)

        # Crossover: combine top 2 candidates
        if len(candidates) >= 2 and time.time() - t0 < time_budget:
            parent_a = candidates[0][0]
            parent_b = candidates[1][0]
            crossover_point = np.random.randint(len(parent_a) // 4, 3 * len(parent_a) // 4)
            child = np.concatenate([parent_a[:crossover_point], parent_b[crossover_point:]])
            seq_mx = mx.array(child.reshape(1, -1), dtype=mx.int32)
            inputs = seq_mx[:, :-1]
            targets = seq_mx[:, 1:]
            loss = model(inputs, targets, reduction="none").reshape(-1)
            nb = mx.take(token_bytes, targets.reshape(-1), axis=0)
            mask = nb > 0
            nats = mx.sum(loss * mask).item()
            nbytes = int(mx.sum(nb).item())
            mx.eval()
            if nbytes > 0:
                child_bpb = nats / (math.log(2) * nbytes)
                # Replace worst candidate if child is better
                if child_bpb > candidates[-1][1]:
                    candidates[-1] = (child, child_bpb)

        generations += 1

    # Find the worst across all candidates
    best_seq, best_bpb = max(candidates, key=lambda c: c[1])

    return TestResult(
        test_name="adversarial_search",
        adversarial_bpb=best_bpb,
        baseline_bpb=baseline_bpb,
        robustness_gap=best_bpb - baseline_bpb,
        description=f"worst_found={best_bpb:.3f} generations={generations} candidates={len(candidates)}",
        num_tokens_tested=generations * len(candidates) * MAX_SEQ_LEN,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_tests(model, tokenizer, baseline_bpb, time_budget):
    """Run all adversarial tests and collect results."""
    tests = [
        test_memorization_leakage,
        test_subgroup_disparity,
        test_window_boundary_exploit,
        test_noise_robustness,
        test_adversarial_search,
    ]
    per_test_budget = time_budget / len(tests)
    results = []

    for test_fn in tests:
        print(f"  Running {test_fn.__name__}...")
        t0 = time.time()
        result = test_fn(model, tokenizer, baseline_bpb, per_test_budget)
        dt = time.time() - t0
        print(f"    gap={result.robustness_gap:.3f} ({dt:.1f}s)")
        results.append(result)

    return results
