#!/usr/bin/env python3
"""
End-to-end test of AutoAnything with toy problems.

Simulates agent submissions, scores them, and generates a progress chart.
Runs in a temp directory — does not modify the repo working tree.

Usage:
    python examples/run_test.py rastrigin
    python examples/run_test.py tsp --submissions 20
    python examples/run_test.py packing --include-failures -o chart.png
"""

import argparse
import math
import os
import random
import shutil
import sys
import tempfile

from autoanything.history import init_db, record_evaluation
from autoanything.plotting import generate_chart


# ---------------------------------------------------------------------------
# Scoring functions (inlined from context/ files for full isolation)
# ---------------------------------------------------------------------------

def rastrigin(x):
    """Rastrigin function in 10 dimensions. Global minimum: 0.0 at origin."""
    if len(x) != 10:
        raise ValueError(f"Expected 10 dimensions, got {len(x)}")
    return 10 * len(x) + sum(
        xi ** 2 - 10 * math.cos(2 * math.pi * xi) for xi in x
    )


TSP_CITIES = [
    (60, 200), (180, 200), (80, 180), (140, 180), (20, 160),
    (100, 160), (200, 160), (140, 140), (40, 120), (100, 120),
    (180, 100), (60, 80), (120, 80), (180, 60), (100, 40),
    (40, 40), (140, 20), (20, 20), (200, 20), (120, 160),
]


def tour_distance(tour):
    """Total Euclidean distance of a closed 20-city tour."""
    if len(tour) != 20 or set(tour) != set(range(20)):
        raise ValueError("Tour must be a permutation of [0..19]")
    total = 0.0
    for i in range(20):
        x1, y1 = TSP_CITIES[tour[i]]
        x2, y2 = TSP_CITIES[tour[(i + 1) % 20]]
        total += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return round(total, 4)


PACK_RECTS = [
    (40, 20), (30, 15), (50, 10), (25, 25), (35, 20),
    (20, 30), (45, 15), (15, 35), (30, 10), (20, 20),
    (40, 25), (10, 40),
]


def packing_score(placements):
    """Bounding-box area + 10 000 per overlapping pair."""
    if len(placements) != 12:
        raise ValueError(f"Expected 12 placements, got {len(placements)}")
    rects = []
    max_x = max_y = 0
    for i, (x, y, rotated) in enumerate(placements):
        w, h = PACK_RECTS[i]
        if rotated:
            w, h = h, w
        rects.append((x, y, x + w, y + h))
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
    overlaps = 0
    for i in range(12):
        for j in range(i + 1, 12):
            ax1, ay1, ax2, ay2 = rects[i]
            bx1, by1, bx2, by2 = rects[j]
            if ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1:
                overlaps += 1
    return max_x * max_y + 10000 * overlaps


# ---------------------------------------------------------------------------
# Submission generators
# ---------------------------------------------------------------------------

def gen_rastrigin(n, include_failures, seed):
    """Generate rastrigin submissions: zeroing-out improvements + random noise.

    Zeroing any non-zero dimension always improves the Rastrigin score
    (per-dim minimum is -10 at x=0). Noise adds gaussian perturbation
    to all dims, which re-corrupts zeroed dims and usually makes things worse.
    """
    rng = random.Random(seed)
    initial = [2.5, -1.8, 3.1, -0.5, 4.2, -3.7, 1.9, -2.3, 0.8, -4.1]
    current_best = list(initial)

    imp_descs = [
        "zero out largest coordinate",
        "zero out 2 large coordinates",
        "zero out coords |x|<2",
        "zero out 2 more coords",
        "zero out remaining coords",
        "zero all (near-optimal)",
    ]

    noise_descs = [
        "random perturbation \u03c3=1.5",
        "explore near boundary",
        "gaussian noise (wide)",
        "random restart attempt",
        "perturb even-index dims",
        "flip-sign experiment",
        "high-variance walk",
        "random direction step",
        "shift by random constant",
        "exploratory noise",
    ]

    submissions = []
    imp_idx = noise_idx = 0

    for i in range(n):
        if include_failures and i == n // 2:
            submissions.append(("crash: wrong dimensions", None))
            continue

        # Improvement every 3rd slot (i=1, 4, 7, 10, 13, ...)
        if (i + 2) % 3 == 0 and imp_idx < len(imp_descs):
            candidate = list(current_best)
            nz = sorted(
                [j for j in range(10) if candidate[j] != 0.0],
                key=lambda j: abs(candidate[j]), reverse=True,
            )
            # Zero out 1-3 of the largest non-zero dims
            if imp_idx == 0:
                n_zero = 1
            elif imp_idx < 4:
                n_zero = min(2, len(nz))
            else:
                n_zero = len(nz)  # zero all remaining
            for j in nz[:n_zero]:
                candidate[j] = 0.0

            desc = imp_descs[imp_idx % len(imp_descs)]
            imp_idx += 1
        else:
            # Noise: perturb all dims (re-corrupts zeroed dims → always worse)
            candidate = [v + rng.gauss(0, 1.5) for v in current_best]
            desc = noise_descs[noise_idx % len(noise_descs)]
            noise_idx += 1

        if rastrigin(candidate) < rastrigin(current_best):
            current_best = list(candidate)

        submissions.append((desc, candidate))

    return initial, submissions


def gen_tsp(n, include_failures, seed):
    """Generate TSP submissions: mix of 2-opt reversals and random swaps."""
    rng = random.Random(seed)
    initial = list(range(20))
    current_best = list(initial)
    submissions = []

    for i in range(n):
        if include_failures and i == n // 2:
            submissions.append(("crash: invalid tour", None))
            continue

        candidate = list(current_best)
        roll = rng.random()

        if roll < 0.45:
            # 2-opt: reverse a segment (often improves)
            a = rng.randint(0, 17)
            b = rng.randint(a + 2, 19)
            candidate[a:b + 1] = reversed(candidate[a:b + 1])
            desc = f"2-opt reverse [{a}:{b}]"
        elif roll < 0.75:
            # Swap two cities
            a, b = rng.sample(range(20), 2)
            candidate[a], candidate[b] = candidate[b], candidate[a]
            desc = f"swap cities {candidate[b]}\u2194{candidate[a]}"
        else:
            # Or-opt: relocate a city
            src = rng.randint(0, 19)
            city = candidate.pop(src)
            dst = rng.randint(0, 18)
            candidate.insert(dst, city)
            desc = f"move city {city} to pos {dst}"

        if tour_distance(candidate) < tour_distance(current_best):
            current_best = list(candidate)

        submissions.append((desc, candidate))

    return initial, submissions


def _bottom_left_fill(order, rotations):
    """Bottom-left-fill heuristic that minimises bounding-box area."""
    placed = []  # list of (x1, y1, x2, y2)
    placements = [None] * 12
    bb_x = bb_y = 0  # current bounding box

    for idx in order:
        w, h = PACK_RECTS[idx]
        if rotations[idx]:
            w, h = h, w

        # Candidate positions: origin + every corner combination
        cands = {(0, 0)}
        for px1, py1, px2, py2 in placed:
            cands.update([(px2, 0), (0, py2), (px2, py1), (px1, py2)])
            for qx1, qy1, qx2, qy2 in placed:
                cands.add((px2, qy2))
                cands.add((qx2, py2))

        best_pos = None
        best_area = float("inf")

        for cx, cy in cands:
            if cx < 0 or cy < 0:
                continue
            # Check no overlap
            ok = True
            for px1, py1, px2, py2 in placed:
                if cx < px2 and cx + w > px1 and cy < py2 and cy + h > py1:
                    ok = False
                    break
            if ok:
                area = max(bb_x, cx + w) * max(bb_y, cy + h)
                if area < best_area or (area == best_area and (cy, cx) < (
                        best_pos[1] if best_pos else 999,
                        best_pos[0] if best_pos else 999)):
                    best_pos = (cx, cy)
                    best_area = area

        if best_pos is None:
            best_pos = (0, bb_y)

        x, y = best_pos
        placements[idx] = (x, y, rotations[idx])
        placed.append((x, y, x + w, y + h))
        bb_x = max(bb_x, x + w)
        bb_y = max(bb_y, y + h)

    return placements


def gen_packing(n, include_failures, seed):
    """Generate packing submissions: BLF improvements + jitter noise.

    Pre-computes BLF with many orderings/rotations, sorts worst-to-best,
    and schedules them as improvements.  Noise is random position jitter
    of the current best layout, which usually creates overlaps (rejected).
    """
    rng = random.Random(seed)

    initial = [
        (0, 0, False), (0, 20, False), (0, 35, False), (0, 45, False),
        (0, 70, False), (0, 90, False), (0, 120, False), (0, 135, False),
        (0, 170, False), (0, 180, False), (0, 200, False), (0, 225, False),
    ]

    # Pre-compute BLF with many strategy/rotation combos
    blf_strategies = [
        ("BLF area-desc",
         sorted(range(12), key=lambda j: PACK_RECTS[j][0] * PACK_RECTS[j][1], reverse=True),
         [False] * 12),
        ("BLF width-desc",
         sorted(range(12), key=lambda j: PACK_RECTS[j][0], reverse=True),
         [False] * 12),
        ("BLF height-desc",
         sorted(range(12), key=lambda j: PACK_RECTS[j][1], reverse=True),
         [False] * 12),
        ("BLF max-dim",
         sorted(range(12), key=lambda j: max(PACK_RECTS[j]), reverse=True),
         [False] * 12),
        ("BLF perimeter-desc",
         sorted(range(12), key=lambda j: sum(PACK_RECTS[j]), reverse=True),
         [False] * 12),
        ("BLF area-desc + rotate all",
         sorted(range(12), key=lambda j: PACK_RECTS[j][0] * PACK_RECTS[j][1], reverse=True),
         [True] * 12),
        ("BLF width-desc + rotate all",
         sorted(range(12), key=lambda j: PACK_RECTS[j][0], reverse=True),
         [True] * 12),
        ("BLF height-desc + rotate wide",
         sorted(range(12), key=lambda j: PACK_RECTS[j][1], reverse=True),
         [PACK_RECTS[j][0] > PACK_RECTS[j][1] for j in range(12)]),
        ("BLF area-desc + rotate tall",
         sorted(range(12), key=lambda j: PACK_RECTS[j][0] * PACK_RECTS[j][1], reverse=True),
         [PACK_RECTS[j][1] > PACK_RECTS[j][0] for j in range(12)]),
        ("BLF area-asc",
         sorted(range(12), key=lambda j: PACK_RECTS[j][0] * PACK_RECTS[j][1]),
         [False] * 12),
        ("BLF aspect-ratio order",
         sorted(range(12), key=lambda j: PACK_RECTS[j][0] / PACK_RECTS[j][1]),
         [False] * 12),
        ("BLF reverse order",
         list(range(11, -1, -1)),
         [False] * 12),
    ]

    results = []
    for desc, order, rots in blf_strategies:
        candidate = _bottom_left_fill(order, rots)
        results.append((packing_score(candidate), desc, candidate))

    # Build a strictly-improving sequence (worst → best)
    results.sort(key=lambda x: x[0], reverse=True)
    baseline_score = packing_score(initial)
    improvements = []
    best_so_far = baseline_score
    for score, desc, candidate in results:
        if score < best_so_far:
            improvements.append((desc, candidate))
            best_so_far = score

    noise_descs = [
        "jitter rect {idx} by ({dx:+d},{dy:+d})",
        "shift rect {idx} right by {dx}",
        "nudge rect {idx} down by {dy}",
        "swap rects {a}\u2194{b} positions",
        "random shift rect {idx}",
    ]

    submissions = []
    imp_idx = noise_idx = 0
    current_best = list(initial)

    for i in range(n):
        if include_failures and i == n // 2:
            submissions.append(("crash: wrong placement count", None))
            continue

        # Improvement at i=1,4,7,10,13,...
        if (i + 2) % 3 == 0 and imp_idx < len(improvements):
            desc, candidate = improvements[imp_idx]
            imp_idx += 1
        else:
            # Noise: jitter a random rect (usually creates overlaps → rejected)
            candidate = list(current_best)
            idx = rng.randint(0, 11)
            dx = rng.randint(-30, 30)
            dy = rng.randint(-30, 30)
            x, y, rot = candidate[idx]
            candidate[idx] = (max(0, x + dx), max(0, y + dy), rot)
            tmpl = noise_descs[noise_idx % len(noise_descs)]
            a, b = rng.sample(range(12), 2)
            desc = tmpl.format(idx=idx, dx=dx, dy=dy, a=a, b=b)
            noise_idx += 1

        if packing_score(candidate) < packing_score(current_best):
            current_best = list(candidate)

        submissions.append((desc, candidate))

    return initial, submissions


# ---------------------------------------------------------------------------
# Problem registry
# ---------------------------------------------------------------------------

PROBLEMS = {
    "rastrigin": dict(
        name="Rastrigin Function",
        score_label="Rastrigin f(x)",
        direction="minimize",
        score_fn=rastrigin,
        gen_fn=gen_rastrigin,
    ),
    "tsp": dict(
        name="Traveling Salesman (20 cities)",
        score_label="Tour Distance",
        direction="minimize",
        score_fn=tour_distance,
        gen_fn=gen_tsp,
    ),
    "packing": dict(
        name="Rectangle Packing",
        score_label="Bounding Box Score",
        direction="minimize",
        score_fn=packing_score,
        gen_fn=gen_packing,
    ),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end test of AutoAnything with toy problems"
    )
    parser.add_argument(
        "problem", choices=PROBLEMS.keys(),
        help="Which test problem to run",
    )
    parser.add_argument(
        "-n", "--submissions", type=int, default=15,
        help="Number of simulated agent submissions (default: 15)",
    )
    parser.add_argument(
        "--include-failures", action="store_true",
        help="Include intentionally crashing submissions",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output chart path (default: test_progress_<problem>.png)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    prob = PROBLEMS[args.problem]
    output = args.output or f"test_progress_{args.problem}.png"

    print(f"Running {prob['name']} test "
          f"({args.submissions} submissions, seed={args.seed})")
    print()

    # Temp dir for the database (keeps repo clean)
    tmpdir = tempfile.mkdtemp(prefix=f"autoanything-test-{args.problem}-")
    db_path = os.path.join(tmpdir, "history.db")
    conn = init_db(db_path)

    # Generate submissions
    initial_state, submissions = prob["gen_fn"](
        args.submissions, args.include_failures, args.seed,
    )

    # Score and record baseline
    baseline = prob["score_fn"](initial_state)
    record_evaluation(conn, "test", "master", baseline, "baseline", "baseline", 0.0)
    incumbent = baseline
    print(f"  {'BL':>3s}  {'BASELINE':8s}  {baseline:>12.4f}  baseline")

    # Process submissions
    n_accepted = n_rejected = n_crashed = 0

    for i, (desc, candidate) in enumerate(submissions, 1):
        if candidate is None:
            record_evaluation(
                conn, "test", f"proposals/agent/{desc}", None, "crash",
                desc, 0.0, error_message="Invalid state",
            )
            n_crashed += 1
            print(f"  {i:>3d}  {'CRASH':8s}  {'':>12s}  {desc}")
            continue

        try:
            score = prob["score_fn"](candidate)
        except Exception as e:
            record_evaluation(
                conn, "test", f"proposals/agent/{desc}", None, "crash",
                desc, 0.0, error_message=str(e),
            )
            n_crashed += 1
            print(f"  {i:>3d}  {'CRASH':8s}  {'':>12s}  {desc}: {e}")
            continue

        if score < incumbent:
            record_evaluation(
                conn, "test", f"proposals/agent/{desc}", score, "accepted",
                desc, 0.0,
            )
            incumbent = score
            n_accepted += 1
            print(f"  {i:>3d}  {'ACCEPTED':8s}  {score:>12.4f}  {desc}")
        else:
            record_evaluation(
                conn, "test", f"proposals/agent/{desc}", score, "rejected",
                desc, 0.0,
            )
            n_rejected += 1
            print(f"  {i:>3d}  {'rejected':8s}  {score:>12.4f}  {desc}")

    # Summary
    print()
    print(f"  {n_accepted} accepted, {n_rejected} rejected, {n_crashed} crashed")
    print(f"  {baseline:.4f} \u2192 {incumbent:.4f}")
    print()

    # Generate chart
    try:
        generate_chart(
            db_path, output,
            title=(f"{prob['name']}: {len(submissions) + 1} Experiments, "
                   f"{n_accepted} Kept"),
            direction=prob["direction"],
            score_label=prob["score_label"],
        )
        print(f"  Chart saved to {output}")
    except Exception as e:
        print(f"  Chart generation failed: {e}")
        print("  (install matplotlib: pip install matplotlib)")

    # Clean up
    shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
