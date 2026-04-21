"""
Evaluation harness.

Runs whatever the research agent has written into ``algo.py`` and prints
canonical performance metrics (Sharpe is the primary metric). One TSV row
per run is appended to ``results.tsv``.

Usage
-----

    uv run test.py                                           # split=train (default)
    AUTORESEARCH_ALLOW_TEST=1 uv run test.py --split test    # held-out eval

Split model
-----------

The research agent iterates against the **train** split (2017-2022). Every
``uv run test.py`` the LLM does is an in-sample evaluation -- that is the
signal it uses to accept or reject a change to ``algo.py``. This is
intentional: the LLM itself is the outer optimizer, so the window it sees
per run IS its training distribution.

The **test** split (2023-2025) is the held-out evaluation. The research
loop never sees it; ``load.Data(split="test")`` refuses to construct unless
``AUTORESEARCH_ALLOW_TEST=1`` is set. You run it manually, at most once
per research cycle, to see whether the LLM's best algo generalizes or just
overfit 2017-2022 through selection pressure.

Contract with algo.py
---------------------

``algo.py`` must define a single callable::

    def strategy(data: load.Data) -> pandas.DataFrame:
        '''
        Return target portfolio weights.

        Index   : DatetimeIndex, tz-aware UTC, strictly increasing, no dups.
                  Every timestamp must be a real bar timestamp at either the
                  "day" or "30min" resolution of ``data.bars``.

        Columns : symbols, all drawn from ``data.universe()``.

        Values  : target weight of the symbol in portfolio NAV at that row's
                  timestamp. +0.5 = 50% long, -0.25 = 25% short, 0 or NaN =
                  flat. No cap on gross exposure -- leverage is on you.

        Semantics: the weights at row t are executed at that bar's close and
                  held until the next row. Returns earned from t to t+1 are
                  W[t] . R[t -> t+1]. Transaction cost of
                  TRANSACTION_COST_BPS one-way per unit of |dW| is applied at
                  every rebalance (including the initial entry from zero).
        '''

That's the entire contract. How you fetch data, what models you use, whether
you hold intraday or daily, single-name or basket, long-only or long/short --
all of that is up to you, as long as you return a weights DataFrame of the
shape above. ``data`` is pre-scoped to the split the harness was invoked
with, so the same ``strategy()`` function runs unchanged in both train and
held-out test.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from load import Data, SPLITS


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).parent
RESULTS_TSV = ROOT / "results.tsv"

# One-way transaction cost per unit of turnover, in basis points. Liquid-ETF
# best-case is well under 1bp; a mixed single-name universe will trade wider.
# 2bp is an intentionally tight-but-not-free number -- agents should prefer
# lower-turnover strategies, but a legitimate edge should survive this.
TRANSACTION_COST_BPS = 1.5


# --------------------------------------------------------------------------- #
# Strategy loading and validation
# --------------------------------------------------------------------------- #

def _import_strategy():
    try:
        from algo import strategy
    except Exception as e:
        print(
            "FATAL: could not import `strategy` from algo.py.\n"
            f"  {type(e).__name__}: {e}\n",
            file=sys.stderr,
        )
        traceback.print_exc()
        sys.exit(2)
    if not callable(strategy):
        print("FATAL: `algo.strategy` exists but is not callable.", file=sys.stderr)
        sys.exit(2)
    return strategy


def _infer_timeframe(index: pd.DatetimeIndex) -> str:
    """Pick the bar timeframe the weights align to.

    Heuristic: if there's exactly one timestamp per calendar date, treat it
    as daily; otherwise assume 30-minute bars. This covers every case
    supported by ``load.Data`` without asking the algo to declare anything.
    """
    if len(index) == len(index.normalize().unique()):
        return "day"
    return "30min"


def _validate_weights(w) -> pd.DataFrame:
    if not isinstance(w, pd.DataFrame):
        raise TypeError(
            f"strategy() must return a pandas.DataFrame; got {type(w).__name__}"
        )
    if w.empty:
        raise ValueError("strategy() returned an empty DataFrame")
    if not isinstance(w.index, pd.DatetimeIndex):
        raise TypeError("weights index must be a pandas.DatetimeIndex")
    if w.index.tz is None:
        raise ValueError(
            "weights index must be timezone-aware (UTC preferred). "
            "data.bars() returns UTC timestamps -- use those directly."
        )
    if not w.index.is_monotonic_increasing:
        raise ValueError("weights index must be sorted ascending")
    if w.index.duplicated().any():
        raise ValueError("weights index contains duplicate timestamps")
    if w.shape[1] == 0:
        raise ValueError("weights has no symbol columns")
    vals = w.to_numpy(dtype=float)
    if np.isinf(vals).any():
        raise ValueError("weights contain +/-inf values")
    return w


# --------------------------------------------------------------------------- #
# P&L engine
# --------------------------------------------------------------------------- #

def _wide_close(data: Data, symbols: list[str], timeframe: str) -> pd.DataFrame:
    """Fetch close prices as a DatetimeIndex x symbol wide frame."""
    bars = data.bars(symbols, timeframe=timeframe)
    if bars.empty:
        raise ValueError(
            f"No bars returned for {len(symbols)} symbol(s) at timeframe={timeframe!r}. "
            "Check your universe."
        )
    return bars["close"].unstack("symbol")


def _simulate(weights: pd.DataFrame, prices: pd.DataFrame, bps: float):
    """Compute gross and net per-period returns.

    Convention
    ----------
    - ``weights.loc[t]`` is executed at bar ``t``'s close.
    - It is held from ``t`` through ``t+1``.
    - Gross period return at ``t+1`` is ``W[t] . (P[t+1]/P[t] - 1)``.
    - Turnover at ``t`` is ``sum(|W[t] - W[t-1]|)`` (entry from zero counts).
      Cost = turnover * bps/1e4. Stamped at ``t+1`` for book-keeping.
    - First row yields no return (no prior price), but its entry turnover is
      paid and charged to row 1.

    Returns
    -------
    gross_ret, net_ret, turnover_per_period : pd.Series
        All indexed by bar timestamp of the end of the period (``t+1``).
    """
    w = weights.fillna(0.0)
    # Align prices exactly to the weight timestamps. Any weight timestamp
    # without a matching price is caller error (caught upstream).
    P = prices.reindex(index=w.index, columns=w.columns)

    # Per-symbol period return, stamped at t+1. First row is NaN.
    R = P.pct_change()

    # Gross portfolio return stamped at t+1.
    gross = (w.shift(1) * R).sum(axis=1, min_count=1)

    # Turnover (one-way) at each rebalance time t; shift forward one step so
    # the cost lands in the period it actually funds.
    turnover_t = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost_tp1 = turnover_t.shift(1) * (bps / 1e4)

    net = gross - cost_tp1.fillna(0.0)

    # Drop the leading NaN row (no return at t=0).
    mask = gross.notna()
    return gross[mask], net[mask], turnover_t


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _periods_per_year(timeframe: str) -> float:
    # Daily: ~252 US trading days/year.
    # 30-min: 13 session bars/day * 252 days = 3276 periods/year.
    return 252.0 if timeframe == "day" else 252.0 * 13.0


def _compute_metrics(
    gross: pd.Series,
    net: pd.Series,
    turnover_t: pd.Series,
    weights: pd.DataFrame,
    timeframe: str,
) -> dict:
    ppy = _periods_per_year(timeframe)
    n = len(net)
    if n == 0:
        raise ValueError("no P&L periods produced; weights may be a single row")

    net_arr = net.to_numpy()
    mean = float(net_arr.mean())
    std = float(net_arr.std(ddof=1)) if n > 1 else 0.0

    sharpe = (mean / std) * np.sqrt(ppy) if std > 0 else float("nan")
    t_stat = (mean / (std / np.sqrt(n))) if std > 0 else float("nan")
    hit_rate = float((net_arr > 0).mean())

    total_return = float((1.0 + net).prod() - 1.0)
    # Geometric annualized return.
    ann_return = (
        float((1.0 + net).prod() ** (ppy / n) - 1.0) if (1.0 + net).prod() > 0
        else float("nan")
    )
    ann_vol = std * np.sqrt(ppy)

    curve = (1.0 + net).cumprod()
    max_dd = float((curve / curve.cummax() - 1.0).min())

    # Turnover is one-way per period; annualize by count of periods/yr.
    # One-way annual turnover of 10.0 means gross dollar traded each year
    # equals ~10x NAV. (Round-trip would be 20x.)
    ann_turnover_oneway = float(turnover_t.mean() * ppy)

    gross_expo = weights.abs().sum(axis=1)
    net_expo = weights.sum(axis=1)

    cost_drag = float((gross.mean() - net.mean()) * ppy)  # annualized

    return {
        "sharpe": round(sharpe, 4) if np.isfinite(sharpe) else None,
        "total_return_pct": round(100 * total_return, 4),
        "ann_return_pct": round(100 * ann_return, 4) if np.isfinite(ann_return) else None,
        "ann_vol_pct": round(100 * ann_vol, 4),
        "max_drawdown_pct": round(100 * max_dd, 4),
        "t_stat_mean": round(t_stat, 4) if np.isfinite(t_stat) else None,
        "hit_rate_pct": round(100 * hit_rate, 4),
        "ann_turnover_oneway": round(ann_turnover_oneway, 4),
        "cost_drag_pct_per_yr": round(100 * cost_drag, 4),
        "avg_gross_exposure": round(float(gross_expo.mean()), 4),
        "avg_net_exposure": round(float(net_expo.mean()), 4),
        "max_gross_exposure": round(float(gross_expo.max()), 4),
        "n_periods": int(n),
        "n_symbols": int(weights.shape[1]),
        "timeframe": timeframe,
        "first_bar": str(net.index[0]),
        "last_bar": str(net.index[-1]),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _fmt(x, prec=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{prec}f}"
    return str(x)


def _print_report(m: dict, split: str, wall_seconds: float) -> None:
    split_s, split_e = SPLITS[split]
    width = 60
    bar = "=" * width

    print()
    print(bar)
    print(" TEST RESULTS ".center(width, "="))
    print(bar)
    print(f"Split        : {split}  ({split_s.date()} .. {split_e.date()})")
    print(f"Timeframe    : {m['timeframe']}")
    print(f"Periods      : {m['n_periods']:,}  ({m['first_bar']}  -->  {m['last_bar']})")
    print(f"Symbols      : {m['n_symbols']}")
    print(f"Cost model   : {TRANSACTION_COST_BPS:.1f} bps one-way per unit |dW|")
    print()
    print("RETURNS")
    print(f"  Sharpe              : {_fmt(m['sharpe'])}    <-- PRIMARY METRIC")
    print(f"  Total return        : {_fmt(m['total_return_pct'], 2)}%")
    print(f"  Ann. return         : {_fmt(m['ann_return_pct'], 2)}%")
    print(f"  Ann. volatility     : {_fmt(m['ann_vol_pct'], 2)}%")
    print(f"  t-stat of mean      : {_fmt(m['t_stat_mean'], 2)}")
    print(f"  Hit rate            : {_fmt(m['hit_rate_pct'], 2)}%")
    print()
    print("RISK")
    print(f"  Max drawdown        : {_fmt(m['max_drawdown_pct'], 2)}%")
    print(f"  Avg gross exposure  : {_fmt(m['avg_gross_exposure'], 2)}x")
    print(f"  Max gross exposure  : {_fmt(m['max_gross_exposure'], 2)}x")
    print(f"  Avg net exposure    : {_fmt(m['avg_net_exposure'], 2)}x")
    print()
    print("ACTIVITY")
    print(f"  Ann. turnover (1w)  : {_fmt(m['ann_turnover_oneway'], 2)}x")
    print(f"  Cost drag           : {_fmt(m['cost_drag_pct_per_yr'], 3)}% / yr")
    print()
    print(f"Wall time    : {wall_seconds:.1f}s")
    print(bar)


def _append_tsv(path: Path, row: dict) -> None:
    header = list(row.keys())
    write_header = not path.exists()
    with path.open("a") as f:
        if write_header:
            f.write("\t".join(header) + "\n")
        f.write("\t".join(str(row[k]) for k in header) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate algo.strategy on a fixed split.",
    )
    p.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help=(
            "Which split to evaluate on. 'train' (default) is the research "
            "agent's iteration window. 'test' is the held-out window and "
            "requires AUTORESEARCH_ALLOW_TEST=1 in the environment."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    strategy = _import_strategy()

    print(f"loading {args.split} split...", flush=True)
    try:
        data = Data(split=args.split)
    except PermissionError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 4

    print("running algo.strategy(data)...", flush=True)
    t0 = time.time()
    try:
        raw_weights = strategy(data)
    except Exception as e:
        print(
            f"FATAL: algo.strategy raised {type(e).__name__}: {e}\n",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 3
    t_strategy = time.time() - t0
    print(f"  strategy returned in {t_strategy:.1f}s", flush=True)

    weights = _validate_weights(raw_weights)

    universe = set(data.universe())
    bad = [s for s in weights.columns if s not in universe]
    if bad:
        preview = ", ".join(bad[:10]) + ("..." if len(bad) > 10 else "")
        raise ValueError(
            f"{len(bad)} weight column(s) are not in the canonical universe: {preview}"
        )

    timeframe = _infer_timeframe(weights.index)
    print(f"  inferred timeframe: {timeframe}", flush=True)

    print(f"fetching prices for {weights.shape[1]} symbols...", flush=True)
    prices = _wide_close(data, list(weights.columns), timeframe)

    missing = weights.index.difference(prices.index)
    if len(missing) > 0:
        raise ValueError(
            f"{len(missing)} weight timestamp(s) have no matching bar "
            f"at timeframe={timeframe!r} (showing first 3): "
            f"{[str(t) for t in missing[:3]]}. Weight timestamps must be "
            f"real bar timestamps returned by data.bars(..., timeframe={timeframe!r})."
        )

    print("simulating...", flush=True)
    gross, net, turnover_t = _simulate(weights, prices, TRANSACTION_COST_BPS)

    metrics = _compute_metrics(gross, net, turnover_t, weights.fillna(0.0), timeframe)

    wall = time.time() - t0
    _print_report(metrics, args.split, wall)

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": args.split,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "strategy_wall_seconds": round(t_strategy, 2),
        **metrics,
    }
    _append_tsv(RESULTS_TSV, row)
    print(f"Appended row to {RESULTS_TSV.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
