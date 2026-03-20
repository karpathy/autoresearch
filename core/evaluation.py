"""Walk-forward evaluation engine, composite scoring, and diagnostics.

This module provides the core evaluation loop that retrains a model recipe
on each walk-forward window and computes a composite score. It is
asset-agnostic — all asset-specific logic (data loading, window definitions)
is injected via parameters.
"""

import math
import time

import numpy as np
import pandas as pd

from core.backtesting import backtest
from core.config import AssetConfig
from core.epoch import get_holdout_idx, peek_eval_count, read_and_increment_eval_count


# ---------------------------------------------------------------------------
# Internal: run build_model on a window and backtest
# ---------------------------------------------------------------------------

def _eval_window(build_model_fn, all_data, window, window_idx,
                 backtest_config):
    """Train a model on one window's training data and backtest on its eval period.

    Returns (backtest_result, train_seconds, eval_preds) where eval_preds is
    the array of sigma predictions on the eval period (for diagnostics).
    """
    train_mask = ((all_data["timestamp"] >= window["train_start"]) &
                  (all_data["timestamp"] <= window["train_end"]))
    train_data = all_data[train_mask].reset_index(drop=True)

    t0 = time.time()
    predict_fn = build_model_fn(train_data)
    train_seconds = time.time() - t0

    print(f"  Window {window_idx}: trained in {train_seconds:.1f}s, ", end="", flush=True)

    # Generate predictions on full dataset (for feature lookback context)
    sigma_preds, timestamps, vol = predict_fn(all_data)
    sigma_preds = np.asarray(sigma_preds, dtype=np.float64).ravel()
    timestamps = np.asarray(timestamps).ravel()

    assert len(sigma_preds) == len(timestamps), (
        f"predict_fn returned {len(sigma_preds)} predictions but "
        f"{len(timestamps)} timestamps"
    )

    pred_df = pd.DataFrame({"timestamp": timestamps, "sigma_pred": sigma_preds})

    # Extract eval period and backtest
    eval_mask = ((all_data["timestamp"] >= window["eval_start"]) &
                 (all_data["timestamp"] <= window["eval_end"]))
    eval_data = all_data[eval_mask].reset_index(drop=True)
    merged = eval_data.merge(pred_df, on="timestamp", how="inner")

    if len(merged) == 0:
        print("no predictions!")
        return {
            "sharpe": -10.0, "max_drawdown": -1.0,
            "n_trades": 0, "total_return": -1.0,
            "subperiod_returns": [-1.0] * len(window["subperiods"]),
        }, train_seconds, np.array([])

    eval_preds = merged["sigma_pred"].values

    bt = backtest(
        sigma_predictions=eval_preds,
        close_prices=merged["close"].values,
        timestamps=merged["timestamp"].values,
        subperiods=window["subperiods"],
        config=backtest_config,
    )
    print("evaluated")
    return bt, train_seconds, eval_preds


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_windows(scored_results, scored_windows, forward_hours):
    """Compute composite score from scored window results.

    Returns dict with: score, sharpe_min, max_drawdown, total_trades,
                       consistency, dd_mult, trade_mult, n_profitable, n_total
    """
    # Base: minimum Sharpe across scored windows.
    sharpes = [r["sharpe"] for r in scored_results]
    base = min(sharpes)

    # Drawdown: worst across scored windows
    worst_dd_raw = min(r["max_drawdown"] for r in scored_results)  # most negative
    dd = abs(worst_dd_raw)
    if dd <= 0.10:
        dd_mult = 1.0
    else:
        dd_mult = 1.0 / (1.0 + ((dd - 0.10) / 0.15) ** 2)

    # Trade count: per-window exponential. Min across scored windows.
    window_trade_mults = []
    for w, result in zip(scored_windows, scored_results):
        eval_hours = (w["eval_end"] - w["eval_start"]).total_seconds() / 3600
        scale = eval_hours / (forward_hours * 7)
        window_trade_mults.append(1 - math.exp(-result["n_trades"] / scale))
    trade_mult = min(window_trade_mults)
    total_trades = sum(r["n_trades"] for r in scored_results)

    # Consistency: subperiod profitability
    all_sp_returns = []
    for r in scored_results:
        all_sp_returns.extend(r["subperiod_returns"])
    n_profitable = sum(1 for ret in all_sp_returns if ret > 0)
    n_total = len(all_sp_returns)
    consistency = n_profitable / n_total if n_total > 0 else 0.0

    if base >= 0:
        score = base * dd_mult * trade_mult * consistency
    else:
        score = base / max(dd_mult, 0.01) / max(trade_mult, 0.01) / max(consistency, 0.01)

    return {
        "score": score,
        "sharpe_min": base,
        "max_drawdown": worst_dd_raw,
        "total_trades": total_trades,
        "dd_mult": dd_mult,
        "trade_mult": trade_mult,
        "window_trade_mults": window_trade_mults,
        "n_profitable": n_profitable,
        "n_total": n_total,
        "consistency": consistency,
    }


# ---------------------------------------------------------------------------
# Public Evaluation API
# ---------------------------------------------------------------------------

def evaluate_model(build_model_fn, load_data_fn, windows, config: AssetConfig) -> dict:
    """Black-box walk-forward evaluation with rotating holdout.

    The recipe (build_model_fn) is retrained independently on each
    walk-forward window. One window is held out from scoring each epoch
    (rotates every epoch_length evaluations). The composite score is
    computed from the scored windows only.

    The agent sees the composite score, a holdout health flag,
    and the current epoch number (so it knows when rotations occur).
    Per-window results are hidden.

    Args:
        build_model_fn: Callable that takes a pd.DataFrame (training data)
            and returns a predict_fn. The predict_fn takes a pd.DataFrame
            and returns (sigma_predictions, timestamps, vol).
        load_data_fn: Callable that returns the full OHLCV DataFrame.
        windows: List of window dicts with train_start, train_end,
            eval_start, eval_end, subperiods keys.
        config: AssetConfig with evaluation parameters.

    Returns:
        dict with keys: score, sharpe_min, max_drawdown, total_trades,
                        consistency, holdout_health, epoch.
    """
    all_data = load_data_fn()
    wf = config.walk_forward

    # Determine holdout window for this epoch
    eval_count = read_and_increment_eval_count(config.eval_count_path)
    holdout_idx = get_holdout_idx(eval_count, len(windows), wf.epoch_length,
                                  config.salt_env_var)
    epoch = eval_count // wf.epoch_length

    # Evaluate ALL windows
    print(f"Evaluating ({len(windows)} walk-forward windows, 1 held out)...")
    all_results = []
    for i, w in enumerate(windows):
        bt, _, _ = _eval_window(build_model_fn, all_data, w, i + 1,
                                config.backtest)
        all_results.append(bt)

    # Split into scored and holdout
    scored_indices = [i for i in range(len(windows)) if i != holdout_idx]
    scored_results = [all_results[i] for i in scored_indices]
    scored_windows = [windows[i] for i in scored_indices]
    holdout_result = all_results[holdout_idx]

    # Composite score
    scores = _score_windows(scored_results, scored_windows, wf.forward_hours)

    # Holdout health check
    holdout_sharpe = holdout_result["sharpe"]
    if holdout_sharpe >= wf.holdout_ok_threshold:
        holdout_health = "OK"
    elif holdout_sharpe >= wf.holdout_warn_threshold:
        holdout_health = "CAUTION"
    else:
        holdout_health = "WARN"

    return {
        "score": scores["score"],
        "sharpe_min": scores["sharpe_min"],
        "max_drawdown": scores["max_drawdown"],
        "total_trades": scores["total_trades"],
        "consistency": f"{scores['n_profitable']}/{scores['n_total']}",
        "holdout_health": holdout_health,
        "epoch": epoch,
    }


# ---------------------------------------------------------------------------
# Per-Window Diagnostic (human-only)
# ---------------------------------------------------------------------------

def _pred_stats_line(preds):
    """One-line prediction distribution summary."""
    abs_p = np.abs(preds)
    e10 = (abs_p > 0.10).sum() / len(preds) * 100
    e20 = (abs_p > 0.20).sum() / len(preds) * 100
    e30 = (abs_p > 0.30).sum() / len(preds) * 100
    return (f"  Predictions: mean={preds.mean():+.4f}, std={preds.std():.4f}, "
            f"|pred|>0.10: {e10:.0f}%, >0.20: {e20:.0f}%, >0.30: {e30:.0f}%")


def run_diagnostic(build_model_fn, load_data_fn, windows, config: AssetConfig,
                   extra_holdout_window=None):
    """Per-window diagnostic breakdown with prediction stats.

    Human-only tool. Retrains the recipe on each walk-forward window and
    prints per-window backtest results and prediction distributions.
    Optionally evaluates an extra holdout window (e.g. 2026 data).

    Does NOT increment the eval counter.
    """
    all_data = load_data_fn()
    wf = config.walk_forward

    # Epoch info (read-only — do not increment counter)
    count = peek_eval_count(config.eval_count_path)
    holdout_idx = get_holdout_idx(count, len(windows), wf.epoch_length,
                                  config.salt_env_var)
    epoch = count // wf.epoch_length

    print("=" * 60)
    print("=== EPOCH INFO ===")
    print(f"Current eval count: {count}")
    print(f"Current epoch: {epoch} (holdout window: {holdout_idx})")
    print(f"Next rotation at eval: {(epoch + 1) * wf.epoch_length}")
    print("=" * 60)
    print(f"\nTraining and evaluating across {len(windows)} walk-forward windows...\n")

    window_results = []
    window_preds = []
    window_train_times = []
    for i, w in enumerate(windows):
        bt, train_time, eval_preds = _eval_window(
            build_model_fn, all_data, w, i + 1, config.backtest)
        window_results.append(bt)
        window_preds.append(eval_preds)
        window_train_times.append(train_time)

    # Split into scored and holdout (same logic as evaluate_model)
    scored_indices = [i for i in range(len(windows)) if i != holdout_idx]
    scored_results = [window_results[i] for i in scored_indices]
    scored_windows = [windows[i] for i in scored_indices]

    # Compute composite score on scored windows only
    scores = _score_windows(scored_results, scored_windows, wf.forward_hours)

    # Print diagnostic breakdown
    print("\n" + "=" * 60)
    print("=== DIAGNOSTIC BREAKDOWN ===")
    print("(Human-only — this information is hidden from the agent)")
    print("=" * 60)
    print(f"\nScored composite: score={scores['score']:.4f}, sharpe_min={scores['sharpe_min']:.4f}, "
          f"consistency={scores['n_profitable']}/{scores['n_total']}")
    print(f"  dd_mult={scores['dd_mult']:.4f}, trade_mult={scores['trade_mult']:.4f}")

    losing_subperiods = []
    scored_trade_mult_idx = 0

    for i, (w, result, preds) in enumerate(
            zip(windows, window_results, window_preds)):
        train_range = f"{w['train_start'].date()}-{w['train_end'].date()}"
        eval_range = f"{w['eval_start'].date()}-{w['eval_end'].date()}"
        status = "[HOLDOUT]" if i == holdout_idx else "[SCORED]"
        print(f"\n--- Window {i}: Train {train_range}, "
              f"Eval {eval_range} {status} ---")
        print(f"  Sharpe: {result['sharpe']:.4f}"
              f"{'  <-- MIN (scored)' if i != holdout_idx and result['sharpe'] == scores['sharpe_min'] else ''}")
        print(f"  Max drawdown: {result['max_drawdown']:.1%}")
        if i != holdout_idx:
            tm = scores["window_trade_mults"][scored_trade_mult_idx]
            print(f"  Trades: {result['n_trades']}  (trade_mult={tm:.4f})"
                  f"{'  <-- MIN' if tm == scores['trade_mult'] else ''}")
            scored_trade_mult_idx += 1
        else:
            print(f"  Trades: {result['n_trades']}  (holdout — not scored)")
        print(f"  Total return: {result['total_return']:+.1%}")

        for (sp_start, sp_end, label), sp_ret in zip(
                w["subperiods"], result["subperiod_returns"]):
            symbol = "OK" if sp_ret > 0 else "LOSS"
            print(f"  {label} ({sp_start.date()} to {sp_end.date()}): "
                  f"return {sp_ret:+.2%}  {symbol}")
            if sp_ret <= 0:
                losing_subperiods.append(label)

        if len(preds) > 0:
            print(_pred_stats_line(preds))

    # Holdout health
    holdout_result = window_results[holdout_idx]
    holdout_health = "OK" if holdout_result["sharpe"] >= wf.holdout_warn_threshold else "WARN"
    print(f"\nHoldout health: {holdout_health} "
          f"(Window {holdout_idx} Sharpe: {holdout_result['sharpe']:.4f})")
    print(f"Losing subperiods: {', '.join(losing_subperiods) if losing_subperiods else 'None'}")

    # Extra holdout window (e.g. 2026 true holdout)
    if extra_holdout_window is not None:
        w = extra_holdout_window
        bt_th, _, th_preds = _eval_window(
            build_model_fn, all_data, w, "holdout", config.backtest)
        print(f"\n--- True Holdout (UNSCORED) ---")
        print(f"  (Trained on {w['train_start'].date()}-{w['train_end'].date()}, "
              f"evaluated on {w['eval_start'].date()}-{w['eval_end'].date()})")
        print(f"  Sharpe: {bt_th['sharpe']:.4f}")
        print(f"  Max drawdown: {bt_th['max_drawdown']:.1%}")
        print(f"  Trades: {bt_th['n_trades']}")
        print(f"  Total return: {bt_th['total_return']:+.1%}")
        for (sp_start, sp_end, label), sp_ret in zip(
                w["subperiods"], bt_th["subperiod_returns"]):
            print(f"  {label}: return {sp_ret:+.2%}")
        if len(th_preds) > 0:
            print(_pred_stats_line(th_preds))

    print()
