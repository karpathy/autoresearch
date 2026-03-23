# AutoResearch Program: Weekly Iron Condor Bot

## Bot Location
- Local: /Users/jacoby/.openclaw/workspace/ibkr-trading/weekly_condor_bot.py
- Schedule: local.weekly-condor — Mon 7:45 AM entry, Mon-Fri 1 PM management

## What You Optimize
SPY weekly iron condor: sell 10-delta puts and calls, $5 wings, VIX 15-30 filter. Your job: improve Sharpe and reduce drawdowns.

## Metric
**Primary:** Sharpe ratio (target: >1.0, currently 0.69)
**Secondary:** Win rate (target: >85%, currently 82.8%)
**Tertiary:** Max drawdown (target: <-2%, currently -2.5%)

## Tunable Parameters

| Parameter | Current | Range | Description |
|---|---|---|---|
| DELTA_TARGET | 10 | 5-20 | Delta for short strikes |
| WING_WIDTH | $5 | $2-10 | Width of condor wings |
| VIX_MIN | 15 | 10-20 | Minimum VIX to enter |
| VIX_MAX | 30 | 25-40 | Maximum VIX to enter |
| CONTRACTS | 2 | 1-5 | Contracts per side |
| PROFIT_TARGET | 50% | 30-70% | Close at this % of max credit |
| STOP_LOSS | 2x credit | 1.5-3x | Stop loss multiplier |
| MANAGEMENT_TIME | 1 PM MT | 10AM-3PM | When to check positions |
| CLOSE_DAY | Thursday 3 PM ET | Wed-Fri | When to force close before expiry |

## How to Evaluate

### Backtest (10yr data available)
```bash
cd /workspace/ibkr-trading && python3 weekly_condor_bot.py --backtest --years 10
```
Current baseline: 82.8% WR, Sharpe 0.69, PF 1.33, -2.5% max DD (203 trades)

## Experiment Ideas
1. **Asymmetric deltas** — Sell 8Δ puts / 12Δ calls (skew toward put premium)
2. ~~**VIX band narrowing**~~ — ❌ DISCARDED 2026-03-22: VIX 18-25 Sharpe 0.12 vs 15-30 Sharpe 0.69. Trades halved (203→95), annual return collapsed to 0.0%.
3. **Wing width vs capital efficiency** — $3 wings: less capital, tighter, higher WR?
4. **Monday vs Tuesday entry** — Does waiting 1 day improve fills after weekend gap?
5. **Early close on Wednesday** — Close Wed 3PM instead of Thu. Less gamma risk.
6. **Dynamic delta based on VIX** — VIX 15-20 → 12Δ, VIX 20-25 → 10Δ, VIX 25-30 → 8Δ
7. **Profit target scaling** — VIX<20 → 40% target (take profits faster), VIX>25 → 60% (let it run)
8. **Skip OPEX weeks** — Monthly OpEx has higher gamma risk. Skip those Mondays?

## Constraints
- SPY ONLY (QQQ condors LOSE money — proven in every variant)
- VIX 15-30 filter is mandatory (see experiment: 18-25 band → Sharpe 0.12 — too restrictive)
- Max $500 loss per trade
- Max $500 weekly circuit breaker
- Paper until 4 weeks profitable (earliest live: 4/6)

## Key Lessons
- QQQ weekly condors don't work — too volatile
- VIX 15-30 is the correct band (not 18-25). High-VIX weeks (>25) provide needed diversification.
- VIX regime breakdown: Mid(15-25): 168 trades WR=85%, High(>25): 35 trades WR=74%.
- Stop losses are rare (7/203 trades) but devastating ($2,741 avg loss).

## Results Format (TSV)
```
date	experiment	param_changed	old_value	new_value	sharpe	win_rate	max_dd	status	description
```
