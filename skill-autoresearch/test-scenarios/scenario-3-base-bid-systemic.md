# Test Scenario 3: Systemic Underbidding — Base Bid Raise Needed

## Client Message
"Our lead volume has dropped significantly over the past 3 months. We're open to whatever it takes to get back to where we were. Budget is not a constraint."

## Campaign CSVs (5 months)

| Month | Searches | Impressions | Imp % | Clicks | Leads | Cost | Avg CPC | Avg Bid | Win Rate | Sold | Sold CPA |
|-------|----------|-------------|-------|--------|-------|------|---------|---------|----------|------|----------|
| Nov 2025 | 18,200 | 4,550 | 25.0% | 341 | 76 | $4,774 | $14.00 | $15.00 | 22.1% | 11 | $434 |
| Dec 2025 | 17,800 | 3,916 | 22.0% | 294 | 65 | $4,116 | $14.00 | $15.00 | 19.4% | 9 | $457 |
| Jan 2026 | 18,500 | 3,145 | 17.0% | 236 | 52 | $3,304 | $14.00 | $15.00 | 15.2% | 7 | $472 |
| Feb 2026 | 18,100 | 2,534 | 14.0% | 190 | 42 | $2,660 | $14.00 | $15.00 | 12.0% | 6 | $443 |
| Mar 2026 (25 days) | 12,400 | 1,488 | 12.0% | 112 | 25 | $1,568 | $14.00 | $15.00 | 10.8% | 3 | $523 |

## Source Settings Export

| Source | Status | Multiplier | Opportunities | Bids | Bid Rate | Impressions | Clicks | Leads | Win Rate | CPC | Profit |
|--------|--------|------------|---------------|------|----------|-------------|--------|-------|----------|-----|--------|
| QuoteWizard | Active | 100% | 3,200 | 640 | 20.0% | 70 | 29 | 12 | 10.9% | $14.20 | $310 |
| EverQuote | Active | 100% | 2,800 | 700 | 25.0% | 84 | 35 | 15 | 12.0% | $13.80 | $445 |
| MediaAlpha | Active | 100% | 2,100 | 420 | 20.0% | 46 | 19 | 8 | 11.0% | $14.50 | $120 |
| SmartFinancial | Active | 100% | 1,500 | 300 | 20.0% | 30 | 12 | 5 | 10.0% | $14.00 | $85 |
| InsuranceLeads | Active | 105% | 900 | 225 | 25.0% | 28 | 12 | 5 | 12.4% | $14.30 | $110 |

## RightPricing Report (Auto — FL, Trailing 7 Days)

| Source Type | 1st Position RPC | 2nd Position RPC | 3rd+ Position RPC |
|-------------|------------------|------------------|-------------------|
| QuoteWizard | $24.00 | $20.00 | $16.00 |
| EverQuote | $22.50 | $18.50 | $15.00 |
| MediaAlpha | $25.00 | $21.00 | $17.00 |
| SmartFinancial | $23.00 | $19.00 | $15.50 |
| InsuranceLeads | $21.50 | $18.00 | $14.50 |

## Account Context
- Vertical: Auto Insurance
- State: Florida
- Account Age: 14 months
- Targeting: Statewide

## Expected Analysis Outcomes

1. **Root Cause**: Competitive Positioning (searches stable, win rate down from 22% to ~11%, impression share down from 25% to 12%).
2. **ALL sources show win rate problems** with similar shortfalls (~10-12% win rate across the board). This is the key signal for a base bid raise.
3. **Base bid is the correct lever**, NOT multipliers, because:
   - All multipliers are at or near 100% (no per-source differentiation)
   - All sources show the same problem (systemic underbidding)
   - Base bid of $15.00 is below 3rd position RPC for most sources
   - A single base bid raise to ~$22-24 is simpler than adjusting 5 multipliers
4. **Target base bid calculation**: Should target 1st position. For QuoteWizard (largest source): $24.00 / (100/100) = $24.00 base bid. Round up 5-10% = ~$25-26.
5. **Must show impact on ALL sources** when recommending base bid change.
6. **Client is willing to be aggressive** ("whatever it takes", "budget not a constraint") — could recommend the 2× Base Bid Strategy with day parting.
7. **Partial month projection**: March projected leads = (25/25) × 31 = 31 leads.
