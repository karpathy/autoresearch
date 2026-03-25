# Test Scenario 5: Bid Rate Problem Disguised as Volume Issue

## Client Message
"We have tons of opportunity volume showing up but barely any leads. Raising our bids hasn't helped. What's going on?"

## Campaign CSVs (3 months)

| Month | Searches | Impressions | Imp % | Clicks | Leads | Cost | Avg CPC | Avg Bid | Win Rate | Sold | Sold CPA |
|-------|----------|-------------|-------|--------|-------|------|---------|---------|----------|------|----------|
| Jan 2026 | 22,000 | 1,100 | 5.0% | 83 | 18 | $1,826 | $22.00 | $24.00 | 4.2% | 2 | $913 |
| Feb 2026 | 21,500 | 860 | 4.0% | 65 | 14 | $1,430 | $22.00 | $24.00 | 3.5% | 2 | $715 |
| Mar 2026 (22 days) | 14,800 | 518 | 3.5% | 39 | 8 | $858 | $22.00 | $24.00 | 3.0% | 1 | $858 |

## Source Settings Export

| Source | Status | Multiplier | Opportunities | Bids | Bid Rate | Impressions | Clicks | Leads | Win Rate | CPC | Profit |
|--------|--------|------------|---------------|------|----------|-------------|--------|-------|----------|-----|--------|
| QuoteWizard | Active | 120% | 4,500 | 135 | 3.0% | 41 | 17 | 7 | 30.4% | $22.50 | $180 |
| MediaAlpha | Active | 115% | 3,800 | 76 | 2.0% | 24 | 10 | 4 | 31.6% | $23.00 | $60 |
| EverQuote | Active | 110% | 2,900 | 145 | 5.0% | 48 | 20 | 9 | 33.1% | $21.50 | $290 |
| SmartFinancial | Active | 100% | 1,200 | 48 | 4.0% | 15 | 6 | 3 | 31.3% | $22.00 | $45 |

## RightPricing Report (Home — AZ, Trailing 7 Days)

| Source Type | 1st Position RPC | 2nd Position RPC | 3rd+ Position RPC |
|-------------|------------------|------------------|-------------------|
| QuoteWizard | $26.00 | $22.00 | $18.00 |
| MediaAlpha | $27.50 | $23.00 | $19.00 |
| EverQuote | $24.50 | $20.50 | $17.00 |
| SmartFinancial | $23.00 | $19.50 | $16.00 |

## Account Context
- Vertical: Home Insurance
- State: Arizona
- Account Age: 6 months
- Ad Groups: 2 ad groups
  - "AZ State" — targets all Arizona zip codes
  - "Phoenix Metro" — targets Phoenix, Scottsdale, Tempe, Mesa zip codes
- Targeting filters: Property type "Doesn't Equal" Condo, "Doesn't Equal" Townhome, "Doesn't Equal" Mobile Home

## Expected Analysis Outcomes

1. **Primary problem is BID RATE, not bid amount.** All sources show 2-5% bid rate despite 12,000+ total opportunities. Win rates are actually healthy (30%+) when bids DO enter auctions.
2. **DO NOT recommend raising bids** — bids are already competitive. QuoteWizard effective bid = $24.00 × 1.20 = $28.80, ABOVE 1st position RPC ($26.00). Same pattern across sources.
3. **Ad Group Cannibalization DETECTED**: "AZ State" targets all of Arizona while "Phoenix Metro" targets specific Phoenix-area zip codes within Arizona. These overlap and bid against each other, suppressing bid rate.
4. **Over-filtering detected**: Using "Doesn't Equal" for Condo, Townhome, AND Mobile Home significantly restricts the available opportunity pool. This is a case of the "Doesn't Equal" filtering trap (Pitfall #9).
5. **Recommendations should be**:
   - INVESTIGATE: Fix ad group geographic overlap (remove Phoenix zips from AZ State, or consolidate to one ad group)
   - INVESTIGATE: Review property type filters — "Doesn't Equal" on 3 property types is likely over-filtering. Consider using "Equals" with the desired types instead.
   - HOLD: Do not touch bids — win rates are strong when the source enters auctions.
6. **This is NOT a bid problem** — the skill must correctly identify the upstream issues and avoid the trap of recommending bid increases.
