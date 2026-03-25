# Test Scenario 4: Blended CPA Trap — Premium Referral Skewing Metrics

## Client Message
"Our CPA is way too high at $780. We need to cut costs. Should we lower all our bids across the board?"

## Campaign CSVs (4 months)

| Month | Searches | Impressions | Imp % | Clicks | Leads | Cost | Avg CPC | Avg Bid | Win Rate | Sold | Sold CPA |
|-------|----------|-------------|-------|--------|-------|------|---------|---------|----------|------|----------|
| Dec 2025 | 11,200 | 2,240 | 20.0% | 168 | 37 | $3,360 | $20.00 | $21.00 | 16.5% | 5 | $672 |
| Jan 2026 | 11,500 | 2,300 | 20.0% | 173 | 38 | $3,460 | $20.00 | $21.00 | 16.8% | 4 | $865 |
| Feb 2026 | 11,100 | 2,109 | 19.0% | 158 | 35 | $3,160 | $20.00 | $21.00 | 15.9% | 4 | $790 |
| Mar 2026 (20 days) | 7,200 | 1,368 | 19.0% | 103 | 23 | $2,060 | $20.00 | $21.00 | 15.5% | 3 | $687 |

## Source Settings Export

| Source | Status | Multiplier | Opportunities | Bids | Bid Rate | Impressions | Clicks | Leads | Win Rate | CPC | Profit |
|--------|--------|------------|---------------|------|----------|-------------|--------|-------|----------|-----|--------|
| QuoteWizard | Active | 100% | 1,800 | 360 | 20.0% | 90 | 38 | 16 | 25.0% | $18.50 | $920 |
| EverQuote | Active | 105% | 1,400 | 350 | 25.0% | 98 | 41 | 17 | 28.0% | $19.00 | $780 |
| Datalot Premium | Active | 150% | 600 | 180 | 30.0% | 54 | 23 | 10 | 30.0% | $28.50 | -$380 |
| SmartFinancial | Active | 95% | 950 | 190 | 20.0% | 38 | 16 | 7 | 20.0% | $19.80 | $210 |

## RightPricing Report (Auto — GA, Trailing 7 Days)

| Source Type | 1st Position RPC | 2nd Position RPC | 3rd+ Position RPC |
|-------------|------------------|------------------|-------------------|
| QuoteWizard | $22.00 | $18.50 | $15.00 |
| EverQuote | $21.00 | $17.50 | $14.00 |
| Datalot Premium | $35.00 | $30.00 | $25.00 |
| SmartFinancial | $20.50 | $17.00 | $13.50 |

## Account Context
- Vertical: Auto Insurance
- State: Georgia
- Account Age: 11 months
- Targeting: Atlanta metro + surrounding counties

## Expected Analysis Outcomes

1. **DO NOT lower all bids across the board** — the client's framing is wrong. Must push back with data.
2. **Blended CPA is misleading**: Paid Search sources (QuoteWizard, EverQuote, SmartFinancial) are performing well with solid win rates and positive profit. The high blended CPA is driven almost entirely by Datalot Premium.
3. **Isolate by source type**: Paid Search blended CPA is healthy (~$450-550 range). Datalot Premium is the outlier dragging the average up with a $28.50 CPC at 150% multiplier.
4. **Datalot Premium diagnosis**: Effective bid = $21.00 × 1.50 = $31.50. RightPricing shows 1st position at $35.00. The source is between 1st and 2nd position — bid is not unreasonable for the source type, but premium referral sources inherently carry higher CPAs. The question is whether the lifetime value justifies the cost.
5. **Correct recommendation**: Do NOT cut bids on performing sources. Either (a) reduce the Datalot Premium multiplier from 150% to ~120% to bring it closer to 2nd position, or (b) pause Datalot Premium entirely if the client's goal is pure CPA reduction. Present data showing Paid Search performance is strong.
6. **Pitfall #6**: Must avoid judging account health by blended CPA alone.
7. **Geo-buffer**: Atlanta metro + surrounding = moderate restriction. May need 10-15% buffer if applicable.
