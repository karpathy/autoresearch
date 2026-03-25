---
name: campaign-analysis
description: This skill should be used when the user invokes "campaign-analysis" directly, asks to "analyze a GOAL client campaign", "diagnose campaign performance", "generate bid recommendations", "review campaign data", "what's wrong with this campaign", or provides GOAL Platform CSV data and asks for optimization recommendations. Also triggers when the user mentions bid multipliers, source-level diagnosis, win rate analysis, RightPricing calibration, or GOAL Platform performance troubleshooting for any client account.
---

# GOAL Platform Campaign Analysis

Diagnose GOAL Platform client campaign performance and generate calibrated bid optimization recommendations using a systematic four-layer analytical framework. The final output is a polished, shareable HTML report.

## Framework Overview

The analysis operates across four layers, each building on the previous:

| Layer | Name | Purpose |
|-------|------|---------|
| 1 | Client Context | Parse the stated problem, willingness to adjust, and implicit constraints |
| 2 | Campaign Trends | Build monthly trend tables, calculate baselines, classify root cause |
| 3 | Source Diagnosis | Drill into source settings to classify each source's problem type |
| 4 | Market Calibration | Cross-reference effective bids against RightPricing market data |

**Critical rule:** Source-level performance data alone is insufficient for bid recommendations. Without market pricing context (RightPricing report), it is impossible to determine whether the issue is bid amount, bid rate, targeting, or filter settings. Always request RightPricing data before finalizing bid recommendations.

**Bid lever neutrality:** When analysis indicates a bid increase is needed, always evaluate both raising the base bid and adjusting source multipliers. Choose the lever that best fits the data — do not default to either. See the Base Bid vs. Multiplier Decision Framework in `references/analysis-framework.md`.

## Workflow

### Step 1: Gather Inputs

Before beginning analysis, collect the following from the user:

**Required:**
- Monthly Campaign CSVs (minimum 3 months, 6 ideal) — exported from the GOAL dashboard containing Searches, Impressions, Clicks, Leads, Calls, CTR, Cost, Avg CPC, Avg Bid, Avg Pos, Win Rate, Sold, Sold CVR, Sold CPA, Sold Rev
- Client message or context — the stated problem, questions, or goals

**Strongly Recommended:**
- Source Settings Export — CSV showing all traffic sources with Status, Source Name, Bid Multiplier, Opportunities, Bids, Bid Rate, Impressions, Clicks, Leads, Calls, Win Rate, Sold, CPC, Profit, CM
- RightPricing Report — market-level pricing showing RPC at each ad wall position (1st, 2nd, 3rd+) broken down by source type. **This is the single most important input for calibrating bid recommendations.**

**Optional:**
- Ad Group Settings (targeting filters, age ranges, coverage types)
- Change History (platform changes made by the agent)
- Competitive Context (competitor activity, new market entrants)
- Client disposition data (contacted, quoted, bound from their CRM)

If data is missing, note which layers will be limited and proceed with what is available. Flag missing RightPricing data as a critical gap.

### Step 2: Execute the Four-Layer Analysis

Work through each layer sequentially. Consult `references/analysis-framework.md` for detailed procedures, diagnostic matrices, and decision trees at each layer.

**Layer 1 — Client Context Parsing:**
- Extract the stated problem (low volume, high CPA, declining ROAS, etc.)
- Identify willingness to adjust (bid, budget, targeting, sources)
- Note implicit constraints — respect the scope of the client's ask

**Layer 2 — Campaign Trend Analysis:**
- Build monthly trend table from CSVs
- Handle partial months using projection formula
- Calculate baselines (averages across complete months)
- Quantify performance gaps (lead volume, win rate, impression share, spend)
- Classify root cause using the diagnostic matrix (see `references/diagnostics-and-formulas.md`)
- For accounts under 90 days, use leading indicators (Contact Rate, Quote Rate) instead of Sold CPA

**Layer 3 — Source-Level Diagnosis:**
- Filter to active sources with Opportunities > 0
- Check for ad group cannibalization (geographic overlap)
- Calculate effective bids: Base Bid × (Multiplier / 100)
- Rank sources by opportunity volume (focus impact on high-volume sources)
- Classify each source's problem type using the source diagnostic matrix
- Evaluate off-hours bidding opportunity for budget-constrained clients

**Layer 4 — Market Pricing Calibration:**
- Map each source's effective bid to RightPricing market positions
- Cross-reference position with source problem type to generate the recommendation
- **Determine the optimal bid lever** — base bid raise vs. source multiplier adjustment. Evaluate both options using the decision framework in `references/analysis-framework.md` and choose based on the data pattern.
- Calculate target values for the chosen lever: Target Multiplier = (Target RPC / Base Bid) × 100%, OR Target Base Bid = Target RPC / (Source Multiplier / 100)
- Apply 20–30% geo-buffer if targeting is severely restricted
- Round up 5–10% for bid volatility

### Step 3: Generate Recommendations

Categorize all recommendations into exactly three named groups. **All three groups MUST appear as distinct, labeled sections in the output — even if a group has no items** (in which case, state "None warranted" or "No changes needed" under that heading).

1. **Bid Adjustments** — Either base bid changes or source multiplier changes (or both), chosen based on the data pattern. For each recommendation, include the current value, recommended value, resulting effective bid, target market position, and a brief rationale for why this lever was chosen over the alternative. Apply proportional incremental changes rather than arbitrary flat increases. If no bid changes are warranted, include this section with an explicit "No bid adjustments recommended" note and brief explanation.

2. **Investigation Items** — Anomalies that cannot be resolved by bid changes alone. Flag for follow-up with the client or GOAL product team.

3. **Hold Steady** — Sources performing well that should not be changed. Explain why leaving them alone is deliberate, not an oversight.

**Prioritize by impact:** Rank recommendations by Volume Impact Potential = Opportunities × (Target Bid Rate − Current Bid Rate) × Expected Win Rate.

**Respect client constraints:** Map recommendations back to Layer 1 willingness. Do not recommend 3× increases if the client said "a little."

### Step 4: Build the Output Report

Generate a polished, standalone HTML file following the template in `references/output-template.html`. The report should be professional enough for client-facing delivery.

**Report structure:**
1. Executive Summary — one-paragraph diagnosis with root cause classification
2. Campaign Trend Dashboard — monthly metrics table with baseline comparisons and gap indicators
3. Source Performance Matrix — source-by-source breakdown with problem classification and current vs. recommended bids
4. Market Position Map — effective bids plotted against RightPricing positions (when data available)
5. Recommendations Table — prioritized by impact, categorized into bid adjustments (base bid and/or multiplier changes), investigations, and hold-steady
6. Methodology Notes — brief explanation of the analysis framework for client transparency

**Styling:**
- GOAL brand blue (#0479DF) as primary accent
- Clean, professional typography (Inter or system sans-serif)
- Color-coded status indicators (green = performing, amber = investigate, red = action needed)
- Responsive design that works on screen and prints cleanly
- All CSS inlined — no external dependencies

### Step 5: Present and Iterate

Share the HTML report. Offer to:
- Adjust specific recommendations based on feedback
- Generate a PDF version for formal delivery
- Deep-dive into any individual source or metric
- Re-run the analysis with updated data

## Key Formulas

| Formula | Calculation |
|---------|-------------|
| Effective Bid | Base Bid (Avg Bid from CSV) × (Source Multiplier / 100) |
| Target Multiplier | (Target RPC / Base Bid) × 100% |
| Target Base Bid | Target RPC / (Source Multiplier / 100) |
| Projected Monthly Value | (Actual Value / Days Elapsed) × Total Days in Month |
| Impression Share | Impressions / Searches (from campaign CSV) |
| Bid Rate | Bids / Opportunities (from source settings) |
| Win Rate | Wins (Impressions) / Bids (from source settings) |
| Volume Impact Potential | Opportunities × (Target Bid Rate − Current Bid Rate) × Expected Win Rate |

## Critical Guardrails

- **Never recommend bid increases without RightPricing context.** A source appearing to need a higher bid may already be above first position — the problem is elsewhere.
- **Do not treat all sources equally.** A 10% change on a 2,000-opportunity source is vastly more impactful than a 50% change on a 40-opportunity source.
- **Watch for bid rate problems.** A low bid rate means the source barely enters auctions. Raising the bid (whether base bid or multiplier) for a low-bid-rate source only increases cost on the tiny fraction of auctions entered — the fix is upstream (targeting, filters, ad group settings).
- **Never judge account health by blended CPA alone.** A high blended CPA is often caused by one un-optimized Premium Referral source. Always isolate Paid Search performance first.
- **Multiplier math matters.** Setting multiplier to 20% means bidding at 20% of the base bid, not a 20% increase. A 20% increase requires setting to 120%.

## Reference Files

- **`references/analysis-framework.md`** — Detailed Layer 1–4 procedures, diagnostic decision trees, maturity phase analysis for accounts under 90 days, off-hours bidding strategy, and the 2× Base Bid advanced tactic
- **`references/diagnostics-and-formulas.md`** — Root cause classification matrix, source problem diagnostic matrix, market position mapping table, recommendation crossover matrix, common pitfalls with explanations, and complete analysis checklist
- **`references/output-template.html`** — Base HTML template for the final report artifact with GOAL branding, responsive layout, and print styles
