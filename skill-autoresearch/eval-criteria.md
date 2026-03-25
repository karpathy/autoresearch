# Eval Criteria for campaign-analysis Skill

Every criterion is binary: PASS or FAIL. No sliding scales.

These criteria are applied to the skill's output (the HTML report + reasoning) for each test scenario. The eval agent reads the scenario's "Expected Analysis Outcomes" and checks the skill output against each applicable criterion.

---

## Core Analytical Criteria

### C1: Correct Root Cause Classification
Does the analysis correctly classify the root cause using the diagnostic matrix (Competitive Positioning, Market Demand Decline, Budget Constraint, or Mixed) based on the relationship between Searches, Win Rate, and Impression Share trends?

**FAIL if**: Root cause is wrong, missing, or the analysis doesn't explicitly state a classification.

### C2: Correct Effective Bid Calculation
Are effective bids calculated correctly as Base Bid × (Multiplier / 100) for each source? Verify at least 2 source calculations are present and mathematically correct.

**FAIL if**: Any effective bid calculation is wrong, or effective bids are not shown.

### C3: Correct Source Problem Classification
Does the analysis correctly classify each source's problem type (Bid Rate Problem, Win Rate Problem, Performing Well, Not Bidding) based on the source diagnostic matrix?

**FAIL if**: Any source is misclassified in a way that would change the recommendation (e.g., a bid rate problem source classified as a win rate problem).

### C4: Correct Bid Lever Selection
When a bid increase is needed, does the analysis evaluate both base bid and multiplier options and select the correct lever based on the decision framework? (Multiplier for source-specific issues, base bid for systemic underbidding, both when appropriate.)

**FAIL if**: The wrong lever is chosen, or the analysis defaults to one lever without evaluating both.

### C5: No Bid Increase for Bid Rate Problems
Does the analysis correctly avoid recommending bid increases (base bid or multiplier) for sources with bid rate problems (<15% bid rate)?

**FAIL if**: A bid increase is recommended for any source where the primary problem is bid rate.

---

## Guardrail Criteria

### G1: RightPricing Gap Flagged
When RightPricing data is NOT provided, does the analysis explicitly flag this as a critical gap and qualify that bid recommendations are preliminary?

**FAIL if**: RightPricing is missing and the analysis makes definitive bid recommendations without flagging the gap. (N/A if RightPricing data is provided.)

### G2: No Blended CPA Misjudgment
Does the analysis avoid judging overall account health solely by blended CPA? When a premium referral source is skewing the average, does the analysis isolate source types and identify the real driver?

**FAIL if**: The analysis treats a high blended CPA as a universal account problem without isolating which sources are driving it. (N/A if no premium referral sources are present.)

### G3: Account Maturity Respected
For accounts under 90 days, does the analysis use leading indicators (Contact Rate, Quote Rate) instead of Sold CPA to assess account health?

**FAIL if**: The analysis makes CPA-based health judgments on an account under 90 days when disposition data is available showing leading indicators. (N/A if account is >90 days.)

### G4: Client Constraints Respected
Do recommendations stay within the scope of what the client expressed willingness to do? (e.g., "a little" = incremental, not 2× increases)

**FAIL if**: Recommendations exceed the client's stated constraints without explicitly flagging them as "beyond stated scope" options.

### G5: Ad Group Cannibalization Detected
When multiple ad groups have overlapping geographic targeting, does the analysis identify and flag this?

**FAIL if**: Overlapping ad groups exist in the data and the analysis does not mention the overlap. (N/A if only one ad group or no overlap.)

---

## Output Quality Criteria

### O1: HTML Report Structure Complete
Does the output include all required report sections: (1) Executive Summary with root cause badge, (2) Key Metrics Snapshot, (3) Campaign Trend Dashboard, (4) Source Performance Matrix, (5) Market Position Map or gap callout, (6) Recommendations, (7) Methodology?

**FAIL if**: Any required section is missing from the HTML report.

### O2: Recommendations Properly Categorized
Are recommendations categorized into the three required groups: Bid Adjustments, Investigation Items, and Hold Steady?

**FAIL if**: Recommendations are presented as a flat list without categorization, or any category is missing when it should be present.

### O3: Partial Month Projected
When the current month has incomplete data, does the analysis project the full month using the formula and clearly label it as a projection?

**FAIL if**: Partial month data is presented as-is without projection, or the projection is present but not labeled. (N/A if no partial month in the data.)

---

## Scoring

- Run each scenario through the skill
- Apply all applicable criteria (skip N/A ones)
- Score = (PASS count) / (Applicable criteria count) × 100%
- Overall score = average across all scenarios

**Target**: 90%+ overall pass rate
