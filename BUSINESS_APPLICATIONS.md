# Autoresearch: 10x Improvements & Business Applications

## The Core Insight Worth Millions

Autoresearch isn't really about ML training. It's a **general-purpose autonomous optimization framework** disguised as a research tool. The pattern is:

```
1. Define a single, measurable metric (val_bpb)
2. Constrain the search space (one file, fixed time budget)
3. Let an AI agent iterate autonomously (modify → test → keep/discard)
4. Track everything (git + TSV)
5. Never stop
```

This is **evolutionary optimization with AI as the mutation engine**. And it applies to virtually any business problem where you can define "better" numerically.

---

## Part 1: Making Autoresearch 10x Better (Without More Complexity or Cost)

### 1. Multi-Objective Pareto Frontier Tracking

**Current:** Single metric (val_bpb). Agent optimizes one number.

**10x version:** Track 2-3 metrics simultaneously (val_bpb, memory, throughput) and let the agent explore the Pareto frontier. No code changes to `train.py` or `prepare.py` needed — this is purely a `program.md` upgrade.

Add to `program.md`:
```
When logging results, also note if a run is Pareto-optimal
(better on ANY metric without being worse on all others).
Maintain a "frontier" section in results.tsv.
Periodically revisit Pareto-optimal runs that traded off
differently and try to combine their approaches.
```

**Why it's 10x:** Right now the agent ignores a run that uses 30% less memory but gets 0.001 worse val_bpb. That run might contain a key insight. Tracking the frontier means you explore more of the solution space.

### 2. Structured Experiment Hypotheses in program.md

**Current:** The agent just tries things. No systematic exploration.

**10x version:** Add a hypothesis tree to `program.md`:

```markdown
## Experiment Strategy

Before each run, write a one-line HYPOTHESIS in your commit message:
"I predict [change X] will [improve/worsen] val_bpb by ~[amount] because [reason]."

After each run, note if you were right or wrong. Use this to build
an internal model of what matters:
- Which hyperparameters have the highest sensitivity?
- Which architectural changes are orthogonal (can be combined)?
- What's the current bottleneck: optimization, architecture, or data utilization?
```

**Why it's 10x:** Transforms random search into directed search. The agent builds a mental model of the loss landscape instead of stumbling through it. Calibrated predictions compound — after 20 runs the agent knows which levers actually matter.

### 3. Checkpoint & Fork Strategy

**Current:** Linear branch. Keep or discard. No ability to explore two promising directions simultaneously.

**10x version:** In `program.md`, allow the agent to maintain 2-3 "checkpoint commits" as named tags and periodically fork from different checkpoints:

```markdown
## Checkpointing

After every 10 experiments, tag the current best as `best-N`
(e.g. best-10, best-20).

Every 5th experiment, instead of iterating on HEAD, pick a random
previous checkpoint and try a completely different direction from there.
This prevents getting trapped in local optima.
```

**Why it's 10x:** Evolutionary algorithms with population diversity outperform greedy hill-climbing. This adds diversity for zero additional compute cost — just git tags.

### 4. Ablation-First Protocol

**Current:** Agent adds things. Complexity grows.

**10x version:** Every 5th experiment must be a *deletion*:

```markdown
## Ablation Requirement

Every 5th experiment MUST be a simplification: remove a component,
reduce a hyperparameter, delete lines of code. If performance is
equal or better, keep it.

The goal is to find the SIMPLEST code that achieves the current
best val_bpb. Complexity is debt. Pay it down regularly.
```

**Why it's 10x:** Prevents the inevitable drift toward Rube Goldberg architectures. The best ML breakthroughs (Transformers, ReLU, dropout) were *simplifications*. This systematically searches for them.

### 5. Cross-Run Analysis Prompts

**Current:** Agent looks at the last run. Doesn't analyze patterns across runs.

**10x version:** Every 20 experiments, prompt the agent to analyze `results.tsv` holistically:

```markdown
## Periodic Analysis (every 20 runs)

Pause and analyze results.tsv:
1. What's the cumulative improvement so far?
2. Which categories of changes produced the biggest gains?
3. Which categories consistently failed?
4. Are there diminishing returns? If so, try a radically different approach.
5. What combinations of successful changes haven't been tried?
```

**Why it's 10x:** Pattern recognition across experiments. A human researcher does this naturally ("hmm, all my LR changes helped but architecture changes didn't — maybe I should focus on optimization"). The agent needs to be told to do it.

### 6. Gradient of Confidence

**Current:** Binary keep/discard based on a single 5-minute run.

**10x version:** For borderline results (within 0.5% of current best), run twice and average:

```markdown
## Statistical Significance

If a run's val_bpb is within 0.5% of the current best (better OR worse),
run it again to confirm. Noise exists. Don't keep a lucky run or
discard an unlucky one. Two consistent results > one surprising result.
```

**Why it's 10x:** Reduces false positives/negatives. At the frontier where improvements are small, noise dominates. A 10-minute confirmation run saves you from building on a fluke.

---

## Part 2: Applying This to Business Problems

The autoresearch pattern is: **constrained autonomous iteration against a measurable objective**. Here's how it maps to business domains.

---

### For Executives: Strategic Decision Simulation

**The Problem:** Executives make high-stakes decisions with incomplete information. They can't A/B test a reorg or a market entry.

**The Autoresearch Pattern Applied:**

```
metric:       = financial model output (NPV, IRR, payback period)
search_space: = the assumptions in your financial model (one spreadsheet)
time_budget:  = 2 minutes per simulation run
agent_action: = modify assumptions, run model, evaluate output
```

**Concrete Implementation:**

Create a `strategy_model.py` (equivalent to `train.py`) that encodes your business model — revenue projections, cost structure, market sizing, competitive dynamics. The agent modifies assumptions and parameters, runs the model, and tracks which combinations produce the best outcomes.

```markdown
# program.md for Executive Strategy

You are optimizing a business strategy model. The metric is
risk-adjusted NPV over 5 years.

Each run: modify assumptions in strategy_model.py, run the
simulation, record the output.

Explore:
- Pricing strategies (±20% around current)
- Market entry sequencing (which segments first?)
- Hiring pace vs. revenue growth alignment
- Capital allocation across product lines

Constraints:
- Total headcount cannot exceed 500
- Burn rate must stay under $X/month
- Must maintain >6 months runway at all times
```

**Why it's 10x for executives:** Instead of debating assumptions in a boardroom, you explore 100 assumption-sets overnight. You wake up knowing: "If we're wrong about market size by 30%, our strategy still works IF we also adjust pricing by 15%." That's insight you can't get from a single spreadsheet review.

**Attacking the Constraint:** The constraint for executives is *uncertainty in assumptions*. This method doesn't eliminate uncertainty — it maps the entire landscape of "what happens if we're wrong about X?" and finds strategies that are robust across many scenarios.

---

### For Sales: Outreach Sequence Optimization

**The Problem:** Sales teams send sequences of emails/calls. Open rates, reply rates, and conversion rates vary wildly. Most teams pick one sequence and stick with it.

**The Autoresearch Pattern Applied:**

```
metric:       = reply rate (or meeting-booked rate)
search_space: = email copy, subject lines, timing, sequence structure
time_budget:  = send batch, wait 24-48 hours, measure
agent_action: = rewrite sequence, deploy to test cohort, measure response
```

**Concrete Implementation:**

```markdown
# program.md for Sales Sequence Optimization

You are optimizing a cold outreach sequence. The metric is
positive reply rate (replies that express interest / total sent).

Each experiment:
1. Modify the sequence (subject, body, timing, number of touches)
2. Deploy to a test cohort of 50 prospects
3. After 48 hours, measure positive reply rate
4. Log results

Explore:
- Subject line variations (question vs. statement vs. personalized)
- Email length (2 sentences vs. 5 sentences vs. detailed)
- Value proposition framing (ROI vs. pain vs. social proof)
- Sequence cadence (daily vs. 2-day vs. weekly)
- Number of touches (3 vs. 5 vs. 7)

Constraints:
- Must stay compliant with CAN-SPAM
- Cannot make false claims
- Must maintain brand voice (professional, not pushy)
```

**Why it's 10x for sales:** Most sales teams test 2-3 sequences per quarter manually. This tests 2-3 per *day*. Within a month you've explored the entire space of what works for your ICP. And because the agent tracks what worked, it builds compounding knowledge: "Short subject lines + pain-focused body + 3-day cadence outperforms everything else for enterprise prospects."

**Attacking the Constraint:** The constraint in sales is *not knowing what messaging resonates*. Instead of relying on rep intuition or copying competitors, you systematically explore the space and converge on what actually works for YOUR prospects.

---

### For Marketing: Landing Page & Ad Copy Optimization

**The Problem:** Marketing teams spend weeks debating copy, design, and CTAs. Then they launch one version and hope.

**The Autoresearch Pattern Applied:**

```
metric:       = conversion rate (or cost per acquisition)
search_space: = page copy, layout structure, CTA text, headline
time_budget:  = generate variant, deploy, measure for 24 hours
agent_action: = rewrite page elements, deploy A/B variant, measure conversion
```

**Concrete Implementation:**

```markdown
# program.md for Landing Page Optimization

You are optimizing a landing page. The metric is visitor-to-signup
conversion rate.

Each experiment:
1. Modify landing_page.html (the single file you edit)
2. Deploy as B variant in the A/B test
3. After 24 hours (or 500 visitors), measure conversion rate
4. If improved, promote to new A variant
5. If not, discard and try something else

Explore:
- Headlines (benefit-driven vs. curiosity vs. social proof)
- CTA button text and color
- Page length (short vs. long-form)
- Trust signals (logos, testimonials, stats)
- Form fields (fewer = higher conversion but lower quality?)

Track separately:
- Conversion rate (primary)
- Bounce rate
- Time on page
- Signup quality (do these leads actually convert to paid?)
```

**Why it's 10x for marketing:** The same "modify one file, measure, keep/discard" loop. Instead of 4 A/B tests per quarter, you run 4 per week. The compounding effect is enormous — after a month, your landing page has been through 16+ iterations, each building on proven winners.

**Attacking the Constraint:** The constraint in marketing is *iteration speed*. Traditional A/B testing is slow because humans design variants, debate them, and wait for statistical significance. Automating variant generation (AI writes the copy) and shrinking the feedback loop (smaller test cohorts, faster decisions) breaks the constraint.

---

### For Product: Feature Prioritization via Impact Modeling

**The Problem:** Product teams debate which features to build. Everyone has opinions. Nobody has data until after the feature ships.

**The Autoresearch Pattern Applied:**

```
metric:       = predicted user engagement / retention lift
search_space: = feature combinations, implementation scope, rollout order
time_budget:  = model simulation (minutes) or limited beta test (days)
agent_action: = modify feature spec, simulate/test impact, measure
```

**Concrete Implementation:**

Build a user behavior model (even a simple one based on historical data) that predicts how engagement changes when you add/modify features. Let the agent explore the space of "what if we built X but scoped it to Y and combined it with Z?"

```markdown
# program.md for Feature Prioritization

You optimize a product roadmap model. The metric is predicted
12-month retention lift per engineering-month invested.

Each experiment:
1. Select a feature combination and scope level
2. Run the impact model
3. Record predicted retention lift and estimated eng cost
4. Track Pareto frontier (max impact per eng-month)

Constraints:
- Total engineering budget: 6 engineers × 6 months = 36 eng-months
- Must include at least 1 infrastructure/debt item
- Cannot exceed 3 concurrent projects
```

**Attacking the Constraint:** The constraint is *opportunity cost* — you can't build everything, so which combination of features maximizes impact? The agent explores thousands of combinations and finds the Pareto-optimal roadmap.

---

### For Finance/Operations: Cost Structure Optimization

**The Problem:** Companies have complex cost structures with many interdependent levers. Cutting one cost often increases another.

**The Autoresearch Pattern Applied:**

```
metric:       = profit margin (or unit economics)
search_space: = vendor choices, staffing models, process parameters
time_budget:  = model simulation (seconds)
agent_action: = modify cost model parameters, run simulation, measure margin
```

**Why it works:** Cost optimization has the same structure as ML hyperparameter search — lots of interdependent variables, nonlinear interactions, and a clear objective function. The agent can explore "what if we outsource X but bring Y in-house while switching vendor Z?" far faster than a human can model in Excel.

---

### For Customer Success: Churn Prevention

**The Problem:** Churn prediction models exist, but the *intervention* side is under-optimized. What message, at what time, through what channel, prevents churn best?

**The Autoresearch Pattern Applied:**

```
metric:       = save rate (churning customers retained / total at-risk)
search_space: = intervention playbook (message, timing, channel, offer)
time_budget:  = deploy intervention, measure for 7 days
agent_action: = modify playbook, deploy to at-risk cohort, measure saves
```

---

## Part 3: The Meta-Pattern (What Makes This Work Anywhere)

### The Five Requirements

For the autoresearch pattern to work on any problem, you need:

1. **A measurable metric** — one number that means "better." Without this, the agent can't decide keep/discard. Most business problems CAN be reduced to a number; people just don't bother.

2. **A constrained search space** — "one file to edit." If the agent can change anything, it changes everything and you can't learn from the results. Constraining the search space forces focused iteration.

3. **A fast feedback loop** — 5 minutes in autoresearch. The faster you can evaluate a change, the more iterations you get, and iterations are the whole game. If your feedback loop is 6 months (annual planning), you get 2 iterations per year. If it's 1 day, you get 365.

4. **Reversibility** — git reset. If a change makes things worse, you must be able to undo it cheaply. This is what makes aggressive experimentation safe.

5. **Automated execution** — the agent runs without human intervention. The moment a human is in the loop for every iteration, you're bottlenecked by human availability and attention.

### The Constraint-Attack Framework

Every business problem has a binding constraint — the one thing that, if relaxed, would improve everything:

| Domain | Typical Binding Constraint | How Autoresearch Attacks It |
|--------|---------------------------|---------------------------|
| Executive Strategy | Uncertainty in assumptions | Explore 100s of assumption-sets, find robust strategies |
| Sales | Unknown message-market fit | Test 100s of sequences, converge on what resonates |
| Marketing | Slow iteration cycles | Automate variant generation, shrink test windows |
| Product | Opportunity cost of wrong bets | Simulate 1000s of feature combos, find Pareto optimal |
| Finance | Interdependent cost levers | Model all interactions, find global optimum |
| Customer Success | Unknown intervention effectiveness | Test playbook variants systematically |
| Hiring | Candidate evaluation noise | Structured scoring with rapid calibration loops |
| Pricing | Fear of leaving money on the table | Micro-experiments across segments and tiers |

### The 10x Leverage Points (Applicable Everywhere)

1. **Speed of iteration beats quality of iteration.** 100 mediocre experiments > 3 perfect ones. The search finds good solutions; human brilliance is not required at each step.

2. **Constraints are features, not bugs.** The 5-minute time budget isn't a limitation — it's what makes results comparable. In business: fixed test cohort sizes, fixed time windows, fixed budgets are what make experiments meaningful.

3. **The human programs the organization, not the work.** In autoresearch, the human writes `program.md`, not `train.py`. In business, the executive should define the metric, constraints, and search space — then let the system iterate. Don't micromanage the experiments.

4. **Keep/discard is the most underrated decision framework.** Most businesses accumulate complexity because they never discard. The autoresearch pattern forces a binary: did this make things better? No? Revert. This alone would transform most organizations.

5. **Compounding iteration is the real moat.** After 100 experiments, you don't just have a better model — you have 100 data points about what works and what doesn't. That *knowledge* is the real asset. Apply this to sales sequences: after 100 variations, you know more about your market's preferences than any competitor who tested 5.

---

## Part 4: Where to Start (Practical Next Steps)

### Lowest-Hanging Fruit for Business Application

**Start with something that has ALL five requirements already:**

1. **Email subject line optimization** — metric: open rate. Search space: subject line text. Feedback: 24 hours. Reversible: just send different next time. Automated: AI writes variants.

2. **Pricing page copy** — metric: conversion rate. Search space: headline + CTA + pricing display. Feedback: 48 hours with enough traffic. Reversible: swap back. Automated: AI generates variants.

3. **Internal process documentation** — metric: time-to-completion for new hires following the docs. Search space: the documentation itself. Feedback: each new hire is a "run." Reversible: version control. Automated: AI rewrites unclear sections based on where people get stuck.

### Building the Platform

To generalize autoresearch into a business tool, you'd need:

```
business-autoresearch/
├── program.md          # Domain-specific instructions (human writes)
├── experiment.py       # The thing being optimized (agent edits)
├── evaluate.py         # Fixed metric calculation (do not modify)
├── results.tsv         # Experiment log
└── constraints.yaml    # Guardrails (budget, compliance, brand, etc.)
```

The pattern is identical. Only the domain changes.

---

## Summary

Autoresearch is a 3-file embodiment of a powerful idea: **autonomous iteration against a fixed metric within fixed constraints**. It's currently applied to ML training, but the pattern is domain-agnostic. The 10x improvements come not from making the code more complex, but from making the *iteration smarter* — better experiment selection, better analysis of past results, better exploration of the search space. And the business applications come from recognizing that every business function has optimization problems with the same structure: a metric to improve, a constrained set of levers, and a feedback loop that can be shortened.

The companies that figure out how to run 100 experiments while their competitors run 3 will win. Not because any single experiment is brilliant, but because the compound effect of 100 keep/discard decisions is unstoppable.
