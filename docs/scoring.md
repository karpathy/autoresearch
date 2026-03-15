# Writing Scoring Functions

Your scoring function is the most important thing in your problem. It encodes what "better" means. Agents will relentlessly exploit every degree of freedom your metric leaves open, so a precise scoring function produces precise results, and a sloppy one produces garbage that happens to score well.

This document covers how to think about scoring, common patterns, and how to use LLMs as judges for subjective or multi-dimensional problems.

## The basics

`scoring/score.py` exports a single function:

```python
def score():
    return {"score": 42.0}
```

The framework calls `score()`, reads the returned dict, and extracts the primary metric (default key: `"score"`). You can return additional metrics — they're recorded but only the primary one drives optimization.

The function runs in a subprocess with the problem directory as cwd, so imports from `state/` and `context/` work directly:

```python
def score():
    from state.solution import weights
    from context.data import test_set, evaluate
    return {"score": evaluate(weights, test_set)}
```

## Thinking about your metric

Before writing code, answer these questions:

**What does the number mean?** A score should have a clear interpretation. "Rastrigin function value at the solution point" is clear. "Quality" is not. If you can't explain what a 10% improvement means in concrete terms, your metric needs work.

**What can agents exploit?** Agents will find the shortest path to a better number. If your metric is "lines of code" they'll delete everything. If it's "test pass rate" they'll delete the tests. Think adversarially: what's the dumbest way to improve this number? Then close that loophole in the scoring function.

**Is the starting score meaningful?** The initial state should produce a valid, non-trivial score. If the baseline is already zero or infinity, agents have no gradient to follow. Seed `state/` with something reasonable but clearly improvable.

**Is there a known optimum?** If so, set `bounded: true` in `problem.yaml`. This helps the framework reason about convergence. The Rastrigin function has a known global minimum of 0.0; a trading strategy's Sharpe ratio does not have a known maximum.

## Common patterns

### Direct computation

The simplest case — score is a deterministic function of state:

```python
def score():
    from state.solution import x
    from context.problem import rastrigin
    return {"score": rastrigin(x)}
```

Good for: mathematical optimization, constraint satisfaction, algorithmic problems.

### Run-and-measure

Score by executing something and measuring the result:

```python
import subprocess

def score():
    result = subprocess.run(
        ["python", "state/train.py"],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Training failed:\n{result.stderr[-2000:]}")

    # parse metrics from stdout
    import re
    output = result.stdout
    val_loss = float(re.search(r"val_loss:\s+([\d.]+)", output).group(1))
    return {"score": val_loss}
```

Good for: ML training, benchmarks, compilation, anything with a runtime component.

### Multi-metric with a primary

Return several metrics but designate one as primary:

```python
def score():
    from state.solution import model
    from context.benchmark import run_benchmark

    results = run_benchmark(model)
    return {
        "score": results["latency_p99"],  # primary — this drives optimization
        "throughput": results["qps"],      # tracked but doesn't drive merges
        "memory_mb": results["peak_mem"],
    }
```

All metrics are recorded in history. Agents see them on the leaderboard. But only the primary metric determines whether a proposal is an improvement.

### Constraint penalties

When solutions must satisfy hard constraints, penalize violations:

```python
def score():
    from state.tour import tour
    from context.cities import cities, tour_distance

    # hard constraint: must visit every city exactly once
    if sorted(tour) != list(range(len(cities))):
        return {"score": float("inf")}  # invalid — worst possible for minimize

    return {"score": tour_distance(tour)}
```

Returning `inf` (for minimize) or `-inf` (for maximize) is a clear signal. Agents learn quickly that constraint violations are fatal.

### Subprocess with structured output

When scoring requires an external process:

```python
import json
import subprocess

def score():
    result = subprocess.run(
        ["node", "context/evaluate.js"],
        input=open("state/config.json").read(),
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])
    metrics = json.loads(result.stdout)
    return {"score": metrics["lighthouse_score"], **metrics}
```

Good for: non-Python evaluators, browser-based testing, external tools.

## LLM-as-judge scoring

This is where it gets interesting. For subjective or multi-dimensional problems — writing quality, design aesthetics, API ergonomics — there's no formula. But an LLM can evaluate against a rubric you define, and structured outputs guarantee you get parseable scores back.

The pattern:

1. Define dimensions with clear rubrics
2. Have the LLM score each dimension (structured output)
3. Aggregate to a single number using weights the agents never see

The weights are the secret. They encode your values. Weight "originality" at 3x and the swarm converges on bold work. Change the weights tomorrow and the same agents produce something completely different — without changing any instructions.

### Setting up

Add the optional `llm` dependency group:

```bash
uv sync --extra llm
# or
pip install autoanything[llm]
```

This installs both `anthropic` and `openai` SDKs.

### Example: scoring an essay with Claude

```python
from pydantic import BaseModel, Field
import anthropic

# --- Rubric dimensions (agents never see this) ---

class EssayScores(BaseModel):
    argument_structure: int = Field(
        description="1-10: logical flow, clear thesis, supporting evidence"
    )
    evidence_quality: int = Field(
        description="1-10: specific, relevant, accurately cited"
    )
    prose_clarity: int = Field(
        description="1-10: readable, concise, no jargon without purpose"
    )
    originality: int = Field(
        description="1-10: fresh perspective, avoids cliches, surprising insights"
    )
    counterargument_handling: int = Field(
        description="1-10: acknowledges and addresses opposing views"
    )

# --- Hidden weights (this is where your values live) ---

WEIGHTS = {
    "argument_structure": 2.0,
    "evidence_quality": 1.5,
    "prose_clarity": 1.0,
    "originality": 3.0,  # we really value originality
    "counterargument_handling": 1.5,
}


def score():
    essay = open("state/essay.md").read()
    prompt = open("context/rubric_prompt.txt").read()

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"{prompt}\n\n---\n\n{essay}"}],
        output_format=EssayScores,
    )

    scores = response.parsed_output
    dimensions = scores.model_dump()

    # weighted aggregate — agents see the final number, not the weights
    total = sum(dimensions[k] * WEIGHTS[k] for k in WEIGHTS)
    max_possible = sum(10 * w for w in WEIGHTS.values())
    normalized = total / max_possible  # 0.0 to 1.0

    return {"score": round(normalized, 4), **dimensions}
```

Set `direction: maximize` in `problem.yaml`. Agents see a number between 0 and 1 go up. They also see the individual dimension scores on the leaderboard (since we spread them into the return dict), so they know *what* to work on — but they don't know the weights, so they can't game the aggregation.

### Example: scoring with OpenAI

Same pattern, different SDK:

```python
from pydantic import BaseModel, Field
from openai import OpenAI

class CodeReviewScores(BaseModel):
    correctness: int = Field(description="1-10: does the code do what it claims")
    readability: int = Field(description="1-10: clear naming, structure, comments where needed")
    efficiency: int = Field(description="1-10: appropriate algorithms, no unnecessary work")
    robustness: int = Field(description="1-10: handles edge cases, validates inputs")

WEIGHTS = {
    "correctness": 3.0,
    "readability": 1.0,
    "efficiency": 2.0,
    "robustness": 2.0,
}


def score():
    code = open("state/solution.py").read()

    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a code reviewer. Score the following code on each dimension. Be rigorous and consistent.",
            },
            {"role": "user", "content": code},
        ],
        response_format=CodeReviewScores,
    )

    scores = completion.choices[0].message.parsed
    dimensions = scores.model_dump()

    total = sum(dimensions[k] * WEIGHTS[k] for k in WEIGHTS)
    max_possible = sum(10 * w for w in WEIGHTS.values())

    return {"score": round(total / max_possible, 4), **dimensions}
```

### Reducing variance in LLM scoring

LLM judges are noisy. The same essay scored twice might get 7 then 8. This matters because the framework only keeps improvements — noisy scoring means lucky rolls get merged and unlucky good proposals get discarded.

Strategies:

**Multiple samples, take the mean:**

```python
def score():
    essay = open("state/essay.md").read()
    runs = [_judge(essay) for _ in range(3)]
    avg = {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}
    return {"score": _aggregate(avg), **avg}
```

Cost triples, variance drops by ~sqrt(3). Worth it for expensive-to-generate proposals.

**Temperature zero:**

```python
response = client.messages.parse(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    temperature=0,
    messages=[...],
    output_format=EssayScores,
)
```

Not truly deterministic, but reduces variance significantly.

**Comparative rather than absolute scoring:**

Instead of "rate this essay 1-10", compare against a reference:

```python
prompt = f"""Score this essay relative to the reference essay.
A score of 5 means equivalent quality. Above 5 means better, below means worse.

Reference essay:
{open('context/reference_essay.md').read()}

Candidate essay:
{essay}
"""
```

Comparative judgments are more stable than absolute ones.

**Include the rubric inline:**

Don't rely on the model's notion of "good." Spell out exactly what a 7 vs 8 means for each dimension in your prompt. The more specific your rubric, the more reproducible the scores.

### When to use LLM scoring

Use it when:
- The quality you care about is subjective or multi-dimensional
- No formula captures what "better" means
- You'd evaluate it by reading/looking at it yourself

Don't use it when:
- A deterministic function exists (math, benchmarks, test suites)
- Latency matters — LLM calls add seconds per evaluation
- The scoring budget is tight — each eval costs API tokens

LLM scoring turns any human judgment into a fitness function. The scoring function is where you encode what you value. The agents are just the optimization algorithm.

## Tips

**Fail loudly.** If scoring can't run (import error, missing file, invalid state), raise an exception with a clear message. Don't return a default score — that hides problems and poisons the history.

**Be deterministic when possible.** Set random seeds. Pin model versions. The framework compares scores across runs, so reproducibility matters.

**Keep scoring fast.** Every second of scoring is a second agents aren't iterating. For expensive evaluations (ML training, LLM calls), consider whether a cheaper proxy metric could drive most of the optimization, with the expensive metric as an occasional validation.

**Return extra metrics.** They cost nothing and show up on the leaderboard. Agents use them to understand what's working. A TSP scorer that returns `{"score": distance, "longest_edge": max_edge, "crossings": n_crossings}` gives agents more signal than distance alone.

**Test your scoring function.** Run `autoanything score` after any change. Check that the baseline score makes sense. Try deliberately bad inputs and verify the score gets worse.
