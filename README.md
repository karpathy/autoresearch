# AutoLaboratory

Let an AI agent try improvements to your code or prompts, and keep only the ones a measurement backs up. Every attempt runs on an isolated copy, is compared with a baseline by your own evaluator, and is recorded outside your project.

Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy. The package, its commands and its files are still named `autoresearch`.

## See it work

You need Python 3.10+ and Git. Nothing else is installed, no API key is needed, and the demo cleans up after itself.

```text
git clone https://github.com/MilleniumGenAI/AutoLaboratory.git
cd AutoLaboratory
python examples/toy/run_demo.py
```

The demo prepares a lab for a one-file project, measures a baseline, makes a one-line change, and lets the lab decide. Real output, trimmed to the result:

```text
toy demo: offline evaluator; temp project and lab, removed on exit
baseline  mean_error=2.000  status=accepted
trial     mean_error=0.000  id=trial-0001
decision  accepted  (mean_error 2.000 -> 0.000, min)
```

The demo script makes the edit itself. In real use, your agent reads the lab's history and proposes the change. The toy project shows how the flow works; it says nothing about how much a real project will improve.

## Why not just let the agent loop?

An agent working directly in your repository can overwrite your work and judge its own results. The lab separates those roles:

| Without a lab | With AutoLaboratory |
| --- | --- |
| The agent edits your working tree | The agent edits only an isolated candidate copy; you decide whether to promote it |
| The agent says the change helped | Your evaluator measures it against a declared parent, with repeat rules and guard metrics |
| Several changes land together | One hypothesis per trial, with its reason recorded |
| Failed ideas are forgotten | Every run, decision and metric stays in a lab under `~/.autoresearch` |
| An interrupted run starts over | Runs can resume when the evaluator supports it and fingerprints still match |

Karpathy's original lets an agent iterate on a model training setup. This project applies the same idea to any project with a measurable evaluator and adds the isolation and record-keeping above.

## Use it on your project

Install the skill for your agent:

```text
python .agents/skills/autoresearch/install.py --target claude
```

Other targets are `agents`, `cursor`, `codex` and `all`.

Then give the agent a goal, such as "reduce the evaluator's false negatives." It reads the project, asks for what is missing, and shows a short passport with the goal, metric, evaluator, constraints and limits. Confirm the passport before the first measurement.

Before any evaluator runs, the host writes a [harness profile](.agents/skills/autoresearch/references/harness-profile.md) outside the project. It records the approval, sandbox and network policy that the host enforces. An agent cannot start a measurement without it.

The commands your agent will run:

```text
python .agents/skills/autoresearch/scripts/autoresearch.py doctor
python .agents/skills/autoresearch/scripts/lab.py preflight --spec <spec.json>
python .agents/skills/autoresearch/scripts/prepare.py --project <project> --brief-text - --spec <spec.json> --confirmed
python .agents/skills/autoresearch/scripts/lab.py --config <lab.json> guide --json
python .agents/skills/autoresearch/scripts/lab.py --config <lab.json> --confirmed baseline
python .agents/skills/autoresearch/scripts/lab.py --config <lab.json> --confirmed trial start --hypothesis <hypothesis.json>
python .agents/skills/autoresearch/scripts/lab.py --config <lab.json> status
```

## Before you rely on it

- **Spending is not limited.** AutoLaboratory records requests and cost but never stops a paid call. Set a spending limit on each API key your evaluator uses.
- **It does not sandbox your evaluator.** The host harness must enforce isolation and network policy; the profile only states that it does.
- **"Accepted" is not "goal achieved."** Accepted means a trial beat its parent under the lab's rules. Goal success needs declared thresholds and confirmation repeats. See [measurement semantics](.agents/skills/autoresearch/references/measurement.md).
- **A local dashboard starts with regular commands.** It binds to `127.0.0.1` and opens no browser. Use `--headless` or `AUTORESEARCH_NO_DASHBOARD=1` to skip it.
- **The codebase is not small.** The runtime is about 22k lines of Python, using only the standard library.

## Compatibility

Works through a Markdown skill and ordinary command-line tools, with no agent-specific API. Confirmed end to end in Claude Code on Windows. Codex, Cursor and Gemini CLI are not verified yet; see [agent compatibility](docs/compatibility.md). CI runs the test suite on Ubuntu and Windows with Python 3.10 and 3.12. macOS is not tested.

## Documentation

- [Skill instructions](.agents/skills/autoresearch/SKILL.md) and [CLI contracts](.agents/skills/autoresearch/references/cli.md)
- [Evaluator contract](.agents/skills/autoresearch/references/evaluator-contract.md) and [project safety](.agents/skills/autoresearch/references/safety.md)
- [Installation and updates](.agents/skills/autoresearch/references/installation.md)
- Upgrade notes: [0.8.0](docs/v08-migration.md), [0.7.0](docs/v07-migration.md), [0.6.0](docs/v06-migration.md)

The current version is 0.8.0. To run the offline test suite:

```text
python -m unittest discover -s tests
```

## License

MIT; see [LICENSE](LICENSE). The license names nightwishrus as the copyright holder. The upstream [karpathy/autoresearch](https://github.com/karpathy/autoresearch) README declares the MIT license; that repository has no separate LICENSE file. The bundled [`openresearch`](.agents/skills/openresearch) skill comes from alphaXiv and keeps its own MIT notice in [its LICENSE](.agents/skills/openresearch/LICENSE).
