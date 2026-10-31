# AutoResearch Lab

Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy.

AutoResearch is a local lab for testing one measurable code or prompt hypothesis at a time. An AI agent scouts the project, prepares an isolated candidate, runs the evaluator, and records each decision in a lab outside the source project. The workflow helps preserve evidence; it does not guarantee a quality improvement or replace the evaluator's goal criteria.

The agent interface is a Markdown skill plus ordinary command-line tools. Python 3.10+ and Git are required. No agent-specific API is required; an agent may use the skill only when its host harness can satisfy the documented capability profile. The skill lives in [`.agents/skills/autoresearch/SKILL.md`](.agents/skills/autoresearch/SKILL.md). If a harness does not discover the project-local skill, open or point it to that file directly.

The current source version is 0.7.0. See the [0.7.0 migration notes](docs/v07-migration.md) before upgrading from 0.6.x, and the [0.6.0 migration notes](docs/v06-migration.md) before upgrading from 0.5.x.

## Start a lab

Give the agent a goal, such as “reduce the evaluator's false negatives.” The agent reads the project before asking for missing facts and shows a short passport with the project, goal, metric, evaluation method, constraints, and limits. Confirm that passport before the first measurement. An existing `program.md` or user-provided brief can supply the task; the agent never creates one in the measured project.

The basic command path is:

```text
python .agents/skills/autoresearch/scripts/autoresearch.py doctor
python .agents/skills/autoresearch/scripts/lab.py preflight --spec <spec.json>
python .agents/skills/autoresearch/scripts/prepare.py --project <project> --brief-text - --spec <spec.json> --confirmed
python .agents/skills/autoresearch/scripts/lab.py --config <lab.json> guide --json
python .agents/skills/autoresearch/scripts/lab.py --config <lab.json> --confirmed baseline
python .agents/skills/autoresearch/scripts/lab.py --config <lab.json> guide --json
python .agents/skills/autoresearch/scripts/lab.py --config <lab.json> step
python .agents/skills/autoresearch/scripts/lab.py --config <lab.json> --confirmed trial start --hypothesis <hypothesis.json>
python .agents/skills/autoresearch/scripts/lab.py --config <lab.json> status
```

After the first guide call, copy its `contract_fingerprint` into a host-managed profile outside the project and lab state, set `AUTORESEARCH_HARNESS_PROFILE` to that profile, and only then start an evaluator operation. See [the harness profile contract](.agents/skills/autoresearch/references/harness-profile.md) for the exact schema and an offline example. `preflight` checks the proposed contract and optional saved result without evaluating the project or making network requests. It does not sandbox arbitrary evaluator code. The profile is a host attestation; the host harness must actually enforce approval, sandbox isolation, network policy, and any external provider cap. Passport `max_requests` and `max_cost_usd` are recorded and measured, but do not stop arbitrary subprocess network requests. The skill never checks an API-key balance.

The guide is the canonical compact view of the baseline, best run, explicit goal criteria, parent-relative changes, and next action. The agent should inspect only the history or run artifacts that the guide identifies as relevant. It uses past errors and metric deltas to propose a small number of testable ideas, records one hypothesis, then runs one change per trial.

## Read the result correctly

Framework acceptance means a trial passed the lab's configured comparison with its parent and guards. Goal success means the declared goal thresholds and confirmation repeats passed. Assess any separate external gate on its own before reporting broader goal success. If goal criteria are missing or incomplete, report an exploratory result; do not infer a gate from report prose or call `best_id` a quality success.

Keep evidence origins separate. `historical` is an earlier result; `fresh` and `replay` run the evaluator and may make model calls; `offline_rescore` uses stored responses; `case_resume` continues pending IDs when supported; `prepare --resume` reuses a matching lab contract. For SQLite labs, inspect saved results, checkpoints, and attempts through the guide and API-backed views. For legacy labs, use read-only guide/status and preserve the folder with ZIP export; do not continue or import its runs into a new lab. Use `resume <run_id>` only when the CLI confirms support and fingerprints still match.

State is stored under `AUTORESEARCH_HOME` (default `~/.autoresearch`), not in the measured project. Previous lab roots remain discoverable. Legacy labs are read-only: inspect their saved status and history, then export a byte-indexed ZIP before starting a new lab. Historical runs and fingerprints stay in that archive and cannot be imported into a working lab or used as its baseline. See [CLI contracts](.agents/skills/autoresearch/references/cli.md), [measurement semantics](.agents/skills/autoresearch/references/measurement.md), and [project safety](.agents/skills/autoresearch/references/safety.md).

The local dashboard binds to `127.0.0.1` and starts as a background service on the first regular lab CLI command. Startup does not open or focus a browser; on Windows the service runs without a console window. Open a browser only with an explicit `--open-browser` request, such as `python .agents/skills/autoresearch/scripts/lab.py dashboard ensure --open-browser`. Use global `--headless` or `AUTORESEARCH_NO_DASHBOARD=1` to skip service startup. Before the first confirmed lab, the service shows an empty preparation state and does not create the SQLite database.

SQLite-backed labs use their existing authenticated lifecycle routes. Legacy archive, restore, and delete actions are read-only; preserve an old lab with the explicit ZIP export. Promote only when the user asked for it. By default, experiments stay in the lab's candidate snapshot.

## Install and update

Run [the package installer](.agents/skills/autoresearch/references/installation.md) from a source package obtained through a trusted channel. The installer records file hashes in the target and refuses to overwrite modified managed files; the hashes are not signed. Updates stop before writing if an existing target has modified managed files, unknown content, or an unverifiable schema 1 manifest. Back up that target and use a clean destination; installation never removes or adopts unknown files automatically.

## Verify the project

Run the offline suite with:

```text
python -m unittest discover -s tests -v
```

## License

AutoResearch is released under the MIT License; see [LICENSE](LICENSE). The license names nightwishrus as the copyright holder. The upstream [karpathy/autoresearch](https://github.com/karpathy/autoresearch) README declares the MIT license; that repository has no separate LICENSE file. The bundled [`openresearch`](.agents/skills/openresearch) skill comes from alphaXiv and keeps its own MIT notice in [its LICENSE](.agents/skills/openresearch/LICENSE).
