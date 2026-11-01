# Harness approval profile

Evaluator runs use two separate approvals:

1. The operator approves the current lab contract when preparing it.
2. The operator or harness confirms each evaluator operation with the CLI's `--confirmed` flag (or API `confirmed: true`).

The host also supplies a profile through `AUTORESEARCH_HARNESS_PROFILE`. Keep that file outside both the measured project and the lab state directory. A profile is a trusted-host attestation. AutoResearch validates its shape and contract fingerprint, but cannot prove that the named harness actually enforces its sandbox or network policy. AutoResearch does not limit spending; see [Spending](#spending) below.

Treat evaluator code as arbitrary. Keeping the profile outside the project is path hygiene, not filesystem isolation. The external host sandbox must prevent the evaluator from reading or changing the profile and host credentials; AutoResearch also removes the profile-path variable from the evaluator environment, but does not enforce filesystem permissions.

## Prepare and obtain the fingerprint

Run the non-evaluating preflight, prepare the confirmed lab, then read its fingerprint from the SQLite guide. These steps do not run the evaluator or make provider calls:

```text
python SKILL_DIR/scripts/lab.py preflight --spec <spec.json> --program <brief.md>
python SKILL_DIR/scripts/prepare.py --project <project> --program <brief.md> --spec <spec.json> --confirmed
python SKILL_DIR/scripts/lab.py --config <config-path-from-prepare> guide --json
```

Copy `data.contract_fingerprint` from the guide's JSON envelope exactly. It changes when the active measurement contract or epoch changes. Re-read the guide and update the host profile after such a change.

## Profile format

For an offline evaluator, use `network.mode: "disabled"`:

```json
{
  "schema_version": 1,
  "harness": {"id": "my-harness", "version": "1.0"},
  "approval": {
    "approved": true,
    "contract_fingerprint": "<copy from guide JSON>"
  },
  "sandbox": {"enabled": true, "scope": "workspace"},
  "network": {"mode": "disabled"}
}
```

`sandbox.scope` is `workspace`, `container`, `vm`, or `microvm`. For a network-enabled evaluator, set `network.mode` to `allowlisted` and list the permitted hosts:

```json
"network": {"mode": "allowlisted", "allowed_hosts": ["api.example.com"]}
```

`unknown` and `unrestricted` network modes are rejected. The profile needs only `schema_version`, `harness`, `approval`, `sandbox`, and `network`. Any other top-level field is rejected, except the retired `spend` block described below.

## Spending

AutoResearch does not limit spending. It records request and cost usage and reports it, but it does not stop a request or a charge. Set a spending limit on each API key the evaluator uses, in the provider's account settings.

Profiles written for earlier versions may still contain a `spend` block. It is accepted, not validated, and left out of the profile's attested material. Passport `max_seconds` and `max_trials` remain the only limits that AutoResearch enforces.

## Run an approved operation

Set `AUTORESEARCH_HARNESS_PROFILE` to the absolute path of the host-managed profile. The durable `baseline`, `trial`, `resume`, `recover`, `epoch`, `probe`, and `replay` operations all require the profile and a fresh `--confirmed` for that operation. Confirm an evaluator run, for example:

```text
python SKILL_DIR/scripts/lab.py --config <lab-config> --confirmed baseline
python SKILL_DIR/scripts/lab.py --config <lab-config> --confirmed trial start --hypothesis-json '{"title":"...","observation":"...","expected_effect":"..."}'
```

The CLI fails closed when the current operation is not explicitly confirmed, the profile is missing or invalid, the fingerprint is stale, or required isolation/capability attestations are absent. An old profile approves the contract only; it does not confirm a new operation. Read-only commands and preflight do not need a profile. Preparation creates no evaluator result; `prepare --auto` is deprecated and performs no preparation or evaluation.

An agent is compatible only when its host harness can provide the documented profile and enforce the stated controls. This interface does not imply that every agent platform has a sandbox or approval mechanism.
