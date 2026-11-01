from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

from autoresearch_runtime.harness_policy import authorize_operation, recheck_operation
from autoresearch_runtime.errors import LabError
from autoresearch_runtime.runtimeops import _bind_operation_context, reserve_operation
from autoresearch_runtime.worker import _validate_worker_policy
from autoresearch_runtime.api_v1 import _validate_command
from autoresearch_runtime.fuzzy_args import normalize_cli_args
from autoresearch_runtime.commands.common import (
    explicit_confirmation, resolve_config,
)
from autoresearch_runtime.commands import baseline as baseline_command
from autoresearch_runtime.baseline import run_baseline
from autoresearch_runtime.trial import run_trial


_ABSENT = object()


class HarnessPolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.project = root / "project"
        self.project.mkdir()
        self.profile_path = root / "host" / "harness-profile.json"
        self.profile_path.parent.mkdir()
        self.config = SimpleNamespace(
            lab_id="b5ef9c01-3a58-4ca8-a63c-4eb79216a912",
            source_root=self.project,
            state_dir=root / "state",
            evaluator_capabilities={"provider": "local-evaluator"},
        )
        self.fingerprint = "a" * 64

    def profile(self, *, network="disabled", fingerprint=None, legacy_spend=_ABSENT):
        value = {
            "schema_version": 1,
            "harness": {"id": "trusted-test-harness", "version": "1.2.0"},
            "approval": {
                "approved": True,
                "contract_fingerprint": fingerprint or self.fingerprint,
            },
            "sandbox": {"enabled": True, "scope": "workspace"},
            "network": ({"mode": network, "allowed_hosts": ["api.example.test"]}
                        if network == "allowlisted" else {"mode": network}),
        }
        if legacy_spend is not _ABSENT:
            value["spend"] = legacy_spend  # a profile written before 0.8.0
        return value

    def write_profile(self, value):
        self.profile_path.write_text(json.dumps(value), encoding="utf-8")

    def authorize(self, confirmed=True):
        with patch.dict(os.environ, {"AUTORESEARCH_HARNESS_PROFILE": str(self.profile_path)}):
            with patch("autoresearch_runtime.harness_policy.contract_fingerprint",
                       return_value=self.fingerprint):
                return authorize_operation(self.config, "baseline", confirmed=confirmed)

    def test_offline_profile_requires_explicit_confirmation_and_allows_confirmed_run(self):
        self.write_profile(self.profile())
        with self.assertRaises(LabError) as missing_confirmation:
            self.authorize(None)
        self.assertEqual(missing_confirmation.exception.code, "E_HARNESS_CONFIRMATION_REQUIRED")

        policy = self.authorize(True)
        self.assertEqual(policy["decision"], "allow")
        self.assertEqual(policy["profile"]["network"]["mode"], "disabled")
        self.assertEqual(policy["contract_fingerprint"], self.fingerprint)

    def test_missing_profile_fails_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("autoresearch_runtime.harness_policy.contract_fingerprint",
                       return_value=self.fingerprint):
                with self.assertRaises(LabError) as error:
                    authorize_operation(self.config, "trial", confirmed=True)
        self.assertEqual(error.exception.code, "E_HARNESS_PROFILE_REQUIRED")

    def test_unknown_and_unrestricted_network_profiles_are_rejected(self):
        for mode in ("unknown", "unrestricted"):
            with self.subTest(mode=mode):
                self.write_profile(self.profile(network=mode))
                with self.assertRaises(LabError) as error:
                    self.authorize(True)
                self.assertEqual(error.exception.code, "E_HARNESS_POLICY_REJECTED")

    def test_allowlisted_network_needs_no_spend_cap(self):
        self.write_profile(self.profile(network="allowlisted"))
        policy = self.authorize(True)
        self.assertEqual(policy["decision"], "allow")
        self.assertEqual(policy["profile"]["network"],
                         {"mode": "allowlisted", "allowed_hosts": ["api.example.test"]})
        self.assertNotIn("spend", policy["profile"])

    def test_legacy_spend_block_is_accepted_and_ignored(self):
        self.write_profile(self.profile(network="allowlisted"))
        plain = self.authorize(True)
        for legacy in (
            {"provider": "local-evaluator", "scope": "per_run", "max_cost_usd": 0.25},
            {"scope": "weekly"},
            None,
        ):
            with self.subTest(legacy_spend=legacy):
                self.write_profile(self.profile(network="allowlisted", legacy_spend=legacy))
                policy = self.authorize(True)
                self.assertEqual(policy["decision"], "allow")
                self.assertNotIn("spend", policy["profile"])
                self.assertEqual(policy["profile_sha256"], plain["profile_sha256"])
                self.assertEqual(policy["contract_fingerprint"], plain["contract_fingerprint"])

    def test_contract_approval_must_match_current_fingerprint(self):
        self.write_profile(self.profile(fingerprint="b" * 64))
        with self.assertRaises(LabError) as error:
            self.authorize(True)
        self.assertEqual(error.exception.code, "E_HARNESS_POLICY_REJECTED")

    def test_profile_inside_measured_project_is_not_trusted(self):
        self.profile_path = self.project / "harness-profile.json"
        self.write_profile(self.profile())
        with self.assertRaises(LabError) as error:
            self.authorize(True)
        self.assertEqual(error.exception.code, "E_HARNESS_PROFILE_INVALID")

    @unittest.skipUnless(os.name == "nt", "Windows directory junction behavior")
    def test_profile_through_project_junction_to_host_file_is_not_trusted(self):
        host_dir = self.project.parent / "host-profile"
        host_dir.mkdir()
        host_profile = host_dir / "harness-profile.json"
        host_profile_bytes = json.dumps(self.profile()).encode("utf-8")
        host_profile.write_bytes(host_profile_bytes)

        alias = self.project / "alias"
        created = subprocess.run(
            ["cmd.exe", "/c", "mklink", "/J", str(alias), str(host_dir)],
            capture_output=True, text=True, check=False,
        )
        if created.returncode != 0 or not alias.exists():
            self.skipTest(
                "directory junction creation unavailable: "
                f"{created.stdout.strip()} {created.stderr.strip()}"
            )
        # rmdir removes the junction itself and never walks into its target.
        self.addCleanup(os.rmdir, alias)

        self.profile_path = alias / host_profile.name
        with self.assertRaises(LabError) as error:
            self.authorize(True)

        self.assertEqual(error.exception.code, "E_HARNESS_PROFILE_INVALID")
        self.assertEqual(host_profile.read_bytes(), host_profile_bytes)

    def test_profile_symlink_inside_project_to_host_file_is_not_trusted(self):
        self.write_profile(self.profile())
        link = self.project / "harness-profile.json"
        try:
            link.symlink_to(self.profile_path)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlinks are unavailable: {error}")
        self.profile_path = link

        with self.assertRaises(LabError) as error:
            self.authorize(True)
        self.assertEqual(error.exception.code, "E_HARNESS_PROFILE_INVALID")

    def test_profile_symlink_outside_project_to_project_file_is_not_trusted(self):
        inside = self.project / "profile-target.json"
        inside.write_text(json.dumps(self.profile()), encoding="utf-8")
        link = self.profile_path.parent / "profile-link.json"
        try:
            link.symlink_to(inside)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlinks are unavailable: {error}")
        self.profile_path = link

        with self.assertRaises(LabError) as error:
            self.authorize(True)
        self.assertEqual(error.exception.code, "E_HARNESS_PROFILE_INVALID")

    def test_profile_symlink_is_rejected_before_resolving_its_target(self):
        self.write_profile(self.profile())
        target = self.profile_path
        link = self.profile_path.parent / "profile-link.json"
        link.write_text("symlink placeholder", encoding="utf-8")
        self.profile_path = link
        original_is_symlink = Path.is_symlink
        original_resolve = Path.resolve

        def is_symlink(path):
            return path == link or original_is_symlink(path)

        def resolve(path, *args, **kwargs):
            if path == link:
                return target
            return original_resolve(path, *args, **kwargs)

        with patch.object(Path, "is_symlink", is_symlink), patch.object(Path, "resolve", resolve):
            with self.assertRaises(LabError) as error:
                self.authorize(True)
        self.assertEqual(error.exception.code, "E_HARNESS_PROFILE_INVALID")

    def test_profile_schema_rejects_boolean_version_unknown_fields_and_non_string_network_mode(self):
        invalid_profiles = []
        boolean_version = self.profile()
        boolean_version["schema_version"] = True
        invalid_profiles.append(boolean_version)
        unknown_field = self.profile()
        unknown_field["debug_bypass"] = True
        invalid_profiles.append(unknown_field)
        non_string_mode = self.profile()
        non_string_mode["network"] = {"mode": ["disabled"]}
        invalid_profiles.append(non_string_mode)
        for invalid in invalid_profiles:
            with self.subTest(profile=invalid):
                self.write_profile(invalid)
                with self.assertRaises(LabError) as error:
                    self.authorize(True)
                self.assertEqual(error.exception.code, "E_HARNESS_PROFILE_INVALID")

        self.profile_path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
        with self.assertRaises(LabError) as duplicate:
            self.authorize(True)
        self.assertEqual(duplicate.exception.code, "E_HARNESS_PROFILE_INVALID")

    def test_worker_recheck_rejects_a_changed_profile(self):
        self.write_profile(self.profile())
        original = self.authorize(True)
        self.write_profile(self.profile(network="unknown"))
        with patch.dict(os.environ, {"AUTORESEARCH_HARNESS_PROFILE": str(self.profile_path)}):
            with patch("autoresearch_runtime.harness_policy.contract_fingerprint",
                       return_value=self.fingerprint):
                with self.assertRaises(LabError) as error:
                    recheck_operation(self.config, "baseline", original)
        self.assertEqual(error.exception.code, "E_HARNESS_POLICY_REJECTED")

    def test_worker_recheck_rejects_old_operations_without_a_policy_snapshot(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LabError) as error:
                recheck_operation(self.config, "trial", None)
        self.assertEqual(error.exception.code, "E_HARNESS_POLICY_STALE")

    def test_durable_enqueue_binds_policy_snapshot_and_removes_confirmation_from_executor_params(self):
        policy = {"decision": "allow", "confirmed": True, "profile_sha256": "c" * 64}
        payload = {"command": "baseline", "params": {"confirmed": True, "supersede": False}}
        with patch("autoresearch_runtime.runtimeops.safe_operation_context", return_value={"schema": 1}), \
             patch("autoresearch_runtime.runtimeops.authorize_operation", return_value=policy,
                   create=True) as authorize:
            bound = _bind_operation_context(self.config, "baseline", payload)

        authorize.assert_called_once_with(self.config, "baseline", confirmed=True)
        self.assertEqual(bound["params"]["harness_policy"], policy)
        self.assertNotIn("confirmed", bound["params"])
        self.assertTrue(payload["params"]["confirmed"])

    def test_policy_gate_precedes_enqueue_and_allowed_snapshot_is_persisted(self):
        class CapturingStore:
            def __init__(self):
                self.calls = []

            def enqueue_operation(self, lab_id, kind, idempotency_key, payload):
                self.calls.append((lab_id, kind, idempotency_key, payload))
                return {"status": "queued", "payload": payload}

        store = CapturingStore()
        with patch("autoresearch_runtime.runtimeops._config_for", return_value=self.config), \
             patch("autoresearch_runtime.runtimeops.SQLiteStore", return_value=store), \
             patch("autoresearch_runtime.runtimeops.lab_lock", return_value=nullcontext()), \
             patch("autoresearch_runtime.runtimeops.safe_operation_context", return_value={"schema": 1}):
            with self.assertRaises(LabError) as denied:
                reserve_operation(
                    self.config, "baseline", "missing-confirmation",
                    {"command": "baseline", "params": {"confirmed": False}},
                )
            self.assertEqual(denied.exception.code, "E_HARNESS_CONFIRMATION_REQUIRED")
            self.assertEqual(store.calls, [])

            self.write_profile(self.profile())
            with patch.dict(os.environ, {"AUTORESEARCH_HARNESS_PROFILE": str(self.profile_path)}), \
                 patch("autoresearch_runtime.harness_policy.contract_fingerprint",
                       return_value=self.fingerprint):
                operation = reserve_operation(
                    self.config, "baseline", "approved-run",
                    {"command": "baseline", "params": {"confirmed": True}},
                )

        self.assertEqual(len(store.calls), 1)
        persisted = operation["payload"]["params"]["harness_policy"]
        self.assertEqual(persisted["decision"], "allow")
        self.assertEqual(persisted["profile_sha256"], store.calls[0][3]["params"]["harness_policy"]["profile_sha256"])

    def test_worker_rechecks_policy_and_strips_snapshot_before_executor_dispatch(self):
        snapshot = {"decision": "allow", "confirmed": True}
        params = {"harness_policy": snapshot, "supersede": True}
        with patch("autoresearch_runtime.worker.recheck_operation", return_value=snapshot) as recheck:
            clean = _validate_worker_policy(self.config, "baseline", params)

        recheck.assert_called_once_with(self.config, "baseline", snapshot)
        self.assertEqual(clean, {"supersede": True})
        self.assertEqual(params["harness_policy"], snapshot)

    def test_api_measurement_requires_boolean_confirmation_and_routes_it_to_shared_policy_path(self):
        body = {"command": "baseline", "payload": {}}
        for invalid in (None, False, 1, "true"):
            request = {**body, "confirmed": invalid} if invalid is not None else body
            with self.subTest(confirmed=invalid), self.assertRaises(ValueError):
                _validate_command(request)

        command, params, _ = _validate_command({**body, "confirmed": True})
        self.assertEqual(command, "baseline")
        self.assertIs(params["confirmed"], True)

    def test_cli_confirmation_flag_is_hoisted_from_after_the_command(self):
        self.assertEqual(
            normalize_cli_args(["baseline", "--confirmed", "--lab", "lab-id"]),
            ["--confirmed", "--lab", "lab-id", "baseline"],
        )
        self.assertEqual(explicit_confirmation(SimpleNamespace(confirmed=True)), {"confirmed": True})
        self.assertEqual(explicit_confirmation(SimpleNamespace(confirmed=False)), {})
        self.assertEqual(explicit_confirmation(SimpleNamespace()), {})

    def test_sqlite_measurement_executors_reject_calls_outside_the_durable_worker(self):
        attempts = (
            lambda: run_baseline(self.config),
            lambda: run_trial(self.config, {"title": "t", "observation": "o", "expected_effect": "e"}),
        )
        for attempt in attempts:
            with self.subTest(attempt=attempt), self.assertRaises(LabError) as error:
                attempt()
            self.assertEqual(error.exception.code, "E_DURABLE_OPERATION_REQUIRED")


if __name__ == "__main__":
    unittest.main()
