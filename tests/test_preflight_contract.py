"""Offline contract checks must catch adapter mistakes before evaluation."""
try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import contextlib
import io
import json
import shutil
import unittest
from pathlib import Path

from tests import test_skill_workflow as workflow
from tests.support import tree_digest
from autoresearch_runtime.cli import main
from autoresearch_runtime.config import load_config
from autoresearch_runtime.errors import LabError
from autoresearch_runtime.probe import adopt_probe


class PreflightContractTest(unittest.TestCase):
    setUp = workflow.SkillWorkflowTest.setUp
    _restore_home = workflow.SkillWorkflowTest._restore_home
    spec = workflow.SkillWorkflowTest.spec

    def invoke(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(args))
        raw = out.getvalue() or err.getvalue()
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = {"error": {"message": raw}}
        return code, payload

    def test_preflight_does_not_create_lab_or_execute_evaluator(self):
        spec = self.spec()
        self.evaluator.write_text("raise RuntimeError('must never execute')\n", encoding="utf-8")
        # Refresh the known-case evaluator identity; preflight validates, not measures.
        workflow.SkillWorkflowTest.update_known_case(self, spec)
        before = tree_digest(self.root)
        code, payload = self.invoke("preflight", "--spec", str(spec), "--program", str(self.program))
        self.assertEqual(code, 0, payload)
        self.assertTrue(payload["data"]["contract_valid"])
        self.assertEqual(payload["data"]["integration_status"], "unverified")
        self.assertEqual(tree_digest(self.root), before)

    def test_preflight_uses_live_result_validator(self):
        spec = self.spec()
        result = self.root / "result.json"
        result.write_text(json.dumps({"metrics": {"score": 1}, "usage": {"requests": 41, "cost_usd": None}}))
        code, payload = self.invoke("preflight", "--spec", str(spec), "--program", str(self.program), "--result", str(result))
        self.assertEqual(code, 2)
        self.assertIn("Invalid usage", payload["error"]["message"])
        result.write_text(json.dumps({"metrics": {"score": 1}, "usage": {"requests": 41, "cost_usd": 0, "unknown_requests": 41}}))
        code, payload = self.invoke("preflight", "--spec", str(spec), "--program", str(self.program), "--result", str(result))
        self.assertEqual(code, 0, payload)
        self.assertTrue(payload["data"]["result_valid"])
        self.assertEqual(payload["data"]["integration_status"], "unverified")

    def test_preflight_never_repairs_an_existing_pending_lab(self):
        from autoresearch_runtime.bootstrap import prepare_lab
        from autoresearch_runtime.sqlite_store import SQLiteStore

        prepared = prepare_lab(self.program, self.spec(), confirmed=True)
        config_path = Path(prepared["config"])
        config = load_config(config_path, allow_pending=True)
        store = SQLiteStore()
        lab_id = store.resolve_lab_id(config_path)
        state = store.load_projection(lab_id, "state")
        state.update(setup_pending=True, config_copy_sha256="stale", config_sha256="stale")
        store.save_projection(lab_id, "state", state)
        state_before = store.load_projection(lab_id, "state")
        revision_before = store.get_revision()
        config_bytes_before = store.load_config_bytes(lab_id)
        before = tree_digest(self.root)
        code, payload = self.invoke("preflight", "--spec", str(self.spec()), "--program", str(self.program))
        self.assertEqual(code, 0, payload)
        self.assertEqual(tree_digest(self.root), before)
        self.assertEqual(store.load_projection(lab_id, "state"), state_before)
        self.assertEqual(store.load_config_bytes(lab_id), config_bytes_before)
        self.assertEqual(store.get_revision(), revision_before)

    def test_preflight_without_passport_cannot_authorize_budget(self):
        spec = self.spec()
        raw = json.loads(spec.read_text())
        raw.update(budget={"max_trials": 3}, budget_sources={"max_trials": "max_trials: 3"})
        spec.write_text(json.dumps(raw))
        code, payload = self.invoke("preflight", "--spec", str(spec))
        self.assertEqual(code, 2)
        self.assertIn("exact quote", payload["error"]["message"])

    def test_preflight_and_prepare_reject_spend_limits_with_a_coded_error(self):
        from autoresearch_runtime.bootstrap import main as prepare_main

        spec = self.spec()
        raw = json.loads(spec.read_text())
        raw["budget"] = {"max_requests": 5, "max_cost_usd": 2.0}
        spec.write_text(json.dumps(raw))
        code, payload = self.invoke("preflight", "--spec", str(spec), "--program", str(self.program))
        self.assertEqual(code, 2, payload)
        self.assertEqual(payload["error"]["code"], "E_CONTRACT_SPEND_LIMIT")
        self.assertIn("does not limit spending", payload["error"]["message"])
        self.assertIn("limit on the API key", payload["error"]["message"])

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = prepare_main(["--confirmed", "--program", str(self.program), "--spec", str(spec)])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(err.getvalue())["error"]["code"], "E_CONTRACT_SPEND_LIMIT")

    def test_goal_criteria_is_optional_validated_configuration(self):
        example = Path(__file__).parents[1] / "tests" / "fixtures" / "draft-config"
        toy = self.root / "toy"
        shutil.copytree(example, toy)
        path = toy / "lab.json"
        self.assertIsNone(getattr(load_config(path), "goal_criteria", "missing"))
        raw = json.loads(path.read_text())
        raw["goal_criteria"] = {"thresholds": {"quality": {"min": 0.9}}, "confirmation_repeats": 3}
        path.write_text(json.dumps(raw))
        self.assertEqual(load_config(path).goal_criteria["confirmation_repeats"], 3)
        raw["goal_criteria"]["thresholds"]["quality"]["min"] = True
        path.write_text(json.dumps(raw))
        with self.assertRaises(LabError):
            load_config(path)

    def test_claimed_offline_verification_requires_pinned_fixture(self):
        toy = self.root / "toy"
        shutil.copytree(Path(__file__).parents[1] / "tests" / "fixtures" / "draft-config", toy)
        path = toy / "lab.json"
        raw = json.loads(path.read_text())
        raw["evaluator_capabilities"] = {"case_resume": True, "offline_verified": True}
        path.write_text(json.dumps(raw))
        with self.assertRaises(LabError):
            load_config(path)

    def test_preflight_preserves_exact_provider_attestations(self):
        spec = self.spec()
        raw = json.loads(spec.read_text())
        raw["expected_attestations"] = {"endpoint": "https://sandbox.invalid/v1/chat/completions", "model": "sandbox/model"}
        spec.write_text(json.dumps(raw))
        result = self.root / "provider-result.json"
        result.write_text(json.dumps({"metrics": {"score": 1}, "attestations": {"endpoint": "https://other.invalid", "model": "other/model"}}))
        code, payload = self.invoke("preflight", "--spec", str(spec), "--program", str(self.program), "--result", str(result))
        self.assertEqual(code, 2, payload)
        self.assertIn("attestation mismatch", payload["error"]["message"])


if __name__ == "__main__":
    unittest.main()
