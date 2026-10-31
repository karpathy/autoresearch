try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import
import unittest
from types import SimpleNamespace

from autoresearch_runtime.dashboard import build_problems, build_verdict, chart_series, presentation_status, reason_label
from autoresearch_runtime.dashboard import render_unified_dashboard


class DashboardTest(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            primary="score", direction="max", guards=(("validity", "max"),),
            budget={"max_seconds": 100, "max_trials": 5, "max_requests": 10, "max_cost_usd": 2},
        )
        self.manifests = [
            {
                "id": "baseline", "title": "Baseline", "status": "accepted", "runs": [
                    {"metrics": {"score": 0.40, "validity": 1.0}, "usage": {"requests": 1, "cost_usd": 0.1}, "duration_seconds": 2},
                    {"metrics": {"score": 0.42, "validity": 1.0}, "usage": {"requests": 1, "cost_usd": 0.1}, "duration_seconds": 2},
                ],
            },
            {
                "id": "trial-0001", "title": "Too broad", "status": "rejected", "reason": "primary did not improve",
                "runs": [{"metrics": {"score": 0.39, "validity": 1.0}, "usage": {"requests": 1, "cost_usd": 0.1}, "duration_seconds": 2}],
            },
            {
                "id": "trial-0002", "title": "<script>alert(1)</script>", "status": "accepted", "reason": "improved",
                "runs": [
                    {"metrics": {"score": 0.45, "validity": 1.0}, "usage": {"requests": 1, "cost_usd": 0.1}, "duration_seconds": 2, "details": {"slice": "<img src=x>"}},
                    {"metrics": {"score": 0.46, "validity": 1.0}, "usage": {"requests": 1, "cost_usd": 0.1}, "duration_seconds": 2},
                ],
            },
        ]

    def test_chart_series_tracks_worst_repeats_and_accepted_best(self):
        rows = chart_series(self.manifests, "score", "max")
        self.assertEqual([(row["worst"], row["best_so_far"]) for row in rows], [
            (0.40, 0.40), (0.39, 0.40), (0.45, 0.45),
        ])
        self.assertEqual((rows[2]["low"], rows[2]["high"]), (0.45, 0.46))

    def test_chart_series_for_min_metric_improves_downward(self):
        manifests = [
            {"id": "baseline", "status": "accepted", "runs": [{"metrics": {"latency": 10}}, {"metrics": {"latency": 9}}]},
            {"id": "trial-0001", "status": "accepted", "runs": [{"metrics": {"latency": 8}}, {"metrics": {"latency": 9.5}}]},
        ]
        rows = chart_series(manifests, "latency", "min")
        self.assertEqual([(row["worst"], row["best_so_far"]) for row in rows], [(10, 10), (9.5, 9.5)])

    def test_chart_series_keeps_every_hypothesis_including_missing_metrics(self):
        manifests = [
            {"id": "baseline", "status": "accepted", "title": "Baseline", "runs": [{"metrics": {"score": 0.40}}]},
            {"id": "trial-0001", "status": "budget", "title": "No key", "reason": "OPENROUTER_API_KEY is missing", "runs": []},
            {"id": "trial-0002", "status": "rejected", "title": "Worse", "runs": [
                {"metrics": {"score": 0.20}}, {"metrics": {"score": 0.50}},
            ]},
            {"id": "trial-0003", "status": "accepted", "title": "Better", "runs": [{"metrics": {"score": 0.55}}]},
        ]
        rows = chart_series(manifests, "score", "max")
        self.assertEqual([row["id"] for row in rows], ["baseline", "trial-0001", "trial-0002", "trial-0003"])
        self.assertFalse(rows[1]["has_value"])
        self.assertIsNone(rows[1]["worst"])
        self.assertEqual(rows[1]["reason_label"], "Не найден ключ OPENROUTER — опыт не запущен")
        self.assertEqual(presentation_status("budget", "OPENROUTER_API_KEY is missing")[0], "config")
        self.assertEqual([row["worst"] for row in rows], [0.40, None, 0.20, 0.55])
        self.assertEqual([row["best_so_far"] for row in rows], [0.40, 0.40, 0.40, 0.55])
        self.assertAlmostEqual(rows[2]["delta_vs_baseline"], -0.20)
        self.assertTrue(rows[3]["improved_best"])
        self.assertEqual(reason_label("primary did not improve"), "Главная метрика не улучшилась")
        verdict = build_verdict(metric_name="f1_score", direction="max", baseline=0.3467, best=0.5974,
                                hypotheses=[{"id": "trial-0001", "status": "accepted"}], samples=[0.35, 0.6])
        self.assertIn("п.п.", verdict["text"])
        self.assertNotIn("п,п,", verdict["text"])
        stale = build_problems({"latest_error": "OPENROUTER_API_KEY is missing", "latest_trial_status": "accepted"})
        self.assertEqual(stale, [])
        fresh = build_problems({"latest_error": "OPENROUTER_API_KEY is missing", "latest_trial_status": "budget"})
        self.assertEqual(fresh[0]["title"], "Последний опыт остановился с ошибкой")
        note = build_problems({"comparison_status": "incomparable", "comparison_note": "старый текст каталога",
                               "status": "completed"})
        self.assertEqual(note[0]["level"], "info")
        self.assertEqual(note[0]["title"], "Справка о контракте")
        self.assertIn("собственным результатом", note[0]["text"])
        self.assertEqual(note[0]["detail"], "старый текст каталога")

    def test_chart_series_marks_baseline_parent_decision_and_incomplete_repeats(self):
        manifests = [
            {"id": "baseline-0001", "kind": "baseline", "status": "accepted", "runs": [
                {"metrics": {"score": 0.4, "validity": 1.0}},
                {"metrics": {"score": 0.42, "validity": 1.0}},
            ]},
            {"id": "trial-0001", "status": "rejected", "parent": "baseline-0001",
             "finalization": {"decision": "rejected"}, "runs": [
                 {"metrics": {"score": 0.39, "validity": 1.0}},
             ]},
        ]
        rows = chart_series(manifests, "score", "max", baseline_id="baseline-0001",
                            guards=[("validity", "max")], expected_repeats=2)
        self.assertTrue(rows[0]["is_baseline"])
        self.assertEqual(rows[1]["parent_id"], "baseline-0001")
        self.assertEqual(rows[1]["decision"], "rejected")
        self.assertEqual(rows[1]["completed_repeats"], 1)
        self.assertEqual(rows[1]["expected_repeats"], 2)
        self.assertTrue(rows[1]["incomplete"])
        self.assertEqual(rows[1]["guards"]["validity"]["worst"], 1.0)

    def test_snapshot_escapes_untrusted_text(self):
        summary = {"projects": [{"id": "p", "name": "<script>alert(1)</script>", "labs": [
            {"id": "lab", "name": "Lab", "status": "ready"},
        ]}], "archived": []}
        page = render_unified_dashboard(summary, {"p::lab": {"lab": {"name": "<img src=x>"}}},
                                        live=False, root_id="root")
        self.assertNotIn("<script>alert(1)</script>", page)
        self.assertNotIn("<img src=x>", page)
        self.assertIn("\\u003cscript\\u003ealert(1)\\u003c/script\\u003e", page)
        self.assertIn("\\u003cimg src=x\\u003e", page)

    def test_live_and_offline_report_use_the_same_project_and_hypothesis_template(self):
        summary = {"projects": [{"id": "project-a", "name": "Project A", "labs": [{
            "id": "lab-a", "name": "Improve quality", "status": "ready", "stage": "Готово",
            "metric": "score", "best_score": 0.5, "experiments": 1, "cost_usd": 0.1,
        }]}], "archived": []}
        live = render_unified_dashboard(summary, {}, live=True, root_id="test-root")
        offline = render_unified_dashboard(summary, {"project-a::lab-a": {"lab": {"id": "lab-a"}}},
                                           live=False, root_id="test-root")
        for page in (live, offline):
            self.assertIn("Проекты и лаборатории", page)
            self.assertIn("Project A", page)
        self.assertIn("/assets/live_app.js", live)
        self.assertNotIn("/assets/app.js", live)
        self.assertNotIn("Гипотезы и решения", live)
        from autoresearch_runtime.dashboard import read_asset
        live_script = read_asset("live_app.js")
        self.assertIsNotNone(live_script)
        for action in ("/commands", "startOperation", "pollJob", "/api/v1/jobs"):
            self.assertNotIn(action, live_script.decode("utf-8"))
        self.assertIn("Офлайн-снимок", offline)
        self.assertIn("Гипотезы и решения", offline)
        self.assertIn("scheduleRefresh", offline)
        self.assertIn('fetch("/api/status"', offline)
        self.assertIn("POLL_ACTIVE_MS = 2000", offline)
        self.assertNotIn("/assets/app.js", offline)
        self.assertNotIn("/assets/live_app.js", offline)
        self.assertIn('const boot = JSON.parse(document.getElementById("snapshot").textContent);', offline)


if __name__ == "__main__":
    unittest.main()
