try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import
import unittest

from autoresearch_runtime.dashboard import read_asset, render_unified_dashboard


class DashboardTest(unittest.TestCase):
    def test_page_payload_escapes_markup_and_carries_only_the_root_id(self):
        page = render_unified_dashboard(root_id="<img src=x>")
        self.assertNotIn("<img src=x>", page)
        self.assertIn("\\u003cimg src=x\\u003e", page)
        for placeholder in ("__STYLE__", "__APP__", "__MODE_LABEL__", "__PAYLOAD__"):
            self.assertNotIn(placeholder, page)

    def test_the_dashboard_page_is_the_only_shell_and_loads_only_live_assets(self):
        page = render_unified_dashboard(root_id="test-root")
        self.assertIn("Проекты и лаборатории", page)
        self.assertIn('<link rel="stylesheet" href="/assets/styles.css">', page)
        self.assertIn('<script src="/assets/live_app.js" defer></script>', page)
        self.assertIn('"root_id":"test-root"', page)
        self.assertNotIn("/assets/app.js", page)
        self.assertNotIn("Офлайн", page)
        self.assertNotIn("Гипотезы и решения", page)
        self.assertIsNone(read_asset("app.js"))
        live_script = read_asset("live_app.js")
        self.assertIsNotNone(live_script)
        for action in ("/commands", "startOperation", "pollJob", "/api/v1/jobs"):
            self.assertNotIn(action, live_script.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
