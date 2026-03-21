import unittest
from policy_engine import decide, Result

class TestPolicyEngine(unittest.TestCase):

    def setUp(self):
        """Set up a baseline result for all tests."""
        self.baseline = Result(val_bpb=1.5, complexity=100, status='success')

    def test_discard_crash(self):
        """A crashed candidate should be discarded."""
        candidate = Result(val_bpb=1.0, complexity=50, status='crash')
        decision = decide(candidate, self.baseline)
        self.assertEqual(decision.action, 'DISCARD')
        self.assertIn("status is 'crash'", decision.reason)

    def test_discard_timeout(self):
        """A timed-out candidate should be discarded."""
        candidate = Result(val_bpb=1.0, complexity=50, status='timeout')
        decision = decide(candidate, self.baseline)
        self.assertEqual(decision.action, 'DISCARD')
        self.assertIn("status is 'timeout'", decision.reason)

    def test_keep_significant_improvement(self):
        """Keep if val_bpb is significantly lower, even with higher complexity."""
        candidate = Result(val_bpb=1.4, complexity=120, status='success')
        decision = decide(candidate, self.baseline)
        self.assertEqual(decision.action, 'KEEP')
        self.assertIn("Significant val_bpb improvement", decision.reason)

    def test_discard_worse_val_bpb(self):
        """Discard if val_bpb is worse, even with lower complexity."""
        candidate = Result(val_bpb=1.6, complexity=80, status='success')
        decision = decide(candidate, self.baseline)
        self.assertEqual(decision.action, 'DISCARD')
        self.assertIn("val_bpb is worse", decision.reason)

    def test_discard_marginal_improvement_with_higher_complexity(self):
        """Discard if val_bpb is only marginally better but complexity is higher."""
        candidate = Result(val_bpb=1.4995, complexity=110, status='success')
        decision = decide(candidate, self.baseline)
        self.assertEqual(decision.action, 'DISCARD')
        self.assertIn("complexity increased", decision.reason)

    def test_keep_comparable_val_bpb_with_lower_complexity(self):
        """Keep if val_bpb is comparable but complexity is lower."""
        candidate = Result(val_bpb=1.5001, complexity=90, status='success')
        decision = decide(candidate, self.baseline)
        self.assertEqual(decision.action, 'KEEP')
        self.assertIn("complexity is lower", decision.reason)

    def test_keep_comparable_val_bpb_with_same_complexity(self):
        """Keep if val_bpb and complexity are identical."""
        candidate = Result(val_bpb=1.5, complexity=100, status='success')
        decision = decide(candidate, self.baseline)
        self.assertEqual(decision.action, 'KEEP')
        self.assertIn("identical to baseline", decision.reason)


    def test_discard_comparable_val_bpb_with_higher_complexity(self):
        """Discard if val_bpb is comparable but complexity is higher."""
        candidate = Result(val_bpb=1.5, complexity=110, status='success')
        decision = decide(candidate, self.baseline)
        self.assertEqual(decision.action, 'DISCARD')
        self.assertIn("complexity increased", decision.reason)

    def test_keep_identical_results(self):
        """Keep if the candidate is identical to the baseline."""
        candidate = Result(val_bpb=1.5, complexity=100, status='success')
        decision = decide(candidate, self.baseline)
        self.assertEqual(decision.action, 'KEEP')
        self.assertIn("identical to baseline", decision.reason)

if __name__ == '__main__':
    unittest.main()
