import sys
from unittest.mock import MagicMock
import unittest

# Mock problematic modules to allow importing train
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torch.nn.parallel'] = MagicMock()
sys.modules['torch.distributed'] = MagicMock()
sys.modules['torch.amp'] = MagicMock()
sys.modules['kernels'] = MagicMock()
sys.modules['prepare'] = MagicMock()

import train

class TestTrainUtils(unittest.TestCase):
    def test_get_muon_momentum(self):
        # frac = min(step / 300, 1)
        # return (1 - frac) * 0.85 + frac * 0.95
        self.assertAlmostEqual(train.get_muon_momentum(0), 0.85)
        self.assertAlmostEqual(train.get_muon_momentum(150), 0.90)
        self.assertAlmostEqual(train.get_muon_momentum(300), 0.95)
        self.assertAlmostEqual(train.get_muon_momentum(600), 0.95)

    def test_get_weight_decay(self):
        # return WEIGHT_DECAY * (1 - progress)
        # WEIGHT_DECAY = 0.2
        self.assertAlmostEqual(train.get_weight_decay(0.0), 0.2)
        self.assertAlmostEqual(train.get_weight_decay(0.5), 0.1)
        self.assertAlmostEqual(train.get_weight_decay(1.0), 0.0)

    def test_get_lr_multiplier(self):
        # WARMUP_RATIO = 0.0
        # WARMDOWN_RATIO = 0.5
        # FINAL_LR_FRAC = 0.0
        # def get_lr_multiplier(progress):
        #     if progress < WARMUP_RATIO:
        #         return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        #     elif progress < 1.0 - WARMDOWN_RATIO:
        #         return 1.0
        #     else:
        #         cooldown = (1.0 - progress) / WARMDOWN_RATIO
        #         return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

        # Test progress values given current constants
        self.assertAlmostEqual(train.get_lr_multiplier(0.0), 1.0) # progress >= WARMUP_RATIO(0.0)
        self.assertAlmostEqual(train.get_lr_multiplier(0.25), 1.0) # < 1.0 - WARMDOWN_RATIO(0.5)
        self.assertAlmostEqual(train.get_lr_multiplier(0.5), 1.0) # >= 1.0 - 0.5
        self.assertAlmostEqual(train.get_lr_multiplier(0.75), 0.5) # cooldown = (1.0-0.75)/0.5 = 0.5
        self.assertAlmostEqual(train.get_lr_multiplier(1.0), 0.0) # cooldown = 0

if __name__ == '__main__':
    unittest.main()
