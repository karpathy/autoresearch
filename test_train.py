import unittest
import sys
from unittest.mock import MagicMock

# Mock dependencies that are not available in the environment
mock_torch = MagicMock()
mock_torch.nn = MagicMock()
mock_torch.nn.functional = MagicMock()
mock_torch.distributed = MagicMock()
mock_torch.nn.parallel = MagicMock()

sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_torch.nn
sys.modules['torch.nn.functional'] = mock_torch.nn.functional
sys.modules['torch.distributed'] = mock_torch.distributed
sys.modules['torch.nn.parallel'] = mock_torch.nn.parallel

# Mock kernels and prepare
sys.modules['kernels'] = MagicMock()
sys.modules['prepare'] = MagicMock()

# Now import the function to test
from train import norm

class TestNorm(unittest.TestCase):
    def test_norm_calls_rms_norm(self):
        # Setup
        x = MagicMock()
        # Mock size to return 768 when called as x.size(-1)
        x.size.return_value = 768

        # Execute
        norm(x)

        # Verify
        mock_torch.nn.functional.rms_norm.assert_called_once_with(x, (768,))
        x.size.assert_called_once_with(-1)

    def test_norm_with_different_size(self):
        # Setup
        x = MagicMock()
        x.size.return_value = 512

        # Execute
        norm(x)

        # Verify
        mock_torch.nn.functional.rms_norm.assert_called_with(x, (512,))
        x.size.assert_called_with(-1)

if __name__ == '__main__':
    unittest.main()
