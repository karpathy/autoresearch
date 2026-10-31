try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from autoresearch_runtime.baseline import _request_identity_digest
from autoresearch_runtime.wire_record import record_sent_request


class WirePrivacyTest(unittest.TestCase):
    def test_request_identity_requires_explicit_body_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "lab"
            repeat = state / "runs" / "baseline" / "repeat_01"
            repeat.mkdir(parents=True)
            log_path = repeat / "sent-requests.jsonl"
            config = SimpleNamespace(state_dir=state)

            with patch.dict(os.environ, {"AUTORESEARCH_WIRE_LOG": str(log_path), "AUTORESEARCH_WIRE_LOG_BODY": ""}):
                record_sent_request(
                    url="https://example.test/v1/chat", method="POST", body='{"prompt":"x"}',
                    headers={}, status=200, response_body="{}", attempt=1,
                )
            self.assertIsNone(_request_identity_digest(config, "baseline", 1))

            log_path.unlink()
            body = '{"prompt":"x"}'
            with patch.dict(os.environ, {"AUTORESEARCH_WIRE_LOG": str(log_path), "AUTORESEARCH_WIRE_LOG_BODY": "1"}):
                record_sent_request(
                    url="https://example.test/v1/chat", method="POST", body=body,
                    headers={}, status=200, response_body="{}", attempt=1,
                )
            self.assertEqual(
                _request_identity_digest(config, "baseline", 1),
                hashlib.sha256(body.encode("utf-8")).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
