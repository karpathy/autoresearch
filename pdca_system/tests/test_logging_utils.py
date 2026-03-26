from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pdca_system import logging_utils


class LoggingUtilsTests(unittest.TestCase):
    def test_get_logger_writes_to_shared_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            logger_name = "pdca_system.tests.logging_utils"
            raw_logger = logging.getLogger(logger_name)
            for handler in list(raw_logger.handlers):
                raw_logger.removeHandler(handler)

            original_handler = logging_utils._FILE_HANDLER
            logging_utils._FILE_HANDLER = None
            try:
                with patch.object(logging_utils, "LOG_ROOT", temp_root):
                    logger = logging_utils.get_logger(logger_name)
                    logger.info("shared logger smoke test")
                    for handler in logger.handlers:
                        handler.flush()

                log_path = temp_root / logging_utils.LOG_FILE_NAME
                self.assertTrue(log_path.exists())
                contents = log_path.read_text(encoding="utf-8")
                self.assertIn("shared logger smoke test", contents)
                self.assertIn(logger_name, contents)
            finally:
                for handler in list(raw_logger.handlers):
                    raw_logger.removeHandler(handler)
                    handler.close()
                logging_utils._FILE_HANDLER = original_handler
