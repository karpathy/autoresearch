from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock

from pdca_system.task import LOG_ROOT

LOG_FILE_NAME = "pdca.log"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

_FILE_HANDLER: logging.Handler | None = None
_FILE_HANDLER_LOCK = Lock()


def log_file_path() -> Path:
    return LOG_ROOT / LOG_FILE_NAME


def _build_file_handler() -> logging.Handler:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file_path(), encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return handler


def shared_file_handler() -> logging.Handler:
    global _FILE_HANDLER
    if _FILE_HANDLER is None:
        with _FILE_HANDLER_LOCK:
            if _FILE_HANDLER is None:
                _FILE_HANDLER = _build_file_handler()
    return _FILE_HANDLER


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    handler = shared_file_handler()
    if handler not in logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
