"""Small, data-agnostic experiment runner.

An evaluator owns the domain data and writes one result.json per repeat. This
module only manages files, comparable metrics, and experiment history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .dashboard import render_report_page
from .catalog import update_lab_status




class LabError(Exception):
    """A configuration or experiment contract failure."""

    def __init__(self, message: str, code: str | None = None, hint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint
