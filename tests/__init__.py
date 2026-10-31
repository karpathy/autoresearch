"""Put the skill runtime on sys.path so tests do not depend on repo-root shims."""

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
