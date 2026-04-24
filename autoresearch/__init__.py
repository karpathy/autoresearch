"""autoresearch — nanoGPT-based autonomous ML research loop (research prototype).

This package exposes a small, CPU-friendly skeleton derived from Karpathy's
nanoGPT, plus a CLI for toy training, environment info, and config inspection.

NOTE (honest scope for v1.0.0):
    * Tribunal evaluation is NOT IMPLEMENTED. A stub lives in
      ``autoresearch.tribunal`` so callers can import it without errors.
    * The P2PCLAW Silicon integration is PARTIAL; ``silicon/grid_generator.py``
      in the repo root is a placeholder.
    * The Railway P2PCLAW API dependency is OPTIONAL. The core training loop
      works without network access.

See ``ROADMAP.md`` at the repository root for what's still missing.
"""

from .config import Config, load_config
from .tokenizer import CharTokenizer
from .model import GPT, GPTConfig
from . import tribunal  # noqa: F401  (stub)

__all__ = [
    "Config",
    "load_config",
    "CharTokenizer",
    "GPT",
    "GPTConfig",
    "tribunal",
    "__version__",
]

__version__ = "1.0.0"
