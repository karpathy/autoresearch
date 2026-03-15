"""Autoresearch training package."""

def main() -> int:
    from .runner import main as runner_main

    return runner_main()

__all__ = ["main"]
