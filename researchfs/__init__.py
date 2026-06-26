"""ResearchFS v0: lightweight literature retrieval and experiment logging package."""

from .db import initialize_db, get_connection

__all__ = ["initialize_db", "get_connection"]
