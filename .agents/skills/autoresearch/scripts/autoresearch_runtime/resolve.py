"""Pick a lab from an explicit flag or the SQLite active-lab table."""

from __future__ import annotations

from pathlib import Path

from .errors import LabError
from .paths import home_root_id, project_id, resolve_home


def _store(home: Path):
    from .sqlite_store import SQLiteStore
    return SQLiteStore(home)


def _resolve_lab_flag(home: Path, lab: str) -> Path | str:
    from .sqlite_store import AmbiguousAliasError, EntityNotFoundError

    store = _store(home)
    if store.is_initialized():
        try:
            return str(store.resolve_lab_id(lab))
        except AmbiguousAliasError as ambiguous:
            raise LabError(
                "Several SQLite labs match --lab; pass a unique --lab or --config",
                code="E_LAB_AMBIGUOUS",
                hint="Use an unambiguous lab reference or a known `lab.json` path with --config.",
            ) from ambiguous
        except EntityNotFoundError:
            pass
    selected = Path(lab).expanduser()
    if selected.is_absolute() or "/" in lab or "\\" in lab or selected.name.casefold() == "lab.json":
        return selected  # load_config reports an unknown path or an unsupported legacy file
    raise LabError(
        f"No lab matches --lab {lab!r}",
        code="E_LAB_NOT_FOUND",
        hint="Pass --config <known-lab.json>, use a valid --lab reference, or run prepare for a new lab.",
    )


def _lab_for_project(home: Path, target: Path) -> str | None:
    store = _store(home)
    if not store.is_initialized():
        return None
    active = store.get_active_lab(home_root_id(home), project_id(target))
    if active is not None:
        return str(active)
    page = store.list_labs(limit=100, lifecycle="active", filters={"source_root": target})
    items = page["items"]
    if len(items) > 1:
        lines = [f"  {item['config_path']} ({item['lab_id']})" for item in items]
        raise LabError(
            f"Several labs match project {target}; use --config or --lab:\n" + "\n".join(lines),
            code="E_LAB_AMBIGUOUS",
            hint="prepare --resume or pass --config explicitly",
        )
    return str(items[0]["lab_id"]) if items else None


def resolve_config_ref(
    *,
    config: Path | None,
    lab: str | None,
    project: Path | None,
    labs_root: Path | None,
) -> Path | str:
    """Return a SQLite lab UUID or a lab.json path for `config.load_config`."""
    if config is not None:
        return config
    home = (labs_root or resolve_home()).resolve()
    if lab:
        return _resolve_lab_flag(home, lab)
    target = (project or Path.cwd()).resolve()
    found = _lab_for_project(home, target)
    if found is not None:
        return found
    if project is None and (target / "lab.json").is_file():
        return target / "lab.json"  # load_config reports E_LEGACY_LAB for a real file
    message = (f"No lab for {target}; prepare a new SQLite lab first" if project is not None
               else f"No lab found for {target}")
    raise LabError(
        message,
        code="E_NO_LAB",
        hint="Run prepare with a reviewed spec and dedicated AUTORESEARCH_HOME, or pass --config <lab.json>, --lab <id>, or --project <path>",
    )
