"""CLI argument normalization and fuzzy flag resolution for AutoResearch."""

from __future__ import annotations

from typing import Sequence

GLOBAL_VALUE_FLAGS = {"--config", "--project", "--lab"}
GLOBAL_BOOLEAN_FLAGS = {"--agent", "--confirmed"}
FLAG_ALIASES = {
    "--runs": "--repeats",
    "--from-report": "--source",
    "--input": "--source",
}


def normalize_cli_args(argv: Sequence[str]) -> list[str]:
    """Normalize global flags and aliases in CLI arguments.

    1. Map aliases (--runs -> --repeats, --from-report -> --source, --input -> --source).
    2. Hoist global flags (--agent, --confirmed, --config, --project, --lab) to the front so they
       can appear anywhere without triggering subparser unrecognized argument errors.
    """
    # Step 1: normalize aliases
    aliased: list[str] = []
    for arg in argv:
        if arg in FLAG_ALIASES:
            aliased.append(FLAG_ALIASES[arg])
        else:
            replaced = False
            for alias, target in FLAG_ALIASES.items():
                if arg.startswith(f"{alias}="):
                    aliased.append(f"{target}={arg[len(alias) + 1:]}")
                    replaced = True
                    break
            if not replaced:
                aliased.append(arg)

    # `storage export` and `storage reconcile` own nested --lab options. Keep
    # those options with the nested parser instead of hoisting them globally.
    command_index = 0
    while command_index < len(aliased):
        item = aliased[command_index]
        if item in GLOBAL_BOOLEAN_FLAGS:
            command_index += 1
        elif item in GLOBAL_VALUE_FLAGS and command_index + 1 < len(aliased):
            command_index += 2
        elif any(item.startswith(f"{flag}=") for flag in GLOBAL_VALUE_FLAGS):
            command_index += 1
        else:
            break
    storage_lab_start = (
        command_index + 2
        if aliased[command_index:command_index + 2] in (
            ["storage", "export"], ["storage", "reconcile"],
        )
        else None
    )

    # Step 2: separate global flags from subcommand arguments
    global_args: list[str] = []
    rest_args: list[str] = []
    i = 0
    while i < len(aliased):
        arg = aliased[i]
        local_storage_lab = storage_lab_start is not None and i >= storage_lab_start
        if local_storage_lab and arg == "--lab":
            rest_args.append(arg)
            if i + 1 < len(aliased):
                rest_args.append(aliased[i + 1])
                i += 2
            else:
                i += 1
        elif local_storage_lab and arg.startswith("--lab="):
            rest_args.append(arg)
            i += 1
        elif arg in GLOBAL_BOOLEAN_FLAGS:
            global_args.append(arg)
            i += 1
        elif arg in GLOBAL_VALUE_FLAGS:
            global_args.append(arg)
            if i + 1 < len(aliased):
                global_args.append(aliased[i + 1])
                i += 2
            else:
                i += 1
        elif any(arg.startswith(f"{flag}=") for flag in GLOBAL_VALUE_FLAGS):
            global_args.append(arg)
            i += 1
        else:
            rest_args.append(arg)
            i += 1

    return global_args + rest_args
