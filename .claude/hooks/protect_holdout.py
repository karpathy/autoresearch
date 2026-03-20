#!/usr/bin/env python3
"""PreToolUse hook: Block attempts to read the holdout salt file.

The salt lives in ~/.config/autotrader/holdout_salt (a file, NOT an env var).
This hook blocks Bash commands that reference the salt file path or attempt
to discover it via bulk environment/variable dumps.

Install: register in .claude/settings.local.json under hooks.PreToolUse
"""
import json
import re
import sys

event = json.load(sys.stdin)
tool_name = event.get("tool_name", "")
tool_input = event.get("tool_input", {})

if tool_name == "Bash":
    cmd = tool_input.get("command", "")

    # Patterns that would reveal the holdout salt
    blocked_patterns = [
        # --- Salt file path (various forms) ---
        r"holdout_salt",                   # the filename itself
        r"\.config/autotrader",            # parent directory
        r"config.*autotrader.*salt",       # fragments reassembled

        # --- Env var (legacy, in case still exported) ---
        r"AUTOTRADER_HOLDOUT_SALT",
        r"HOLDOUT_SALT",

        # --- Bulk env/variable dumps ---
        r"\bprintenv\b",                   # dump all env vars
        r"\benv\b\s*($|\|)",              # bare `env` or `env |`
        r"\bset\b\s*\|",                  # `set | grep` etc
        r"\bexport\b\s+-p",               # `export -p`
        r"\bcompgen\b.*-[eAv]",           # bash completion dump of vars

        # --- Programmatic env/file access patterns ---
        r"os\.environ",                    # Python env access
        r"getenv\s*\(",                   # C-style or Python getenv
        r"\bENV\b\s*\[",                  # Ruby ENV[]
        r"process\.env",                   # Node.js env access
        r"subprocess.*env",                # subprocess env access

        # --- Shell config files that might contain exports ---
        r"cat\s+~/\.(?:zshrc|bash_profile|bashrc|profile)",
        r"source\s+~/\.(?:zshrc|bash_profile|bashrc|profile)",

        # --- Broad Python/Ruby/Perl with import obfuscation ---
        r"importlib.*import_module",       # obfuscated Python imports
    ]

    for pattern in blocked_patterns:
        if re.search(pattern, cmd, re.IGNORECASE):
            print(
                "BLOCKED: This command would access holdout configuration. "
                "The holdout selection is deliberately opaque — see rule 13 in program.md.",
                file=sys.stderr,
            )
            sys.exit(2)

# Allow everything else
print(json.dumps({"decision": "allow"}))
