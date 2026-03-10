"""
One-time setup script for the autoresearch-at-home Ensue hub org.

Prerequisites:
  1. Create the 'autoresearch-at-home' org at https://ensue-network.ai
  2. Get an API key for the org (via web UI: Settings > API Keys)
  3. Set ENSUE_API_KEY env var or pass --api-key

Usage:
  ENSUE_API_KEY=lmn_... uv run setup_hub.py
  uv run setup_hub.py --api-key lmn_...
  uv run setup_hub.py --api-key lmn_... --seed-train-py train.py
"""

import argparse
import base64
import json
import os
import sys

from coordinator import ensue_rpc, HUB_ORG

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rpc(api_key: str, tool: str, args: dict) -> dict:
    """RPC wrapper with logging."""
    print(f"  → {tool}({json.dumps(args, indent=None)[:120]}...)")
    result = ensue_rpc(api_key, tool, args)
    return result


def share(api_key: str, command: dict) -> dict:
    """Share tool wrapper."""
    return rpc(api_key, "share", {"command": command})

# ---------------------------------------------------------------------------
# Setup steps
# ---------------------------------------------------------------------------

def setup_hub(api_key: str, seed_train_py: str = "train.py"):
    print(f"Setting up hub org: {HUB_ORG}")
    print("=" * 60)

    # 1. Create auto-approve invite link
    print("\n[1/7] Creating auto-approve invite link...")
    invite = rpc(api_key, "create_invite", {"auto_approve": True})
    invite_token = invite.get("token", "")
    print(f"  Invite token: {invite_token}")

    # 2. Create participants group
    print("\n[2/7] Creating 'participants' group...")
    share(api_key, {"command": "create_group", "group_name": "participants"})

    # 3. Set external group
    print("\n[3/7] Setting external auto-join group to 'participants'...")
    share(api_key, {"command": "set_external_group", "group_name": "participants"})

    # 4. Grant permissions to participants group
    print("\n[4/7] Granting permissions to 'participants' group...")
    target = {"type": "group", "group_name": "participants"}
    namespaces = ["claims/", "results/", "hypotheses/", "insights/", "best/", "leaderboard"]
    for ns in namespaces:
        for action in ["read", "create"]:
            share(api_key, {
                "command": "grant",
                "target": target,
                "action": action,
                "key_pattern": ns,
            })
    # Also grant update on best/ (so participants can update global best)
    share(api_key, {
        "command": "grant",
        "target": target,
        "action": "update",
        "key_pattern": "best/",
    })
    # And update on leaderboard
    share(api_key, {
        "command": "grant",
        "target": target,
        "action": "update",
        "key_pattern": "leaderboard",
    })

    # 5. Make leaderboard and best/ public
    print("\n[5/7] Making leaderboard and best/ publicly readable...")
    share(api_key, {"command": "make_public", "key_pattern": "leaderboard"})
    share(api_key, {"command": "make_public", "key_pattern": "best/"})
    share(api_key, {"command": "make_public", "key_pattern": "results/"})

    # 6. Initialize seed keys
    print("\n[6/7] Initializing seed keys...")

    # Read current train.py as baseline
    with open(seed_train_py) as f:
        train_py_source = f.read()

    # best/train_py
    code_b64 = base64.b64encode(train_py_source.encode()).decode()
    rpc(api_key, "create_memory", {"items": [{
        "key_name": "best/train_py",
        "description": "[autoresearch] Current best train.py source code",
        "value": code_b64,
        "base64": True,
    }]})

    # best/metadata
    meta = {
        "val_bpb": None,
        "memory_gb": None,
        "status": "baseline",
        "commit": "initial",
        "description": "baseline train.py",
        "agent_id": "hub-setup",
        "completed_at": None,
    }
    meta_b64 = base64.b64encode(json.dumps(meta).encode()).decode()
    rpc(api_key, "create_memory", {"items": [{
        "key_name": "best/metadata",
        "description": "[autoresearch] Metadata for current best train.py",
        "value": meta_b64,
        "base64": True,
    }]})

    # leaderboard
    leaderboard = {"entries": [], "updated_at": None}
    lb_b64 = base64.b64encode(json.dumps(leaderboard).encode()).decode()
    rpc(api_key, "create_memory", {"items": [{
        "key_name": "leaderboard",
        "description": "[autoresearch] Global leaderboard - best results per participant",
        "value": lb_b64,
        "base64": True,
    }]})

    # 7. Print summary
    print("\n[7/7] Done!")
    print("=" * 60)
    print(f"Hub org:      {HUB_ORG}")
    print(f"Invite token: {invite_token}")
    print(f"Invite URL:   https://ensue-network.ai/join?token={invite_token}")
    print()
    print("Embed this invite token in program.md so agents can auto-join.")
    print(f"Participants will access shared keys via @{HUB_ORG}/ prefix.")

    return invite_token


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Set up autoresearch-at-home Ensue hub org")
    parser.add_argument("--api-key", help="Ensue API key (or set ENSUE_API_KEY env var)")
    parser.add_argument("--seed-train-py", default="train.py", help="Path to baseline train.py")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ENSUE_API_KEY")
    if not api_key:
        print("Error: No API key. Set ENSUE_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    setup_hub(api_key, args.seed_train_py)


if __name__ == "__main__":
    main()
