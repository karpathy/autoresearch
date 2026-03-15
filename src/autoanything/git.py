"""Git operations — subprocess wrappers for git commands.

All functions accept a cwd parameter to operate on any directory.
No hardcoded paths or global state.
"""

import subprocess


def git(*args, cwd: str, check: bool = True):
    """Run a git command in the specified directory."""
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True, cwd=cwd,
    )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, ["git"] + list(args),
            output=result.stdout, stderr=result.stderr,
        )
    return result


def get_proposal_branches(cwd: str, pattern: str = "proposals/*"):
    """List remote proposal branches matching pattern."""
    result = git("branch", "-r", "--list", f"origin/{pattern}", cwd=cwd, check=False)
    branches = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line and not line.endswith("/HEAD"):
            branch = line.replace("origin/", "", 1)
            branches.append(branch)
    return branches


def get_head_commit(cwd: str) -> str:
    """Get the current HEAD commit SHA."""
    return git("rev-parse", "HEAD", cwd=cwd).stdout.strip()


def get_branch_commit(branch: str, cwd: str) -> str:
    """Get the commit SHA for a remote branch."""
    return git("rev-parse", f"origin/{branch}", cwd=cwd).stdout.strip()


def get_commit_message(commit_sha: str, cwd: str) -> str:
    """Get the first line of a commit message."""
    return git("log", "-1", "--format=%s", commit_sha, cwd=cwd).stdout.strip()


def merge_proposal(branch: str, base_branch: str, cwd: str):
    """Merge a successful proposal into the base branch."""
    git("checkout", base_branch, cwd=cwd)
    git("merge", f"origin/{branch}", "--no-ff",
        "-m", f"Merge {branch}: score improved", cwd=cwd)
