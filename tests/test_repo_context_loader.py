import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autosaas.repo_context_loader import load_repo_context


def test_load_repo_context_detects_nextjs_repo(tmp_path):
    (tmp_path / "package.json").write_text('{"scripts": {"lint": "next lint", "dev": "next dev"}}')
    (tmp_path / "app").mkdir()
    ctx = load_repo_context(tmp_path)
    assert ctx.framework == "nextjs"
    assert ctx.package_manager in {"pnpm", "npm", "yarn", "unknown"}
    assert ctx.scripts == {"lint": "next lint", "dev": "next dev"}
    assert ctx.sensitive_paths == [".env", ".env.local", ".env.production"]
