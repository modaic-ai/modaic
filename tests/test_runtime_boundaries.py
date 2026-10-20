from __future__ import annotations

import ast
from pathlib import Path


def test_runtime_has_no_git_or_unrelated_filesystem_implementation() -> None:
    root = Path(__file__).parents[1] / "src" / "modaic"
    forbidden = {"git", "gitpython", "shutil", "subprocess", "tempfile"}
    for source in root.glob("*.py"):
        imported: set[str] = set()
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported.isdisjoint(forbidden)
        if source.name != "_knowledge_base.py":
            assert "pathlib" not in imported
