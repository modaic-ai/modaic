from __future__ import annotations

import ast
from pathlib import Path


def test_runtime_has_no_git_or_filesystem_implementation() -> None:
    root = Path(__file__).parents[1] / "src" / "modaic"
    forbidden = {"git", "gitpython", "pathlib", "shutil", "subprocess", "tempfile"}
    imported: set[str] = set()
    for source in root.glob("*.py"):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(forbidden)
