import os
import sys
from pathlib import Path


def _ensure_project_src_on_path() -> None:
    """Ensure the repository's src directory is on sys.path for importable 'modaic'.

    Params:
        None

    Returns:
        None
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))


_ensure_project_src_on_path()
