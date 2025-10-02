import os
import re
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

env_file = find_dotenv(usecwd=True)
load_dotenv(env_file)


def compute_cache_dir() -> Path:
    """Return the cache directory used to stage internal modules."""
    cache_dir_env = os.getenv("MODAIC_CACHE")
    default_cache_dir = Path(os.path.expanduser("~")) / ".cache" / "modaic"
    cache_dir = Path(cache_dir_env).expanduser().resolve() if cache_dir_env else default_cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def validate_project_name(text: str) -> bool:
    """Letters, numbers, underscore, hyphen"""
    assert bool(re.match(r"^[a-zA-Z0-9_]+$", text)), (
        "Invalid project name. Must contain only letters, numbers, and underscore."
    )
