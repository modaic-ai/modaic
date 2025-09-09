import os
from pathlib import Path

from dotenv import load_dotenv
import re
load_dotenv()


def compute_cache_dir() -> Path:
    """Return the cache directory used to stage internal modules."""
    cache_dir_env = os.getenv("MODAIC_CACHE")
    default_cache_dir = Path(os.path.expanduser("~")) / ".cache" / "modaic"
    cache_dir = Path(cache_dir_env).expanduser().resolve() if cache_dir_env else default_cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def validate_project_name(text):
    """Letters, numbers, underscore, hyphen"""
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', text))
