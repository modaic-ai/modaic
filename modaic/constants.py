import os
from pathlib import Path

from .utils import compute_cache_dir

MODAIC_CACHE = compute_cache_dir()
PROGRAMS_CACHE = Path(MODAIC_CACHE) / "programs"
EDITABLE_MODE = os.getenv("EDITABLE_MODE", "false").lower() == "true"
TEMP_DIR = Path(MODAIC_CACHE) / "temp"
SYNC_DIR = Path(MODAIC_CACHE) / "sync"


MODAIC_TOKEN = os.getenv("MODAIC_TOKEN")
MODAIC_GIT_URL = os.getenv("MODAIC_GIT_URL", "git.modaic.dev").replace("https://", "").rstrip("/")

USE_GITHUB = "github.com" in MODAIC_GIT_URL

MODAIC_API_URL = os.getenv("MODAIC_API_URL", "https://api.modaic.dev")