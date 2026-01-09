import os
from pathlib import Path

from .utils import compute_cache_dir

MODAIC_CACHE = compute_cache_dir()
MODAIC_HUB_CACHE = Path(MODAIC_CACHE) / "modaic_hub" / "modaic_hub"
EDITABLE_MODE = os.getenv("EDITABLE_MODE", "false").lower() == "true"
STAGING_DIR = Path(MODAIC_CACHE) / "staging"
SYNC_DIR = Path(MODAIC_CACHE) / "sync"


MODAIC_TOKEN = os.getenv("MODAIC_TOKEN")
MODAIC_GIT_URL = os.getenv("MODAIC_GIT_URL", "https://git.modaic.dev").rstrip("/")

USE_GITHUB = "github.com" in MODAIC_GIT_URL

MODAIC_API_URL = os.getenv("MODAIC_API_URL", "https://api.modaic.dev").rstrip("/")
