import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from platformdirs import user_cache_dir

env_file = find_dotenv(usecwd=True)
load_dotenv(env_file)


def compute_cache_dir() -> Path:
    """Return the cache directory used to stage internal modules."""
    cache_dir_env = os.getenv("MODAIC_CACHE")
    if sys.platform.startswith("win"):
        default_cache_dir = Path(user_cache_dir("modaic", appauthor=False))
    else:
        default_cache_dir = Path(os.path.expanduser("~")) / ".cache" / "modaic"
    cache_dir = Path(cache_dir_env).expanduser().resolve() if cache_dir_env else default_cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def validate_project_name(text: str) -> bool:
    """Letters, numbers, underscore, hyphen"""
    assert bool(re.match(r"^[a-zA-Z0-9_]+$", text)), (
        "Invalid project name. Must contain only letters, numbers, and underscore."
    )


class Timer:
    def __init__(self, name: str):
        self.start_time = time.time()
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001, ANN002, ANN003
        self.done()

    def done(self):
        end_time = time.time()
        print(f"{self.name}: {end_time - self.start_time}s")  # noqa: T201


def smart_rmtree(path: Path, ignore_errors: bool = False) -> None:
    """
    Remove a directory and all its contents.
    If on windows use rmdir with /s flag
    If on mac/linux use rm -rf
    """
    if sys.platform.startswith("win"):
        try:
            shutil.rmtree(path, ignore_errors=False)
        except PermissionError:
            subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(path)], check=not ignore_errors)
        except Exception as e:
            if not ignore_errors:
                raise e
    else:
        shutil.rmtree(path, ignore_errors=ignore_errors)


def aggresive_rmtree(path: Path, missing_ok: bool = True) -> None:
    try:
        shutil.rmtree(path, ignore_errors=False)
    except FileNotFoundError as e:
        if not missing_ok:
            raise e
    except Exception as e:
        if sys.platform.startswith("win"):
            subprocess.run(["taskkill", "/F", "/IM", "git.exe"], capture_output=True, check=False)
            time.sleep(0.5)
            subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(path)], capture_output=True, check=True)
        else:
            raise e
