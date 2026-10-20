from __future__ import annotations

import json
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class StoredCredentials:
    api_key: str
    api_url: str | None = None


def config_directory() -> Path:
    override = os.getenv("MODAIC_CONFIG_DIR")
    if override:
        return Path(override)
    if os.name == "nt":
        root = os.getenv("APPDATA")
        if root:
            return Path(root) / "modaic"
    root = os.getenv("XDG_CONFIG_HOME")
    if root:
        return Path(root) / "modaic"
    return Path.home() / ".config" / "modaic"


def credentials_path() -> Path:
    return config_directory() / "credentials.json"


def load_credentials() -> StoredCredentials | None:
    path = credentials_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read Modaic credentials from {path}.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("api_key"), str):
        raise ValueError(f"Modaic credentials in {path} are invalid.")
    api_key = data["api_key"].strip()
    api_url = data.get("api_url")
    if not api_key or (api_url is not None and not isinstance(api_url, str)):
        raise ValueError(f"Modaic credentials in {path} are invalid.")
    return StoredCredentials(api_key=api_key, api_url=api_url)


def save_credentials(credentials: StoredCredentials) -> Path:
    directory = config_directory()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with suppress(OSError):
        directory.chmod(0o700)
    path = credentials_path()
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    payload = json.dumps(
        {"version": 1, "api_key": credentials.api_key, "api_url": credentials.api_url},
        separators=(",", ":"),
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")
        os.replace(temporary, path)
        with suppress(OSError):
            path.chmod(0o600)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return path


def delete_credentials() -> bool:
    try:
        credentials_path().unlink()
    except FileNotFoundError:
        return False
    return True
