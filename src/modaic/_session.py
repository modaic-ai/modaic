from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

import keyring
from keyring.errors import KeyringError, NoKeyringError

from ._credentials import config_directory

StorageBackend = Literal["keyring", "file"]
_SERVICE = "modaic-cli"


@dataclass(frozen=True)
class StoredSession:
    access_token: str
    refresh_token: str
    expires_at: int
    api_url: str
    token_endpoint: str
    client_id: str
    subject: str
    email: str | None = None
    workspace: str | None = None


@dataclass(frozen=True)
class SessionMetadata:
    profile: str
    api_url: str
    storage: StorageBackend
    subject: str
    email: str | None
    expires_at: int
    workspace: str | None = None


def _origin_key(api_url: str) -> str:
    return hashlib.sha256(api_url.rstrip("/").encode()).hexdigest()


def _account(profile: str, api_url: str) -> str:
    return f"v1:{profile}:{_origin_key(api_url)}"


def _metadata_path(profile: str) -> Path:
    return config_directory() / "profiles" / f"{profile}.json"


def _secret_path(profile: str, api_url: str) -> Path:
    return config_directory() / "sessions" / f"{profile}-{_origin_key(api_url)}.json"


@contextmanager
def session_lock(profile: str) -> Iterator[None]:
    directory = config_directory() / "locks"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"{profile}.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.name == "nt":
            msvcrt = importlib.import_module("msvcrt")
            os.write(descriptor, b"0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":
            msvcrt = importlib.import_module("msvcrt")
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _private_write(path: Path, payload: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing to write credentials through symlink {path}.")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _read_object(path: Path) -> dict[str, object] | None:
    try:
        if path.is_symlink():
            raise ValueError(f"Refusing to read credentials through symlink {path}.")
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read Modaic credentials from {path}.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Modaic credentials in {path} are invalid.")
    return value


def save_session(
    session: StoredSession, *, profile: str = "default", storage: StorageBackend = "keyring"
) -> None:
    payload = json.dumps(asdict(session), separators=(",", ":"))
    if storage == "keyring":
        try:
            keyring.set_password(_SERVICE, _account(profile, session.api_url), payload)
        except (KeyringError, NoKeyringError) as exc:
            raise ValueError(
                "No usable OS keyring is available. Configure one or explicitly use --storage=file."
            ) from exc
    else:
        if os.name == "nt":
            raise ValueError("File credential storage is not supported on Windows yet.")
        _private_write(_secret_path(profile, session.api_url), payload)
    metadata = SessionMetadata(
        profile=profile,
        api_url=session.api_url,
        storage=storage,
        subject=session.subject,
        email=session.email,
        expires_at=session.expires_at,
        workspace=session.workspace,
    )
    _private_write(_metadata_path(profile), json.dumps(asdict(metadata), separators=(",", ":")))


def load_metadata(profile: str = "default") -> SessionMetadata | None:
    value = _read_object(_metadata_path(profile))
    if value is None:
        return None
    try:
        storage = value["storage"]
        expires_at = value["expires_at"]
        if storage not in {"keyring", "file"} or not isinstance(expires_at, int):
            raise ValueError
        return SessionMetadata(
            profile=str(value["profile"]),
            api_url=str(value["api_url"]),
            storage=storage,
            subject=str(value["subject"]),
            email=str(value["email"]) if value.get("email") is not None else None,
            expires_at=expires_at,
            workspace=str(value["workspace"]) if value.get("workspace") is not None else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Stored Modaic profile metadata is invalid.") from exc


def load_session(profile: str = "default") -> StoredSession | None:
    metadata = load_metadata(profile)
    if metadata is None:
        return None
    if metadata.storage == "keyring":
        try:
            payload = keyring.get_password(_SERVICE, _account(profile, metadata.api_url))
        except (KeyringError, NoKeyringError) as exc:
            raise ValueError("Could not read the Modaic session from the OS keyring.") from exc
        if payload is None:
            return None
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("Stored Modaic keyring session is invalid.") from exc
    else:
        value = _read_object(_secret_path(profile, metadata.api_url))
        if value is None:
            return None
    try:
        expires_at = value["expires_at"]
        if not isinstance(expires_at, int):
            raise ValueError
        return StoredSession(
            access_token=str(value["access_token"]),
            refresh_token=str(value["refresh_token"]),
            expires_at=expires_at,
            api_url=str(value["api_url"]),
            token_endpoint=str(value["token_endpoint"]),
            client_id=str(value["client_id"]),
            subject=str(value["subject"]),
            email=str(value["email"]) if value.get("email") is not None else None,
            workspace=str(value["workspace"]) if value.get("workspace") is not None else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Stored Modaic session is invalid.") from exc


def delete_session(profile: str = "default") -> bool:
    metadata = load_metadata(profile)
    if metadata is None:
        return False
    removed = False
    if metadata.storage == "keyring":
        try:
            keyring.delete_password(_SERVICE, _account(profile, metadata.api_url))
            removed = True
        except keyring.errors.PasswordDeleteError:
            pass
        except (KeyringError, NoKeyringError) as exc:
            raise ValueError("Could not remove the Modaic session from the OS keyring.") from exc
    else:
        try:
            _secret_path(profile, metadata.api_url).unlink()
            removed = True
        except FileNotFoundError:
            pass
    _metadata_path(profile).unlink(missing_ok=True)
    return removed
