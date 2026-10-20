from __future__ import annotations

import base64
import hashlib
import json
import secrets
import sys
import time
import webbrowser
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from urllib.parse import urlparse

import httpx

from ._session import (
    StorageBackend,
    StoredSession,
    load_metadata,
    load_session,
    save_session,
    session_lock,
)


@dataclass(frozen=True)
class LoginIdentity:
    email: str | None
    subject: str
    api_url: str
    expires_at: int


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(32)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=")
    return verifier, challenge.decode()


def _jwt_payload(token: str) -> dict[str, Any]:
    try:
        part = token.split(".")[1]
        padded = part + "=" * (-len(part) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded))
    except (IndexError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Supabase returned an invalid access token.") from exc
    if not isinstance(value, dict):
        raise ValueError("Supabase returned an invalid access token.")
    return value


def _require_secure_url(value: str, label: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError(f"{label} must use HTTPS.")
    return value


def _cli_version() -> str:
    try:
        return version("modaic")
    except PackageNotFoundError:
        return "0.0.0"


def browser_login(
    *,
    api_url: str,
    profile: str = "default",
    storage: StorageBackend = "keyring",
    headless: bool = False,
    timeout: float = 600,
    client: httpx.Client | None = None,
) -> LoginIdentity:
    owns_client = client is None
    http = client or httpx.Client(timeout=30)
    verifier, challenge = _pkce()
    try:
        created = http.post(
            f"{api_url.rstrip('/')}/auth/cli/authorizations",
            json={
                "codeChallenge": challenge,
                "codeChallengeMethod": "S256",
                "client": {
                    "name": "modaic-cli",
                    "version": _cli_version(),
                    "deviceName": __import__("platform").node() or "unknown-device",
                },
            },
        )
        created.raise_for_status()
        authorization = created.json()
        verification_uri = _require_secure_url(
            str(authorization["verificationUriComplete"]), "Verification URL"
        )
        print(f"Open {verification_uri}", file=sys.stderr)
        print(f"Confirm code: {authorization['userCode']}", file=sys.stderr)
        if not headless and not webbrowser.open(verification_uri):
            print(
                "Could not open a browser; open the URL above on any device.",
                file=sys.stderr,
            )

        deadline = time.monotonic() + timeout
        interval = max(1, int(authorization.get("interval", 5)))
        device_code = str(authorization["deviceCode"])
        approved: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            time.sleep(interval)
            response = http.post(
                f"{api_url.rstrip('/')}/auth/cli/authorizations/result",
                json={"deviceCode": device_code},
            )
            if response.status_code == 202:
                interval = max(interval, int(response.headers.get("retry-after", interval)))
                continue
            if response.status_code == 429:
                interval = max(interval + 5, int(response.headers.get("retry-after", interval + 5)))
                continue
            if response.status_code >= 400:
                code = response.json().get("code", "authorization_failed")
                raise ValueError(f"CLI authorization failed: {code}.")
            approved = response.json()
            break
        if approved is None:
            raise TimeoutError("CLI authorization timed out.")

        token_endpoint = _require_secure_url(str(approved["tokenEndpoint"]), "Token endpoint")
        client_id = str(approved["clientId"])
        token_response = http.post(
            token_endpoint,
            data={
                "grant_type": "authorization_code",
                "code": str(approved["authorizationCode"]),
                "client_id": client_id,
                "redirect_uri": str(approved["redirectUri"]),
                "code_verifier": verifier,
            },
            headers={"content-type": "application/x-www-form-urlencoded"},
        )
        token_response.raise_for_status()
        tokens = token_response.json()
        access_token = str(tokens["access_token"])
        refresh_token = str(tokens["refresh_token"])
        payload = _jwt_payload(access_token)
        if payload.get("client_id") != client_id or not isinstance(payload.get("sub"), str):
            raise ValueError("Supabase returned a token for the wrong OAuth client.")
        expires_at = int(payload.get("exp", int(time.time()) + int(tokens.get("expires_in", 3600))))

        identity_response = http.get(
            f"{api_url.rstrip('/')}/me",
            headers={"authorization": f"Bearer {access_token}"},
        )
        identity_response.raise_for_status()
        identity = identity_response.json()
        email = identity.get("email") if isinstance(identity.get("email"), str) else None
        subject = str(payload["sub"])
        save_session(
            StoredSession(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_at=expires_at,
                api_url=api_url.rstrip("/"),
                token_endpoint=token_endpoint,
                client_id=client_id,
                subject=subject,
                email=email,
            ),
            profile=profile,
            storage=storage,
        )
        return LoginIdentity(
            email=email,
            subject=subject,
            api_url=api_url.rstrip("/"),
            expires_at=expires_at,
        )
    except httpx.HTTPError as exc:
        raise ValueError("Could not complete Modaic browser login.") from exc
    finally:
        if owns_client:
            http.close()


def session_access_token(
    *, profile: str, api_url: str, client: httpx.Client | None = None
) -> str | None:
    with session_lock(profile):
        return _session_access_token_locked(profile=profile, api_url=api_url, client=client)


def _session_access_token_locked(
    *, profile: str, api_url: str, client: httpx.Client | None
) -> str | None:
    metadata = load_metadata(profile)
    if metadata is None:
        return None
    if metadata.api_url.rstrip("/") != api_url.rstrip("/"):
        raise ValueError(
            f"Stored profile {profile!r} belongs to {metadata.api_url}, not {api_url}."
        )
    session = load_session(profile)
    if session is None:
        return None
    if session.expires_at > int(time.time()) + 60:
        return session.access_token

    owns_client = client is None
    http = client or httpx.Client(timeout=30)
    try:
        response = http.post(
            _require_secure_url(session.token_endpoint, "Token endpoint"),
            data={
                "grant_type": "refresh_token",
                "refresh_token": session.refresh_token,
                "client_id": session.client_id,
            },
            headers={"content-type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        tokens = response.json()
        access_token = str(tokens["access_token"])
        payload = _jwt_payload(access_token)
        if payload.get("client_id") != session.client_id or payload.get("sub") != session.subject:
            raise ValueError("Supabase refreshed the wrong CLI session.")
        updated = StoredSession(
            access_token=access_token,
            refresh_token=str(tokens["refresh_token"]),
            expires_at=int(payload["exp"]),
            api_url=session.api_url,
            token_endpoint=session.token_endpoint,
            client_id=session.client_id,
            subject=session.subject,
            email=session.email,
        )
        save_session(updated, profile=profile, storage=metadata.storage)
        return updated.access_token
    except httpx.HTTPError as exc:
        raise ValueError(
            "The stored Modaic session expired; run `modaic auth login` again."
        ) from exc
    finally:
        if owns_client:
            http.close()


def revoke_session(
    *, profile: str, api_url: str, client: httpx.Client | None = None
) -> None:
    token = session_access_token(profile=profile, api_url=api_url, client=client)
    if token is None:
        return
    owns_client = client is None
    http = client or httpx.Client(timeout=30)
    try:
        response = http.delete(
            f"{api_url.rstrip('/')}/auth/cli/session",
            headers={"authorization": f"Bearer {token}"},
        )
        if response.status_code not in {204, 401}:
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ValueError("Could not confirm remote CLI session revocation.") from exc
    finally:
        if owns_client:
            http.close()
