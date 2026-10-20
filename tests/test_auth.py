from __future__ import annotations

import base64
import json
import stat
from typing import Any

import httpx

from modaic import _auth
from modaic._session import StoredSession, delete_session, load_session, save_session


def _jwt(payload: dict[str, Any]) -> str:
    def encode(value: dict[str, Any]) -> str:
        return (
            base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode())
            .rstrip(b"=")
            .decode()
        )

    return f"{encode({'alg': 'ES256'})}.{encode(payload)}.signature"


def test_browser_login_polls_then_exchanges_pkce_without_relay_tokens(
    monkeypatch: Any,
) -> None:
    requests: list[httpx.Request] = []
    poll_count = 0
    access_token = _jwt({"sub": "user-1", "client_id": "cli-client", "exp": 2_000_000_000})

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal poll_count
        requests.append(request)
        if request.url.path.endswith("/auth/cli/authorizations"):
            body = json.loads(request.content)
            assert body["codeChallengeMethod"] == "S256"
            assert "codeVerifier" not in body
            return httpx.Response(
                201,
                json={
                    "deviceCode": "d" * 43,
                    "userCode": "ABCD-EFGH",
                    "verificationUriComplete": "https://modaic.dev/cli?user_code=ABCD-EFGH",
                    "interval": 5,
                },
            )
        if request.url.path.endswith("/auth/cli/authorizations/result"):
            poll_count += 1
            if poll_count == 1:
                return httpx.Response(202, json={"status": "authorization_pending"})
            return httpx.Response(
                200,
                json={
                    "status": "approved",
                    "authorizationCode": "one-time-code",
                    "tokenEndpoint": "https://project.supabase.co/auth/v1/oauth/token",
                    "clientId": "cli-client",
                    "redirectUri": "https://modaic.dev/auth/cli/callback",
                },
            )
        if request.url.path.endswith("/oauth/token"):
            assert b"code_verifier=" in request.content
            assert b"one-time-code" in request.content
            return httpx.Response(
                200,
                json={
                    "access_token": access_token,
                    "refresh_token": "refresh-secret",
                    "expires_in": 3600,
                },
            )
        if request.url.path.endswith("/me"):
            assert request.headers["authorization"] == f"Bearer {access_token}"
            return httpx.Response(200, json={"id": "user-1", "email": "user@example.com"})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    stored: list[StoredSession] = []
    monkeypatch.setattr(_auth.time, "sleep", lambda _: None)
    monkeypatch.setattr(_auth.webbrowser, "open", lambda _: True)
    monkeypatch.setattr(_auth, "save_session", lambda session, **_: stored.append(session))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        identity = _auth.browser_login(api_url="https://modaic.dev/api/v1", client=client)

    assert identity.email == "user@example.com"
    assert stored[0].refresh_token == "refresh-secret"
    assert all("refresh-secret" not in str(request.url) for request in requests)
    assert all("one-time-code" not in str(request.url) for request in requests)


def test_file_session_storage_is_explicit_private_and_deletable(
    monkeypatch: Any, tmp_path: Any
) -> None:
    monkeypatch.setenv("MODAIC_CONFIG_DIR", str(tmp_path))
    session = StoredSession(
        access_token="access-secret",
        refresh_token="refresh-secret",
        expires_at=2_000_000_000,
        api_url="https://modaic.dev/api/v1",
        token_endpoint="https://project.supabase.co/auth/v1/oauth/token",
        client_id="cli-client",
        subject="user-1",
        email="user@example.com",
    )
    save_session(session, storage="file")
    assert load_session() == session
    secret = next((tmp_path / "sessions").iterdir())
    assert stat.S_IMODE(secret.stat().st_mode) == 0o600
    assert "access-secret" not in (tmp_path / "profiles" / "default.json").read_text()
    assert delete_session() is True
    assert load_session() is None


def test_refresh_rotates_both_tokens(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.setenv("MODAIC_CONFIG_DIR", str(tmp_path))
    original = StoredSession(
        access_token="expired-access",
        refresh_token="old-refresh",
        expires_at=1,
        api_url="https://modaic.dev/api/v1",
        token_endpoint="https://project.supabase.co/auth/v1/oauth/token",
        client_id="cli-client",
        subject="user-1",
    )
    save_session(original, storage="file")
    refreshed_access = _jwt({"sub": "user-1", "client_id": "cli-client", "exp": 2_000_000_000})

    def handler(request: httpx.Request) -> httpx.Response:
        assert b"old-refresh" in request.content
        return httpx.Response(
            200,
            json={"access_token": refreshed_access, "refresh_token": "new-refresh"},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert (
            _auth.session_access_token(
                profile="default",
                api_url="https://modaic.dev/api/v1",
                client=client,
            )
            == refreshed_access
        )
    rotated = load_session()
    assert rotated is not None
    assert rotated.refresh_token == "new-refresh"
    assert rotated.access_token == refreshed_access


def test_session_is_never_sent_to_a_different_api_origin(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.setenv("MODAIC_CONFIG_DIR", str(tmp_path))
    save_session(
        StoredSession(
            access_token="access",
            refresh_token="refresh",
            expires_at=2_000_000_000,
            api_url="https://modaic.dev/api/v1",
            token_endpoint="https://project.supabase.co/auth/v1/oauth/token",
            client_id="cli-client",
            subject="user-1",
        ),
        storage="file",
    )
    try:
        _auth.session_access_token(profile="default", api_url="https://stage.modaic.dev/api/v1")
    except ValueError as exc:
        assert "belongs to" in str(exc)
    else:
        raise AssertionError("Cross-origin credential use was not rejected")


def test_logout_revokes_only_the_current_cli_session(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.setenv("MODAIC_CONFIG_DIR", str(tmp_path))
    save_session(
        StoredSession(
            access_token="current-access",
            refresh_token="current-refresh",
            expires_at=2_000_000_000,
            api_url="https://modaic.dev/api/v1",
            token_endpoint="https://project.supabase.co/auth/v1/oauth/token",
            client_id="cli-client",
            subject="user-1",
        ),
        storage="file",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/v1/auth/cli/session"
        assert request.headers["authorization"] == "Bearer current-access"
        return httpx.Response(204)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        _auth.revoke_session(profile="default", api_url="https://modaic.dev/api/v1", client=client)
