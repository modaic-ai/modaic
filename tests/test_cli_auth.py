from __future__ import annotations

import io
from typing import Any

import pytest

from modaic import cli
from modaic._auth import LoginIdentity
from modaic._credentials import StoredCredentials, load_credentials, save_credentials
from modaic._session import StoredSession, load_session, save_session


@pytest.mark.parametrize("command", [["login"], ["auth", "login"]])
def test_login_opens_browser_flow_with_selected_options(
    monkeypatch: Any, capsys: Any, command: list[str]
) -> None:
    calls: list[dict[str, Any]] = []

    def browser_login(**kwargs: Any) -> LoginIdentity:
        calls.append(kwargs)
        return LoginIdentity(
            "user@example.com", "user-id", kwargs["api_url"], 2_000_000_000, kwargs["workspace"]
        )

    monkeypatch.setattr(cli, "browser_login", browser_login)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO())
    assert (
        cli.main(
            [
                "--json",
                *command,
                "--headless",
                "--profile",
                "work",
                "--workspace",
                "acme",
                "--storage",
                "file",
                "--timeout",
                "30",
            ]
        )
        == 0
    )
    assert calls == [
        {
            "api_url": "https://modaic.dev/api/v1",
            "profile": "work",
            "storage": "file",
            "workspace": "acme",
            "headless": True,
            "timeout": 30,
        }
    ]
    output = capsys.readouterr().out
    assert '"profile":"work"' in output
    assert '"workspace":"acme"' in output


@pytest.mark.parametrize("command", [["logout"], ["auth", "logout"]])
@pytest.mark.parametrize("local_only", [False, True])
def test_logout_removes_selected_session_and_honors_local_only(
    monkeypatch: Any, tmp_path: Any, command: list[str], local_only: bool
) -> None:
    monkeypatch.setenv("MODAIC_CONFIG_DIR", str(tmp_path))
    save_session(
        StoredSession(
            "access",
            "refresh",
            2_000_000_000,
            "https://modaic.dev/api/v1",
            "https://auth.example/oauth/token",
            "client",
            "user",
        ),
        profile="work",
        storage="file",
    )
    revoked: list[str] = []
    monkeypatch.setattr(cli, "revoke_session", lambda **kw: revoked.append(kw["profile"]))
    args = [*command, "--profile", "work"] + (["--local-only"] if local_only else [])
    assert cli.main(args) == 0
    assert load_session("work") is None
    assert revoked == ([] if local_only else ["work"])


def test_token_stdin_keeps_api_key_login(monkeypatch: Any, tmp_path: Any) -> None:
    from types import SimpleNamespace

    monkeypatch.setenv("MODAIC_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("test-api-key\n"))

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["api_key"] == "test-api-key"
            self.identity = SimpleNamespace(me=lambda: SimpleNamespace(email="user@example.com"))

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    monkeypatch.setattr(cli, "Modaic", Client)
    assert cli.main(["login", "--token-stdin"]) == 0
    stored = load_credentials()
    assert stored is not None and stored.api_key == "test-api-key"
    assert cli.main(["logout", "--api-key"]) == 0
    assert load_credentials() is None


def test_remote_logout_failure_still_removes_session(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.setenv("MODAIC_CONFIG_DIR", str(tmp_path))
    save_session(
        StoredSession(
            "access",
            "refresh",
            2_000_000_000,
            "https://modaic.dev/api/v1",
            "https://auth.example/oauth/token",
            "client",
            "user",
        ),
        storage="file",
    )
    save_credentials(StoredCredentials("stored-key", "https://modaic.dev/api/v1"))

    def fail(**kwargs: Any) -> None:
        raise ValueError("revocation unavailable")

    monkeypatch.setattr(cli, "revoke_session", fail)
    with pytest.raises(SystemExit) as error:
        cli.main(["logout"])
    assert error.value.code == 3
    assert load_session() is None
    assert load_credentials() is not None
