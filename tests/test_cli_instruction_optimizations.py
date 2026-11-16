from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from modaic import cli


@pytest.mark.parametrize("command", ["instruction-optimizations", "instruct-opts"])
@pytest.mark.parametrize("action", ["create", "list", "get", "logs", "cancel"])
def test_instruction_optimization_commands_dispatch_to_resource(
    action: str, command: str, monkeypatch: Any, capsys: Any
) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    monkeypatch.setattr(cli, "_client", lambda args: client)
    monkeypatch.setattr(cli, "_model", lambda client, path: MagicMock())
    monkeypatch.setattr(cli, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_success", lambda *args, **kwargs: None)
    args = [command, action]
    if action in {"create", "list"}:
        args += ["acme/model"]
    else:
        args += ["job-id"]
    if action == "create":
        args += [
            "--training-file",
            "train",
            "--source-commit-sha",
            "abc",
            "--idempotency-key",
            "key",
        ]
    assert cli.main(args) == 0
    getattr(client.instruction_optimizations, action).assert_called_once()


def test_base_model_optimization_does_not_lookup_a_saved_model(monkeypatch: Any) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    monkeypatch.setattr(cli, "_client", lambda args: client)
    lookup = MagicMock(side_effect=AssertionError("No model lookup required"))
    monkeypatch.setattr(cli, "_model", lookup)
    monkeypatch.setattr(cli, "_emit", lambda *args, **kwargs: None)
    assert (
        cli.main(
            [
                "instruct-opts",
                "create",
                "--model",
                "mo-latest",
                "--destination",
                "acme/triage",
                "--training-file",
                "train",
                "--validation-file",
                "validation",
                "--eval-only-dataset",
                "heldout",
                "--idempotency-key",
                "optimization-dataset-1",
            ]
        )
        == 0
    )
    lookup.assert_not_called()
    args, kwargs = client.instruction_optimizations.create.call_args
    assert args == ("mo-latest",)
    assert kwargs["training_file"] == "train"
    assert kwargs["validation_file"] == "validation"
    assert kwargs["eval_only_dataset"] == "heldout"
    assert kwargs["destination"] == "acme/triage"
    assert "source_commit_sha" not in kwargs
