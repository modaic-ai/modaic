from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, cast
from uuid import uuid4

from pydantic import BaseModel

from ._auth import browser_login, revoke_session, session_access_token
from ._credentials import (
    StoredCredentials,
    delete_credentials,
    load_credentials,
    save_credentials,
)
from ._session import delete_session, load_metadata
from ._transport import resolve_base_url
from .client import Modaic
from .errors import ModaicAPIError, ModaicError
from .types import Model, TrainingControls


class CLIError(Exception):
    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _json_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    return value


def _emit(value: object, *, compact: bool) -> None:
    serialized = _json_value(value)
    if compact:
        print(json.dumps(serialized, separators=(",", ":"), ensure_ascii=False))
    else:
        print(json.dumps(serialized, indent=2, ensure_ascii=False))


def _success(message: str, *, compact: bool, **values: object) -> None:
    if compact:
        _emit({"ok": True, **values}, compact=True)
    else:
        print(message)


def _read_json(path: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CLIError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CLIError(f"Invalid JSON in {path}: {exc.msg}") from exc


def _read_object(path: str) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise CLIError(f"Expected a JSON object in {path}.")
    return cast(dict[str, Any], value)


def _read_examples(path: str) -> list[Mapping[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        try:
            values = [
                json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line
            ]
        except FileNotFoundError as exc:
            raise CLIError(f"File not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise CLIError(f"Invalid JSONL in {path} on line {exc.lineno}.") from exc
    else:
        raw = _read_json(path)
        if isinstance(raw, dict) and isinstance(raw.get("examples"), list):
            values = raw["examples"]
        else:
            values = raw
    if not isinstance(values, list) or not all(isinstance(value, dict) for value in values):
        raise CLIError(
            "Examples file must contain a JSON array, an {examples: [...]} "
            "object, or JSONL objects."
        )
    return cast(list[Mapping[str, Any]], values)


def _split_model(value: str) -> tuple[str, str]:
    workspace, separator, slug = value.partition("/")
    if not separator or not workspace or not slug or "/" in slug:
        raise CLIError("Model must use the workspace/model form.")
    return workspace, slug


def _client(args: argparse.Namespace) -> Modaic:
    stored = load_credentials()
    base_url = args.api_url or os.getenv("MODAIC_API_URL") or (stored.api_url if stored else None)
    resolved_url = resolve_base_url(base_url)
    api_key = os.getenv("MODAIC_API_KEY") or (stored.api_key if stored else None)
    if not api_key:
        api_key = session_access_token(
            profile=os.getenv("MODAIC_PROFILE", "default"), api_url=resolved_url
        )
    if not api_key:
        raise CLIError("Not logged in. Run `modaic login` or set MODAIC_API_KEY.")
    return Modaic(api_key=api_key, base_url=resolved_url)


def _model(client: Modaic, value: str) -> Model[Any, Any, Any]:
    workspace, slug = _split_model(value)
    return client.models.get(workspace=workspace, model=slug)


def _login(args: argparse.Namespace) -> None:
    if not args.token_stdin:
        args.action = "login"
        _auth(args)
        return
    token = sys.stdin.readline().strip()
    if not token:
        raise CLIError("A Modaic token is required.")
    base_url = args.api_url or os.getenv("MODAIC_API_URL")
    with Modaic(api_key=token, base_url=base_url) as client:
        user = client.identity.me()
    resolved_url = resolve_base_url(base_url)
    save_credentials(StoredCredentials(api_key=token, api_url=resolved_url))
    identity = user.email or (user.profile.slug if user.profile else user.id)
    _success(
        f"Logged in as {identity}.", compact=args.json, user=_json_value(user), apiUrl=resolved_url
    )


def _logout(args: argparse.Namespace) -> None:
    if not args.api_key:
        args.action = "logout"
        _auth(args)
        return
    removed = delete_credentials()
    suffix = (
        " MODAIC_API_KEY remains set and will still authenticate commands."
        if os.getenv("MODAIC_API_KEY")
        else ""
    )
    message = ("Logged out." if removed else "No stored Modaic login was found.") + suffix
    _success(
        message,
        compact=args.json,
        removed=removed,
        environmentCredential=bool(os.getenv("MODAIC_API_KEY")),
    )


def _auth(args: argparse.Namespace) -> None:
    profile = args.profile or os.getenv("MODAIC_PROFILE", "default")
    if args.action == "login":
        base_url = resolve_base_url(args.api_url)
        identity = browser_login(
            api_url=base_url,
            profile=profile,
            storage=args.storage,
            headless=args.headless or not sys.stdin.isatty(),
            timeout=args.timeout,
        )
        label = identity.email or identity.subject
        _success(
            f"Logged in as {label} for {identity.api_url}.",
            compact=args.json,
            account=label,
            apiUrl=identity.api_url,
            profile=profile,
            expiresAt=identity.expires_at,
        )
        return

    if args.action == "status":
        if os.getenv("MODAIC_API_KEY"):
            _success(
                "Authenticated with MODAIC_API_KEY.",
                compact=args.json,
                source="environment",
                apiUrl=resolve_base_url(args.api_url),
            )
            return
        legacy = load_credentials()
        if legacy is not None:
            _success(
                "Authenticated with a stored API key.",
                compact=args.json,
                source="stored-api-key",
                apiUrl=resolve_base_url(args.api_url, legacy.api_url),
            )
            return
        metadata = load_metadata(profile)
        if metadata is None:
            raise CLIError("Not logged in. Run `modaic login`.")
        _success(
            f"Stored login for {metadata.email or metadata.subject} ({metadata.api_url}).",
            compact=args.json,
            source="login",
            account=metadata.email or metadata.subject,
            apiUrl=metadata.api_url,
            profile=profile,
            expiresAt=metadata.expires_at,
            expired=metadata.expires_at <= int(time.time()),
            storage=metadata.storage,
        )
        return

    metadata = load_metadata(profile)
    remote_error: ValueError | None = None
    if metadata is not None and not args.local_only:
        try:
            revoke_session(profile=profile, api_url=metadata.api_url)
        except ValueError as exc:
            remote_error = exc
    removed = delete_session(profile)
    suffix = (
        " MODAIC_API_KEY remains set and will still authenticate commands."
        if os.getenv("MODAIC_API_KEY")
        else ""
    )
    _success(
        ("Logged out locally." if removed else "No stored Modaic login was found.") + suffix,
        compact=args.json,
        removed=removed,
        profile=profile,
        environmentCredential=bool(os.getenv("MODAIC_API_KEY")),
    )
    if remote_error is not None:
        raise CLIError(
            "Logged out locally, but remote session revocation was not confirmed.",
            exit_code=3,
        )


def _models(args: argparse.Namespace) -> None:
    with _client(args) as client:
        if args.action == "list":
            _emit(client.models.list(), compact=args.json)
            return
        if args.action == "get":
            _emit(_model(client, args.model_path), compact=args.json)
            return
        if args.action == "create":
            configuration = _read_object(args.config) if args.config else {}
            base_model = args.base_model or configuration.get("model")
            if not base_model:
                raise CLIError("Provide --model or a config containing model.")
            kwargs: dict[str, Any] = {"workspace": args.workspace, "slug": args.slug}
            for target, source in (
                ("description", args.description),
                ("default_branch", args.default_branch),
                ("model", base_model),
                ("questions", configuration.get("questions")),
            ):
                if source is not None:
                    kwargs[target] = source
            _emit(client.models.create(**kwargs), compact=args.json)
            return
        current = _model(client, args.model_path)
        if args.action == "update":
            configuration = _read_object(args.config) if args.config else {}
            kwargs = {"expected_head_sha": current.commit.commit_sha} if current.commit else {}
            for target, source in (
                ("branch", args.branch),
                ("description", args.description),
                ("model", configuration.get("model")),
                ("questions", configuration.get("questions")),
                ("message", args.message),
                ("discard_alignment", True if args.discard_alignment else None),
            ):
                if source is not None:
                    kwargs[target] = source
            if not kwargs or set(kwargs) == {"expected_head_sha"}:
                raise CLIError("Provide --config, --description, or another update option.")
            _emit(client.models.update(args.model_path, **kwargs), compact=args.json)
            return
        if not args.yes:
            if not sys.stdin.isatty():
                raise CLIError(
                    "Refusing to prompt on a non-interactive terminal; pass --yes to delete."
                )
            entered = input(f"Type {args.model_path} to permanently delete it: ").strip()
            if entered != args.model_path:
                raise CLIError("Deletion cancelled; model name did not match.")
        client.models.delete(args.model_path)
        _success(f"Deleted {args.model_path}.", compact=args.json, model=args.model_path)


def _examples(args: argparse.Namespace) -> None:
    with _client(args) as client:
        result: object
        if args.action == "create":
            result = client.examples.ingest(args.model_path, examples=_read_examples(args.file))
        elif args.action == "list":
            result = client.examples.list(args.model_path, page=args.page, page_size=args.page_size)
        elif args.action == "summary":
            result = client.examples.summary(args.model_path)
        elif args.action == "get":
            result = client.examples.get(args.example_id)
        elif args.action == "decisions":
            result = client.examples.list_decisions(args.example_id)
        else:
            annotation = _read_object(args.file)
            ground_truth = annotation.get("ground_truth", annotation.get("groundTruth"))
            ground_reasoning = annotation.get("ground_reasoning", annotation.get("groundReasoning"))
            kwargs: dict[str, Any] = {}
            if ground_truth is not None:
                if not isinstance(ground_truth, dict):
                    raise CLIError("ground_truth must be a JSON object.")
                kwargs["ground_truth"] = ground_truth
            if ground_reasoning is not None:
                if not isinstance(ground_reasoning, str):
                    raise CLIError("ground_reasoning must be a string.")
                kwargs["ground_reasoning"] = ground_reasoning
            if not kwargs:
                raise CLIError("Annotation must contain ground_truth and/or ground_reasoning.")
            result = client.examples.annotate(args.example_id, **kwargs)
        _emit(result, compact=args.json)


def _alignments(args: argparse.Namespace) -> None:
    with _client(args) as client:
        if args.action in {"create", "list"}:
            model = _model(client, args.model_path)
            if args.action == "list":
                _emit(client.alignments.list(args.model_path, limit=args.limit), compact=args.json)
                return
            source = args.source_commit_sha or (model.commit.commit_sha if model.commit else None)
            if not source:
                raise CLIError("No source commit is available; pass --source-commit-sha.")
            kwargs: dict[str, Any] = {
                "branch": args.branch,
                "source_commit_sha": source,
                "idempotency_key": args.idempotency_key or str(uuid4()),
                "seed": args.seed,
            }
            for name in ("max_metric_calls", "reflection_model", "reflection_minibatch_size"):
                value = getattr(args, name)
                if value is not None:
                    kwargs[name] = value
            _emit(client.alignments.create(args.model_path, **kwargs), compact=args.json)
            return
        if args.action == "get":
            _emit(client.alignments.get(args.alignment_id), compact=args.json)
        elif args.action == "logs":
            _emit(client.alignments.logs(args.alignment_id), compact=args.json)
        else:
            client.alignments.cancel(args.alignment_id)
            _success("Alignment cancelled.", compact=args.json, alignmentId=args.alignment_id)


def _upload_progress(quiet: bool) -> Callable[[int, int], None] | None:
    if quiet or not sys.stderr.isatty():
        return None

    def report(uploaded: int, total: int) -> None:
        percent = 100 if total == 0 else int(uploaded * 100 / total)
        end = "\n" if uploaded == total else ""
        print(f"\rUploading: {percent:3d}%", end=end, file=sys.stderr, flush=True)

    return report


def _resources(args: argparse.Namespace) -> None:
    with _client(args) as client:
        if args.action == "upload":
            result = client.knowledge_base.uploads.upload_file(
                args.workspace,
                args.path,
                kind=args.kind,
                name=args.name,
                description=args.description,
                repos=args.repo,
                timeout=args.timeout,
                progress=_upload_progress(args.quiet),
            )
            _emit(result, compact=args.json)
        elif args.action == "list":
            _emit(
                client.knowledge_base.resources.list(
                    workspace=args.workspace,
                    kind=args.kind,
                    glob=args.glob,
                    commit_sha=args.commit_sha,
                    cursor=args.cursor,
                ),
                compact=args.json,
            )
        elif args.action == "get":
            _emit(
                client.knowledge_base.resources.get(
                    args.workspace, args.name, commit_sha=args.commit_sha
                ),
                compact=args.json,
            )
        elif args.action == "download":
            content = client.knowledge_base.resources.download(
                args.workspace, args.name, commit_sha=args.commit_sha
            )
            destination = Path(args.output or args.name)
            destination.write_bytes(content)
            _success(
                f"Downloaded {args.name} to {destination}.",
                compact=args.json,
                name=args.name,
                output=str(destination),
                sizeBytes=len(content),
            )
        else:
            _emit(
                client.knowledge_base.resources.update(
                    args.workspace,
                    args.name,
                    description=args.description if args.description is not None else ...,
                    repos=args.repo,
                ),
                compact=args.json,
            )


_CONTROL_FIELDS = (
    "max_seq_length",
    "n_epochs",
    "n_checkpoints",
    "n_evals",
    "batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "warmup_ratio",
    "max_grad_norm",
    "weight_decay",
    "random_seed",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "early_stopping_warmup_evals",
)


def _training_controls(args: argparse.Namespace) -> TrainingControls | None:
    controls: dict[str, Any] = {}
    for field in _CONTROL_FIELDS:
        value = getattr(args, field)
        if value is not None:
            if field == "batch_size" and isinstance(value, str) and value != "max":
                try:
                    value = int(value)
                except ValueError as exc:
                    raise CLIError("--batch-size must be an integer or 'max'.") from exc
            controls[field] = value
    if args.early_stopping:
        controls["early_stopping_enabled"] = True
    scheduler = args.lr_scheduler_type
    if scheduler:
        scheduler_args = {
            key: value
            for key, value in (("min_lr_ratio", args.min_lr_ratio), ("num_cycles", args.num_cycles))
            if value is not None
        }
        controls["lr_scheduler"] = {
            "lr_scheduler_type": scheduler,
            **({"lr_scheduler_args": scheduler_args} if scheduler_args else {}),
        }
    return cast(TrainingControls, controls) if controls else None


def _fine_tunes(args: argparse.Namespace) -> None:
    with _client(args) as client:
        if args.action == "create":
            training_file = args.training_file
            if args.training_path:
                if not args.workspace or not args.training_name:
                    raise CLIError(
                        "--training-path requires --workspace and --training-name."
                    )
                uploaded = client.knowledge_base.uploads.upload_file(
                    args.workspace,
                    args.training_path,
                    kind="dataset",
                    name=args.training_name,
                    timeout=args.timeout,
                    progress=_upload_progress(args.quiet),
                )
                if not uploaded.result:
                    detail = uploaded.error.message if uploaded.error else uploaded.status
                    raise CLIError(f"Training dataset upload did not complete: {detail}")
                training_file = uploaded.result.name
            kwargs: dict[str, Any] = {
                "model": args.base_model,
                "training_file": training_file,
                "task_id": args.task_id,
                "training_type": args.training_type,
                "idempotency_key": args.idempotency_key,
                "controls": _training_controls(args),
            }
            if args.questions:
                kwargs["questions"] = _read_object(args.questions)
            for name in (
                "destination",
                "branch",
                "source_commit_sha",
                "validation_file",
                "suffix",
                "from_checkpoint",
                "lora_r",
                "lora_alpha",
                "lora_dropout",
                "wandb_api_key",
                "wandb_base_url",
                "wandb_project_name",
                "wandb_name",
                "wandb_entity",
            ):
                value = getattr(args, name)
                if value is not None:
                    kwargs[name] = value
            result = client.fine_tunes.create(**kwargs)
            if args.wait:
                result = client.fine_tunes.wait(result.id, timeout=args.timeout)
            _emit(result, compact=args.json)
        elif args.action == "list":
            _emit(
                client.fine_tunes.list(
                    workspace=args.workspace, model_id=args.model_id, limit=args.limit
                ),
                compact=args.json,
            )
        elif args.action == "get":
            _emit(client.fine_tunes.get(args.job_id), compact=args.json)
        elif args.action == "logs":
            _emit(client.fine_tunes.logs(args.job_id), compact=args.json)
        elif args.action == "wait":
            _emit(client.fine_tunes.wait(args.job_id, timeout=args.timeout), compact=args.json)
        else:
            client.fine_tunes.cancel(args.job_id)
            _success("Fine-tune cancelled.", compact=args.json, jobId=args.job_id)


def _protege(args: argparse.Namespace) -> None:
    with _client(args) as client:
        if args.action == "create":
            result = client.protege.create(
                prompt=args.prompt,
                repo=args.repo,
                eval_only_dataset=args.eval_only_dataset,
            )
            if args.wait:
                result = client.protege.wait(result.id, timeout=args.timeout)
            _emit(result, compact=args.json)
        elif args.action == "get":
            _emit(client.protege.get(args.job_id), compact=args.json)
        elif args.action == "wait":
            _emit(client.protege.wait(args.job_id, timeout=args.timeout), compact=args.json)
        else:
            _emit(client.protege.cancel(args.job_id), compact=args.json)


def _add_model_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model_path", metavar="WORKSPACE/MODEL")


def _add_login_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default="default", help="Login profile (default: default).")
    parser.add_argument(
        "--headless",
        "--no-browser",
        action="store_true",
        help="Print a URL without opening a browser.",
    )
    parser.add_argument(
        "--storage",
        choices=("keyring", "file"),
        default="keyring",
        help="Session storage: OS keyring (default) or private plaintext file (POSIX only).",
    )
    parser.add_argument(
        "--timeout", type=float, default=600, help="Login timeout in seconds (default: 600)."
    )


def _add_logout_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default="default", help="Login profile (default: default).")
    parser.add_argument(
        "--local-only", action="store_true", help="Remove local session without remote revocation."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modaic", description="Modaic command-line client")
    parser.add_argument("--api-url", help="Override the Modaic API base URL.")
    parser.add_argument("--json", action="store_true", help="Emit compact machine-readable JSON.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    commands = parser.add_subparsers(dest="command", required=True)

    login = commands.add_parser("login", help="Sign in through your browser.")
    login.add_argument("--token-stdin", action="store_true", help="Read the token from stdin.")
    _add_login_options(login)
    logout = commands.add_parser("logout", help="Revoke and remove your CLI session.")
    _add_logout_options(logout)
    logout.add_argument("--api-key", action="store_true", help="Remove a legacy stored API key.")

    auth = commands.add_parser("auth", help="Authenticate the Modaic CLI.").add_subparsers(
        dest="action", required=True
    )
    auth_login = auth.add_parser("login", help="Sign in through your browser.")
    _add_login_options(auth_login)
    auth_status = auth.add_parser("status", help="Show the effective authentication source.")
    auth_status.add_argument("--profile", default="default")
    auth_logout = auth.add_parser("logout", help="Remove one stored CLI session.")
    _add_logout_options(auth_logout)

    models = commands.add_parser("models", help="Manage model repositories.").add_subparsers(
        dest="action", required=True
    )
    models.add_parser("list")
    model_get = models.add_parser("get")
    _add_model_argument(model_get)
    model_create = models.add_parser("create")
    model_create.add_argument("workspace")
    model_create.add_argument("slug")
    model_create.add_argument("--config", help="Path to model.json.")
    model_create.add_argument("--model", dest="base_model")
    model_create.add_argument("--description")
    model_create.add_argument("--default-branch")
    model_update = models.add_parser("update")
    _add_model_argument(model_update)
    model_update.add_argument("--config", help="Path to model.json.")
    model_update.add_argument("--description")
    model_update.add_argument("--branch")
    model_update.add_argument("--message")
    model_update.add_argument("--discard-alignment", action="store_true")
    model_delete = models.add_parser("delete")
    _add_model_argument(model_delete)
    model_delete.add_argument("--yes", action="store_true")

    examples = commands.add_parser("examples", help="Manage model examples.").add_subparsers(
        dest="action", required=True
    )
    example_create = examples.add_parser("create")
    _add_model_argument(example_create)
    example_create.add_argument("--file", required=True)
    example_list = examples.add_parser("list")
    _add_model_argument(example_list)
    example_list.add_argument("--page", type=int, default=1)
    example_list.add_argument("--page-size", type=int, default=30)
    example_summary = examples.add_parser("summary")
    _add_model_argument(example_summary)
    for action in ("get", "annotate", "decisions"):
        example = examples.add_parser(action)
        _add_model_argument(example)
        example.add_argument("example_id")
        if action == "annotate":
            example.add_argument("--file", required=True)

    alignments = commands.add_parser(
        "alignments", help="Run and inspect alignments."
    ).add_subparsers(dest="action", required=True)
    alignment_create = alignments.add_parser("create")
    _add_model_argument(alignment_create)
    alignment_create.add_argument("--branch", default="main")
    alignment_create.add_argument("--source-commit-sha")
    alignment_create.add_argument("--max-metric-calls", type=int)
    alignment_create.add_argument("--idempotency-key")
    alignment_create.add_argument("--reflection-model")
    alignment_create.add_argument("--reflection-minibatch-size", type=int)
    alignment_create.add_argument("--seed", type=int, default=0)
    alignment_list = alignments.add_parser("list")
    _add_model_argument(alignment_list)
    alignment_list.add_argument("--limit", type=int, default=20)
    for action in ("get", "logs", "cancel"):
        alignment = alignments.add_parser(action)
        alignment.add_argument("alignment_id")

    resources = commands.add_parser(
        "resources", help="Upload and retrieve knowledge-base resources."
    ).add_subparsers(dest="action", required=True)
    upload = resources.add_parser("upload")
    upload.add_argument("workspace")
    upload.add_argument("path")
    upload.add_argument(
        "--kind", required=True, choices=("dataset", "eval-only-dataset", "artifact")
    )
    upload.add_argument("--name", required=True)
    upload.add_argument("--description")
    upload.add_argument("--repo", action="append", default=[])
    upload.add_argument("--timeout", type=float)
    resource_list = resources.add_parser("list")
    resource_list.add_argument("workspace")
    resource_list.add_argument("--kind", choices=("dataset", "eval-only-dataset", "artifact"))
    resource_list.add_argument("--glob")
    resource_list.add_argument("--commit-sha")
    resource_list.add_argument("--cursor")
    for action in ("get", "download", "update"):
        resource = resources.add_parser(action)
        resource.add_argument("workspace")
        resource.add_argument("name")
        resource.add_argument("--commit-sha")
        if action == "download":
            resource.add_argument("--output")
        if action == "update":
            resource.add_argument("--description")
            resource.add_argument("--repo", action="append")

    fine_tunes = commands.add_parser(
        "fine-tunes", help="Run and inspect fine-tuning jobs."
    ).add_subparsers(dest="action", required=True)
    fine_create = fine_tunes.add_parser("create")
    fine_create.add_argument("--model", dest="base_model", required=True)
    training_source = fine_create.add_mutually_exclusive_group(required=True)
    training_source.add_argument("--training-file")
    training_source.add_argument("--training-path")
    fine_create.add_argument("--training-name")
    fine_create.add_argument("--workspace")
    fine_create.add_argument("--task-id", required=True)
    fine_create.add_argument(
        "--questions",
        help="JSON object of training-only questions when task-id is absent from model.json.",
    )
    fine_create.add_argument("--training-type", choices=("head_only", "lora"), default="head_only")
    fine_create.add_argument("--destination")
    fine_create.add_argument("--branch")
    fine_create.add_argument("--source-commit-sha")
    fine_create.add_argument("--validation-file")
    fine_create.add_argument("--suffix")
    fine_create.add_argument("--from-checkpoint")
    fine_create.add_argument("--idempotency-key")
    fine_create.add_argument("--lora-r", type=int)
    fine_create.add_argument("--lora-alpha", type=float)
    fine_create.add_argument("--lora-dropout", type=float)
    fine_create.add_argument("--wandb-api-key")
    fine_create.add_argument("--wandb-base-url")
    fine_create.add_argument("--wandb-project-name")
    fine_create.add_argument("--wandb-name")
    fine_create.add_argument("--wandb-entity")
    fine_create.add_argument("--max-seq-length", type=int)
    fine_create.add_argument("--n-epochs", type=int)
    fine_create.add_argument("--n-checkpoints", type=int)
    fine_create.add_argument("--n-evals", type=int)
    fine_create.add_argument("--batch-size")
    fine_create.add_argument("--gradient-accumulation-steps", type=int)
    fine_create.add_argument("--learning-rate", type=float)
    fine_create.add_argument("--lr-scheduler-type", choices=("constant", "linear", "cosine"))
    fine_create.add_argument("--min-lr-ratio", type=float)
    fine_create.add_argument("--num-cycles", type=float)
    fine_create.add_argument("--warmup-ratio", type=float)
    fine_create.add_argument("--max-grad-norm", type=float)
    fine_create.add_argument("--weight-decay", type=float)
    fine_create.add_argument("--random-seed", type=int)
    fine_create.add_argument("--early-stopping", action="store_true")
    fine_create.add_argument("--early-stopping-patience", type=int)
    fine_create.add_argument("--early-stopping-min-delta", type=float)
    fine_create.add_argument("--early-stopping-warmup-evals", type=int)
    fine_create.add_argument("--wait", action="store_true")
    fine_create.add_argument("--timeout", type=float)
    fine_list = fine_tunes.add_parser("list")
    fine_list.add_argument("workspace")
    fine_list.add_argument("--model-id")
    fine_list.add_argument("--limit", type=int, default=20)
    for action in ("get", "logs", "cancel"):
        fine = fine_tunes.add_parser(action)
        fine.add_argument("job_id")
    fine_wait = fine_tunes.add_parser("wait")
    fine_wait.add_argument("job_id")
    fine_wait.add_argument("--timeout", type=float)

    protege = commands.add_parser("protege", help="Run Protégé optimization jobs.").add_subparsers(
        dest="action", required=True
    )
    protege_create = protege.add_parser("create")
    protege_create.add_argument("repo", metavar="WORKSPACE/MODEL")
    protege_create.add_argument("--prompt", required=True)
    protege_create.add_argument("--eval-only-dataset")
    protege_create.add_argument("--wait", action="store_true")
    protege_create.add_argument("--timeout", type=float, default=3600.0)
    for action in ("get", "cancel"):
        protege_job = protege.add_parser(action)
        protege_job.add_argument("job_id")
    protege_wait = protege.add_parser("wait")
    protege_wait.add_argument("job_id")
    protege_wait.add_argument("--timeout", type=float, default=3600.0)
    return parser


_HANDLERS: dict[str, Callable[[argparse.Namespace], None]] = {
    "auth": _auth,
    "login": _login,
    "logout": _logout,
    "models": _models,
    "examples": _examples,
    "alignments": _alignments,
    "resources": _resources,
    "fine-tunes": _fine_tunes,
    "protege": _protege,
}


def _fail(message: str, *, json_output: bool, exit_code: int = 1) -> NoReturn:
    if json_output:
        print(json.dumps({"error": {"message": message}}, separators=(",", ":")), file=sys.stderr)
    else:
        print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _HANDLERS[args.command](args)
    except ModaicAPIError as exc:
        detail = str(exc)
        if exc.code:
            detail = f"{detail} ({exc.code})"
        _fail(detail, json_output=args.json)
    except CLIError as exc:
        _fail(str(exc), json_output=args.json, exit_code=exc.exit_code)
    except TimeoutError as exc:
        _fail(str(exc), json_output=args.json, exit_code=4)
    except (ModaicError, ValueError, OSError) as exc:
        _fail(str(exc), json_output=args.json)
    except KeyboardInterrupt:
        _fail("Interrupted.", json_output=args.json, exit_code=130)
    return 0
