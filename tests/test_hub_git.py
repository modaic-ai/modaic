"""Pure-git regression tests for hub.git_snapshot.

These tests do not touch the Modaic hub (no MODAIC_TOKEN required). They lock in
the decision to hard-reset cached branch worktrees to ``origin/<branch>`` instead
of running ``git pull``. ``git pull`` fails with
"fatal: Need to specify how to reconcile divergent branches." when the cached
checkout has diverged from origin (observed in production); a hard reset recovers
cleanly because our hub checkouts are always expected to already be in sync. They
also verify that repeated and concurrent snapshots do not leak Git helpers or
race while creating the shared cache.
"""

import gc
import multiprocessing
from multiprocessing.queues import Queue
from pathlib import Path

import git
import psutil
import pytest
from modaic import hub


def _commit_file(repo: git.Repo, path: Path, content: str, message: str) -> None:
    path.write_text(content)
    repo.git.add("-A")
    repo.git.commit("-m", message)


def _init_repo(path: Path, bare: bool = False) -> git.Repo:
    repo = git.Repo.init(path, bare=bare)
    if not bare:
        repo.git.config("user.email", "test@modaic.dev")
        repo.git.config("user.name", "modaic-test")
    return repo


def _seed_remote(tmp_path: Path) -> Path:
    bare_path = tmp_path / "remote.git"
    bare = _init_repo(bare_path, bare=True)
    seed_path = tmp_path / "seed"
    seed = _init_repo(seed_path)
    try:
        _commit_file(seed, seed_path / "program.py", "VALUE = 1\n", "init")
        seed.create_remote("origin", str(bare_path))
        seed.remotes.origin.push("HEAD:refs/heads/main")
    finally:
        seed.close()
        bare.close()
    return bare_path


def _git_cat_file_children() -> set[int]:
    helpers: set[int] = set()
    for child in psutil.Process().children(recursive=True):
        try:
            if "git cat-file" in " ".join(child.cmdline()):
                helpers.add(child.pid)
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return helpers


def _snapshot_in_process(cache_path: str, remote_path: str, result_queue: Queue) -> None:
    hub.settings.modaic_cache = cache_path
    hub._make_git_url = lambda _repo_path, _access_token: remote_path
    try:
        snapshot_path, commit = hub.git_snapshot("owner/program", access_token="test-token")
        result_queue.put((str(snapshot_path), commit.sha, None))
    except Exception as exc:  # pragma: no cover - surfaced to the parent assertion
        result_queue.put((None, None, repr(exc)))


def test_hard_reset_recovers_where_pull_fails_on_divergence(tmp_path: Path):
    """reset --hard origin/<branch> recovers a diverged checkout; git pull does not."""
    # Bare "remote" seeded with an initial main commit.
    bare = tmp_path / "remote.git"
    _init_repo(bare, bare=True)

    seed = _init_repo(tmp_path / "seed")
    _commit_file(seed, tmp_path / "seed" / "a.txt", "v1", "init")
    seed.create_remote("origin", str(bare))
    seed.remotes.origin.push("HEAD:refs/heads/main")

    # The cached checkout git_snapshot maintains: a clone tracking origin/main.
    cache = git.Repo.clone_from(str(bare), tmp_path / "cache", multi_options=["--branch", "main"])
    cache.git.config("user.email", "test@modaic.dev")
    cache.git.config("user.name", "modaic-test")

    # Simulate divergence: cache gains a local-only commit while origin/main is
    # advanced to an unrelated commit. (Should never happen in our workflow, but did
    # in production per the Sentry report.)
    _commit_file(cache, tmp_path / "cache" / "a.txt", "local-only", "local divergence")

    other = git.Repo.clone_from(str(bare), tmp_path / "other")
    other.git.config("user.email", "test@modaic.dev")
    other.git.config("user.name", "modaic-test")
    _commit_file(other, tmp_path / "other" / "a.txt", "remote-v2", "remote update")
    other.remotes.origin.push("main")

    cache.remotes.origin.fetch()

    # What hub.py used to do: git pull. Force ff-only so the failure is deterministic
    # across git versions / user config (the production failure was the default
    # "divergent branches" variant of the same can't-reconcile condition).
    cache.git.config("pull.ff", "only")
    with pytest.raises(git.exc.GitCommandError):
        cache.remotes.origin.pull("main")

    # What hub.py does now: hard reset to origin recovers cleanly.
    cache.git.reset("--hard", "origin/main")
    assert (tmp_path / "cache" / "a.txt").read_text() == "remote-v2"
    assert cache.head.commit.hexsha == cache.commit("origin/main").hexsha


def test_hard_reset_is_noop_when_already_in_sync(tmp_path: Path):
    """When the checkout already matches origin, reset --hard is a clean no-op (no network)."""
    bare = tmp_path / "remote.git"
    _init_repo(bare, bare=True)

    seed = _init_repo(tmp_path / "seed")
    _commit_file(seed, tmp_path / "seed" / "a.txt", "v1", "init")
    seed.create_remote("origin", str(bare))
    seed.remotes.origin.push("HEAD:refs/heads/main")

    cache = git.Repo.clone_from(str(bare), tmp_path / "cache", multi_options=["--branch", "main"])
    before = cache.head.commit.hexsha

    cache.remotes.origin.fetch()
    cache.git.reset("--hard", "origin/main")

    assert cache.head.commit.hexsha == before
    assert (tmp_path / "cache" / "a.txt").read_text() == "v1"
    assert not cache.is_dirty()


def test_repeated_snapshots_close_git_helpers_without_gc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    remote = _seed_remote(tmp_path)
    cache_root = tmp_path / "cache"
    cache = cache_root / "modaic_hub/modaic_hub"
    monkeypatch.setattr(hub.settings, "modaic_cache", str(cache_root))
    monkeypatch.setattr(hub, "_make_git_url", lambda _repo_path, _access_token: str(remote))
    original_fetch = git.Remote.fetch
    fetch_count = 0

    def counting_fetch(remote_obj: git.Remote, *args, **kwargs):
        nonlocal fetch_count
        fetch_count += 1
        return original_fetch(remote_obj, *args, **kwargs)

    monkeypatch.setattr(git.Remote, "fetch", counting_fetch)

    gc.collect()
    helpers_before = _git_cat_file_children()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(50):
            snapshot_path, commit = hub.git_snapshot("owner/program", access_token="test-token")
            assert snapshot_path == cache / "owner/program/main"
            assert commit.sha
    finally:
        if gc_was_enabled:
            gc.enable()

    assert _git_cat_file_children() == helpers_before
    assert fetch_count == 50


def test_concurrent_processes_share_one_valid_snapshot(tmp_path: Path):
    remote = _seed_remote(tmp_path)
    cache_root = tmp_path / "cache"
    cache = cache_root / "modaic_hub/modaic_hub"
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(target=_snapshot_in_process, args=(str(cache_root), str(remote), result_queue)) for _ in range(4)
    ]

    for process in processes:
        process.start()
    results = [result_queue.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    errors = [error for _path, _sha, error in results if error]
    assert errors == []
    assert len({path for path, _sha, _error in results}) == 1
    assert len({sha for _path, sha, _error in results}) == 1

    snapshot = git.Repo(cache / "owner/program/main")
    try:
        assert snapshot.head.commit.hexsha == results[0][1]
        assert (cache / "owner/program/main/program.py").read_text() == "VALUE = 1\n"
    finally:
        snapshot.close()


def test_snapshot_failure_releases_lock_outside_cleaned_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = _seed_remote(tmp_path)
    cache_root = tmp_path / "cache"
    cache = cache_root / "modaic_hub/modaic_hub"
    monkeypatch.setattr(hub.settings, "modaic_cache", str(cache_root))
    monkeypatch.setattr(hub, "_make_git_url", lambda _repo_path, _access_token: str(remote))

    original_clone = git.Repo.clone_from
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("synthetic clone failure")
        return original_clone(*args, **kwargs)

    monkeypatch.setattr(git.Repo, "clone_from", fail_once)

    with pytest.raises(RuntimeError, match="synthetic clone failure"):
        hub.git_snapshot("owner/program", access_token="test-token")

    snapshot_path, commit = hub.git_snapshot("owner/program", access_token="test-token")
    assert snapshot_path.exists()
    assert commit.sha
    assert len(list((cache / ".locks").glob("*.lock"))) == 1
