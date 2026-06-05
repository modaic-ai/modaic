"""Pure-git regression tests for hub.git_snapshot's branch-reload strategy.

These tests do not touch the Modaic hub (no MODAIC_TOKEN required). They lock in
the decision to hard-reset cached branch worktrees to ``origin/<branch>`` instead
of running ``git pull``. ``git pull`` fails with
"fatal: Need to specify how to reconcile divergent branches." when the cached
checkout has diverged from origin (observed in production); a hard reset recovers
cleanly because our hub checkouts are always expected to already be in sync.
"""

from pathlib import Path

import git
import pytest


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
