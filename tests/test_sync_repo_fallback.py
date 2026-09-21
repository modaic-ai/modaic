import shutil
from pathlib import Path
from modaic.hub import _sync_repo


def test_sync_repo_fallback_without_rsync(monkeypatch, tmp_path):
    """When rsync/robocopy are not available, _sync_repo must fall back to python copy."""
    # Simulate environment where rsync and robocopy do not exist
    monkeypatch.setattr(shutil, "which", lambda cmd: None)

    sync_dir = tmp_path / "sync"
    repo_dir = tmp_path / "repo"
    sync_dir.mkdir()
    repo_dir.mkdir()

    # Create dummy files and subdirs in sync_dir
    (sync_dir / "app.py").write_text("print('hello')", encoding="utf-8")
    sub = sync_dir / "sub"
    sub.mkdir()
    (sub / "util.py").write_text("x = 1", encoding="utf-8")

    # Create symlink inside sync_dir pointing to an external file
    ext_file = tmp_path / "external.txt"
    ext_file.write_text("external content", encoding="utf-8")
    (sync_dir / "linked.txt").symlink_to(ext_file)

    # In repo_dir, create a .git folder (must not be deleted) and an old file (should be deleted under mirror)
    git_dir = repo_dir / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")
    (repo_dir / "old_stale.txt").write_text("old", encoding="utf-8")

    _sync_repo(sync_dir, repo_dir, mirror=True)

    # Assert synchronized files
    assert (repo_dir / "app.py").read_text(encoding="utf-8") == "print('hello')"
    assert (repo_dir / "sub" / "util.py").read_text(encoding="utf-8") == "x = 1"
    # Symlink should be resolved/dereferenced into content
    assert (repo_dir / "linked.txt").read_text(encoding="utf-8") == "external content"
    assert not (repo_dir / "linked.txt").is_symlink()

    # .git should be preserved
    assert (git_dir / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main"

    # Stale file should be deleted under mirror
    assert not (repo_dir / "old_stale.txt").exists()
