from __future__ import annotations

import io
from pathlib import Path

import pytest

from modaic.storage.file_store import (
    FileResult,
    InPlaceFileStore,
    LocalFileStore,
    is_in_dir,
)


@pytest.fixture()
def test_dir_path() -> Path:
    """
    Absolute path to the artifacts directory used for `InPlaceFileStore` tests.

    Returns:
        Path to `tests/artifacts/test_dir`.
    """

    return Path(__file__).resolve().parents[1] / "artifacts" / "test_dir_txt"


@pytest.fixture()
def inplace_store(test_dir_path: Path) -> InPlaceFileStore:
    """
    Create an `InPlaceFileStore` pointing at the static test artifacts directory.

    Params:
        test_dir_path: The directory containing static test files.

    Returns:
        Configured `InPlaceFileStore` instance.
    """

    return InPlaceFileStore(directory=test_dir_path)


@pytest.fixture()
def local_store_dir(tmp_path: Path) -> Path:
    """
    Provide a fresh temporary directory for `LocalFileStore` per test.

    Params:
        tmp_path: Pytest-provided temporary path unique to the test.

    Returns:
        The created temporary directory path.
    """

    return tmp_path


@pytest.fixture()
def local_store(local_store_dir: Path) -> LocalFileStore:
    """
    Create a `LocalFileStore` rooted at a temporary directory.

    Params:
        local_store_dir: The temporary directory to use as the store root.

    Returns:
        Configured `LocalFileStore` instance.
    """

    return LocalFileStore(directory=local_store_dir)


def _write_text_file(path: Path, text: str) -> None:
    """
    Create or overwrite a text file with the provided content.

    Params:
        path: Destination file path.
        text: File content to write.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_inplace_store_initialization_and_contains(test_dir_path: Path) -> None:
    """
    `InPlaceFileStore` initializes on an existing directory and reports membership correctly.

    Params:
        test_dir_path: Directory containing known files.
    """

    store = InPlaceFileStore(directory=test_dir_path)
    # Pick a known filename from artifacts directory
    assert store.contains("file1.txt") is True
    assert store.contains("__missing__.txt") is False


def test_inplace_store_get_and_typing(inplace_store: InPlaceFileStore, test_dir_path: Path) -> None:
    """
    `get` returns a `FileResult` with correct fields for files in place.

    Params:
        inplace_store: Store to fetch from.
        test_dir_path: Directory to enumerate sample file from.
    """

    result = inplace_store.get("file1.txt")
    assert isinstance(result, FileResult)
    assert result.name == "file1.txt"
    assert result.type == "txt"


def test_inplace_store_keys_values_items_and_len(inplace_store: InPlaceFileStore, test_dir_path: Path) -> None:
    """
    `keys`, `values`, `items`, and `__len__` reflect the files present in the directory.

    Params:
        inplace_store: Store to inspect.
        test_dir_path: Directory that is the ground truth for file listing.
    """

    expected_files = sorted([p.name for p in test_dir_path.iterdir() if p.is_file()])
    keys_list = sorted(list(inplace_store.keys()))
    assert keys_list == expected_files
    # values
    values_list = list(inplace_store.values())
    assert {v.name for v in values_list} == set(expected_files)
    # items
    items_list = list(inplace_store.items())
    assert {k for k, _ in items_list} == set(expected_files)
    assert len(inplace_store) == len(expected_files)


def test_local_store_add_inside_uses_existing_path(local_store: LocalFileStore, local_store_dir: Path) -> None:
    """
    Adding a file already inside the store keeps it in place and returns its relative path.

    Params:
        local_store: Store instance under test.
        local_store_dir: Root directory for the store.
    """

    file_path = local_store_dir / "sample.txt"
    content = "hello from inside"
    _write_text_file(file_path, content)

    ref = local_store.add(file_path)
    assert ref == "sample.txt"
    got = local_store.get(ref)
    assert Path(got.file) == file_path
    assert Path(got.file).read_text(encoding="utf-8") == content


def test_local_store_add_outside_copies_into_modaic(
    local_store: LocalFileStore, tmp_path_factory: pytest.TempPathFactory, local_store_dir: Path
) -> None:
    """
    Adding a file from outside the store copies it into `.modaic/` and returns the new relative path.

    Params:
        local_store: Store instance under test.
        tmp_path_factory: Factory to create a separate external temp directory.
        local_store_dir: Root directory for the store.
    """

    outside_dir = tmp_path_factory.mktemp("outside-src")
    src_path = outside_dir / "external.txt"
    content = "external content"
    _write_text_file(src_path, content)

    ref = local_store.add(src_path)
    assert ref.startswith(".modaic/")
    stored = local_store.get(ref)
    assert Path(stored.file).exists()
    assert Path(stored.file).read_text(encoding="utf-8") == content
    # `.modaic` directory exists and file resides under it
    assert Path(local_store_dir / ref).parent.name == ".modaic"


def test_local_store_add_from_io(local_store: LocalFileStore) -> None:
    """
    Adding from a file-like object writes it into `.modaic/`.

    Params:
        local_store: Store instance under test.
    """

    buffer = io.BytesIO(b"streamed bytes")
    ref = local_store.add(buffer)
    assert ref.startswith(".modaic/")
    stored = local_store.get(ref)
    assert Path(stored.file).read_bytes() == b"streamed bytes"


def test_local_store_update_and_remove(
    local_store: LocalFileStore, local_store_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """
    `update` replaces file contents by copying from another path; `remove` deletes the file.

    Params:
        local_store: Store instance under test.
        local_store_dir: Root directory for the store.
        tmp_path_factory: Factory to create an external update source path.
    """

    # Seed a file inside the store so `add` returns the direct relative path
    original_path = local_store_dir / "target.txt"
    _write_text_file(original_path, "original")
    ref = local_store.add(original_path)
    assert ref == "target.txt"
    assert local_store.contains(ref) is True

    # Prepare a new source outside the store and update
    outside_dir = tmp_path_factory.mktemp("update-src")
    new_src = outside_dir / "new.txt"
    _write_text_file(new_src, "updated")
    local_store.update(ref, new_src)

    # Verify update took effect
    got = local_store.get(ref)
    assert Path(got.file).read_text(encoding="utf-8") == "updated"

    # Remove and verify deletion
    local_store.remove(ref)
    assert local_store.contains(ref) is False
    assert not (local_store_dir / ref).exists()


def test_local_store_keys_and_values_ignore_directories(local_store: LocalFileStore, local_store_dir: Path) -> None:
    """
    `keys` lists only files at the top level; internal directories like `.modaic` are ignored.

    Params:
        local_store: Store instance under test.
        local_store_dir: Root directory for the store.
    """

    # Create two files at the root and ensure `.modaic` exists
    _write_text_file(local_store_dir / "a.txt", "A")
    _write_text_file(local_store_dir / "b.txt", "B")
    (local_store_dir / ".modaic").mkdir(exist_ok=True)

    listed = set(local_store.keys())
    assert listed == {"a.txt", "b.txt"}
    values_names = {v.name for v in local_store.values()}
    assert values_names == {"a.txt", "b.txt"}


def test_is_in_dir_true_and_false(tmp_path: Path) -> None:
    """
    `is_in_dir` returns True for paths under a directory and False otherwise.

    Params:
        tmp_path: Temporary directory used to create inside and outside paths.
    """

    base = tmp_path / "base"
    base.mkdir()
    inside = base / "nested" / "file.txt"
    _write_text_file(inside, "x")
    outside = tmp_path / "outside.txt"
    _write_text_file(outside, "y")

    assert is_in_dir(inside, base) is True
    assert is_in_dir(base, inside.parent) is False
    assert is_in_dir(outside, base) is False
