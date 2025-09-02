from abc import ABC, abstractmethod
from typing import (
    List,
    Optional,
    IO,
    Dict,
    Tuple,
    Iterable,
    Iterator,
    Literal,
    TYPE_CHECKING,
    Union,
    Any,
)
from pathlib import Path
import json
import shutil
import uuid
import os
import lmdb
import msgpack
from contextlib import contextmanager

if TYPE_CHECKING:
    from ..context import Context


class FileStore(ABC):
    @abstractmethod
    def get(
        self, context_id: str, select: Optional[List[str]] = None
    ) -> Optional[Path | IO | Dict[str, Path | IO | None]]:
        """
        Fetches local files for a given context id.

        Args:
            context_id: The id of the context to fetch files for.
            select: Optional list of file names to fetch. If None, all files are fetched.

        Raises:
            KeyError: If the context id is not found in the file store. Or the selected alias(s) are not found in the file store.

        Returns:
            A dictionary of aliases to file paths or file-like objects. If an alias is found but the file it refers to does not exist, the value is None.
        """
        if not self.contains(context_id):
            raise ValueError(f"Context id {context_id} not found in file store")

    def get_all(
        self, items: Iterable[str | Tuple[str, List[str]]]
    ) -> Iterator[Optional[Path | IO | Dict[str, Path | IO | None]]]:
        """
        Fetches all files from the file store.

        Args:
            items: An iterable of context ids and optional file names to fetch.

        Raises:
            KeyError: If any context id is not found in the file store.

        Returns:
            An iterator of dictionaries of aliases to file paths or file-like objects.
        """
        for context_id, select in items:
            yield self.get(context_id, select)

    @abstractmethod
    def insert(
        self, context_id: str, files: Dict[str, Path | IO], overwrite: bool = False
    ) -> None:
        """
        Adds files to the file store for a given context id with the given aliases.

        Args:
            context_id: The id of the context to add files for.
            files: A dictionary of aliases to file paths or file-like objects.

        Raises:
            ValueError: If the context id already exists in the file store and overwrite is False.

        """
        if self.contains(context_id) and not overwrite:
            raise ValueError(f"Context id {context_id} already exists in file store")

    def insert_all(
        self,
        items: Iterable[Tuple[str, Dict[str, Path | IO]]],
        overwrite: bool = False,
    ) -> None:
        """
        Adds multiple files to the file store for multiple context ids.

        Args:
            items: An iterable of context ids and dictionaries of aliases to file paths or file-like objects.
            overwrite: Whether to overwrite existing files in the file store. Defaults to False.

        Raises:
            ValueError: If any context id already exists in the file store and overwrite is False.
        """
        for context_id, files in items:
            self.insert(context_id, files, overwrite)

    @abstractmethod
    def update(self, context_id: str, files: Dict[str, Path | IO]) -> None:
        """
        Updates files in the file store for a given context id with the given aliases.

        Args:
            context_id: The id of the context to update files for.
            files: A dictionary of aliases to file paths or file-like objects.

        Raises:
            KeyError: If the context id is not found in the file store.
        """
        if not self.contains(context_id):
            raise ValueError(f"Context id {context_id} not found in file store")

    def update_all(
        self,
        items: Iterable[Tuple[str, Dict[str, Path | IO]]],
    ) -> None:
        """
        Updates multiple files in the file store for multiple context ids.
        """
        for context_id, files in items:
            self.update(context_id, files)

    def upsert(self, context_id: str, files: Dict[str, Path | IO]) -> None:
        """
        Upserts files into the file store for a given context id with the given aliases.

        Args:
            context_id: The id of the context to upsert files for.
            files: A dictionary of aliases to file paths or file-like objects.
        """
        if self.contains(context_id):
            self.update(context_id, files)
        else:
            self.insert(context_id, files)

    def upsert_all(
        self,
        items: Iterable[Tuple[str, Dict[str, Path | IO]]],
    ) -> None:
        """
        Upserts multiple files into the file store for multiple context ids.

        Args:
            items: An iterable of context ids and dictionaries of aliases to file paths or file-like objects.
        """
        for context_id, files in items:
            self.upsert(context_id, files)

    @abstractmethod
    def delete(self, context_id: str, select: Optional[List[str]] = None) -> None:
        """
        Deletes files from the file store for a given context id with the given aliases.
        Args:
            context_id: The id of the context to delete files for.
            file_names: Optional list of file names to delete. If None, all files are deleted and context_id is removed from the file store.

        Raises:
            KeyError: If the context id is not found in the file store.
        """
        if not self.contains(context_id):
            raise ValueError(f"Context id {context_id} not found in file store")

    def delete_all(
        self,
        items: Iterable[str | Tuple[str, List[str]]],
    ) -> None:
        """
        Deletes multiple files from the file store for multiple context ids.

        Args:
            items: An iterable of context ids and optional file names to delete.

        Raises:
            KeyError: If any context id is not found in the file store.
        """
        for context_id, select in items:
            self.delete(context_id, select)

    @abstractmethod
    def contains(self, context_id: str) -> bool:
        """
        Checks if the file store contains a given context id.

        Args:
            context_id: The id of the context to check.

        Returns:
            True if the context id is found in the file store, False otherwise.
        """
        pass

    def __contains__(self, context: Union["Context", str]) -> bool:
        return self.contains(context)


class InPlaceFileStore(FileStore):
    id_to_files: lmdb._Database
    file_to_id_count: lmdb._Database
    _read_txn: lmdb.Transaction
    _write_txn: lmdb.Transaction
    _files_to_remove: set[str]

    def __init__(
        self,
        directory: str | Path,
        mutable: bool = False,
        map_size: int = 64 * 1024**3,
    ):
        self.directory = Path(directory)
        self.mutable = mutable
        modaic_dir = self.directory / ".modaic"
        db_dir = modaic_dir / "metadata"
        db_dir.mkdir(parents=True, exist_ok=True)

        self.env: lmdb.Environment = lmdb.open(
            str(db_dir), map_size=map_size, max_dbs=2
        )
        with self.env.begin(write=True) as txn:
            self.id_to_files = self.env.open_db(b"id_to_files", txn=txn)
            self.file_to_id_count = self.env.open_db(b"file_to_id_count", txn=txn)

        self._read_txn = None
        self._write_txn = None
        self._files_to_remove = set()

    @contextmanager
    def begin(
        self,
        write: bool = False,
    ) -> Iterator[None]:
        """
        Context manager LMDB transaction. Opens a read-write transaction if write is True, otherwise a read-only transaction. If the requested transaction is already open it, for the filestore the existing transaction is used.

        Params:
            write: Whether to open a read-write transaction. Defaults to False.

        Returns:
            The file store instance with an active transaction assigned to self.txn.
        """
        if write and self._write_txn:
            yield None
        elif not write and (self._read_txn or self._write_txn):
            yield None
        else:
            # At this point we know that we need to open a new transaction, write=True: we open _write_txn, write=False: we open _read_txn
            with self.env.begin(
                write=write,
            ) as txn:
                if write:
                    self._write_txn = txn
                else:
                    self._read_txn = txn
                try:
                    yield None
                finally:
                    # Close whatever transaction we opened write=True: we opened a _write_txn,  write=False: we opened _read_txn
                    if write:
                        self._write_txn = None
                    else:
                        self._read_txn = None
                    # If they are both None, we are at the top level. This step removes any files in the _files_to_remove set.
                    if self._write_txn is None and self._read_txn is None:
                        self.cleanup()

    def commit(self, ignore_write: bool = False) -> None:
        """
        Commit the active LMDB transaction stored on this file store.

        Params:
            ignore_write: Whether to ignore any write transactions that are currently open, defaults to False.

        Returns:
            None
        """
        if self._write_txn is None and self._read_txn is None:
            raise ValueError("No active transaction to commit. Use begin() first.")

        if self._write_txn is not None and not ignore_write:
            self._write_txn.commit()
        if self._read_txn is not None:
            self._read_txn.commit()

    def abort(self, ignore_write: bool = False) -> None:
        """
        Abort the active LMDB transaction stored on this file store.

        Params:
            ignore_write: Whether to ignore any write transactions that are currently open, defaults to False.

        Returns:
            None
        """
        if self._write_txn is None and self._read_txn is None:
            raise ValueError("No active transaction to abort. Use begin() first.")
        if self._write_txn is not None and not ignore_write:
            self._write_txn.abort()
        if self._read_txn is not None:
            self._read_txn.abort()

    def end(self) -> None:
        """
        End the current transaction: commit if it is a write transaction, otherwise abort.

        Params:
            None

        Returns:
            None
        """
        if self._write_txn is None and self._read_txn is None:
            return
        try:
            if self._write_txn is not None:
                self._write_txn.commit()
            if self._read_txn is not None:
                self._read_txn.abort()
        finally:
            self._write_txn = None
            self._read_txn = None

    def get(
        self, context_id: str, select: Optional[List[str]] = None
    ) -> Dict[str, Path | IO]:
        super().get(context_id, select)
        return {
            k: v
            for k, v in self.id_to_files[context_id].items()
            if (k in select or select is None)
        }

    def insert(
        self,
        context_id: str,
        files: Dict[str, str | Path | IO],
        overwrite: bool = False,
    ) -> None:
        super().insert(context_id, files, overwrite)
        saved_files = {}
        for alias, file in files.items():
            if isinstance(file, (Path, str)):
                file = Path(file)
                if is_in_dir(file, self.directory):
                    saved_files[alias] = file
                else:
                    saved_files[alias] = shutil.copy(file, self.directory)
            else:
                filename = f"{uuid.uuid4().hex}.txt"
                with open(self.directory / ".modaic" / filename, "w") as f:
                    f.write(file.read())
                saved_files[alias] = filename

        self.id_to_files[context_id].update(saved_files)
        self.file_to_id.update({v: context_id for v in saved_files.values()})

    def update(self, context_id: str, files: Dict[str, str | Path | IO]) -> None:
        super().update(context_id, files)
        old_alias_map = self.id_to_files[context_id]
        new_alias_map = {}

        for alias, new_file in files.items():
            new_file = Path(new_file) if isinstance(new_file, str) else new_file
            if isinstance(new_file, Path):
                if is_in_dir(new_file, self.directory):
                    new_alias_map[alias] = new_file
                else:
                    new_alias_map[alias] = shutil.copy(new_file, self.directory)
            else:
                filename = f"{uuid.uuid4().hex}.txt"
                with open(self.directory / filename, "w") as f:
                    f.write(new_file.read())
                new_alias_map[alias] = filename

        self.metadata[context_id].update(new_alias_map)
        with open(self.directory / ".modaic" / "metadata.json", "w") as f:
            json.dump(self.metadata, f)

    def delete(self, context_id: str, remove_orphaned: bool = False) -> None:
        super().delete(context_id)
        old_alias_map = self.id_to_files[context_id]
        for alias in old_alias_map:
            self.unlink_file(context_id, alias, remove_orphaned=remove_orphaned)

    def contains(self, context_id: str) -> bool:
        return context_id in self.id_to_files

    def link_files(self, context_id: str, paths: Dict[str, Path]):
        """
        Links files in self.directory to a context id. Creates a new alias map if the context id does not exist. Updates the alias map if the context id already exists.

        Args:
            context_id: The id of the context to link files to.
            files: A dictionary of aliases to file paths.

        Raises:
            ValueError: If no active transaction is open.
            ValueError: If the file is not a Path object to a file in self.directory.
        """
        with self.begin(write=True):
            current_alias_map: dict[str, str] = (
                self._get(self.id_to_files, context_id) or {}
            )
            new_alias_map: dict[str, str] = {**current_alias_map}
            for alias, path in paths.items():
                # Will raise a value error if not a subpath
                local_path = str(path.relative_to(self.directory))
                new_alias_map[alias] = local_path
                id_count = self._get(self.file_to_id_count, local_path) or 0
                self._put(self.file_to_id_count, local_path, id_count + 1)

            self._put(self.id_to_files, context_id, new_alias_map)

    def unlink_file(
        self, context_id: str, alias: str = "default", remove_orphaned: bool = False
    ):
        with self.begin(write=True):
            if not self.contains(context_id):
                raise KeyError(f"context_id: {context_id} not found in file store")

            context_files: dict[str, str] = self._get(self.id_to_files, context_id)

            if alias not in context_files:
                raise KeyError(
                    f"file with alias: {alias} not found for context_id: {context_id}"
                )

            local_path: str = context_files.pop(alias)
            id_count = self._get(self.file_to_id_count, local_path)
            id_count -= 1
            if id_count == 0:
                self._del(self.file_to_id_count, local_path)
            else:
                self._put(self.file_to_id_count, local_path, id_count)

        if id_count == 0 and remove_orphaned and self.mutable:
            self._files_to_remove.add(local_path)

    def _get(self, db: lmdb._Database, key: str) -> Optional[Any]:
        with self.begin(write=False):
            v = self._read_txn.get(_b(key), db=db)
            return _unpack(v) if v is not None else None

    def _put(self, db: lmdb._Database, key: str, value: Any) -> None:
        with self.begin(write=True):
            self._write_txn.put(_b(key), _pack(value), db=db)

    def _del(self, db: lmdb._Database, key: str, value: Any = None) -> None:
        with self.begin(write=True):
            if value is None:
                self._write_txn.delete(_b(key), db=db)
            else:
                self._write_txn.delete(_b(key), value=_pack(value), db=db)

    def cleanup(self) -> None:
        for file in self._files_to_remove:
            os.remove(self.directory / file)
        self._files_to_remove.clear()


def is_in_dir(path: str | Path, directory: str | Path) -> bool:
    path = Path(path).resolve()  # follows symlinks
    directory = Path(directory).resolve()
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _b(s: str | bytes) -> bytes:
    return s if isinstance(s, bytes) else s.encode()


def _pack(obj) -> bytes:
    return msgpack.dumps(obj, use_bin_type=True)


def _unpack(b: bytes):
    return msgpack.loads(b, raw=False)
