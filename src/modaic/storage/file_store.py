import shutil
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    IO,
    Iterator,
    NamedTuple,
    Optional,
    Tuple,
)

import immutables


class FileResult(NamedTuple):
    """
    Return type for FileStore.get()
    Args:
        file: The file object
        type: The type of the file without the leading dot (e.g. "csv", "txt", "xlsx", etc.). Usually the extension of the file but can be used for other purposes. (e.g. "google_doc")
        name: The name of the file
        metadata: A map of metadata for the file
    """

    file: Path | IO
    type: str
    name: Optional[str] = None
    metadata: immutables.Map = immutables.Map()


class FileStore(ABC):
    """
    Base class for file stores. File stores are used to store files that hydrate Context objects. The files can be stored locally or remotely.
    """

    @abstractmethod
    def get(self, reference: str) -> FileResult:
        """
        Get a file from the file store by reference (like an id or path).

        Args:
            reference: The id or path of the file in the FileStore.

        Raises:
            FileNotFoundError: If the file is not found.

        Returns:
            A Path object or a file-like object.
        """

    @abstractmethod
    def contains(self, reference: str) -> bool:
        """
        Checks if the file store contains a given reference.

        Args:
            reference: The id or path of the file in the FileStore.

        Returns:
            True if the reference is found in the file store, False otherwise.
        """
        pass

    @abstractmethod
    def keys(self, folder: Optional[str] = None) -> Iterator[str]:
        """
        Iterate over all keys in the file store.
        """
        pass

    def values(self, folder: Optional[str] = None) -> Iterator[FileResult]:
        """
        Iterate over all files in the file store.
        """
        for ref in self.keys(folder):
            yield self.get(ref)

    def items(self, folder: Optional[str] = None) -> Iterator[Tuple[str, FileResult]]:
        """
        Iterate over all keys and files in the file store.
        """
        for ref in self.keys(folder):
            yield ref, self.get(ref)

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over all references in the file store.
        """
        return self.keys()

    def __contains__(self, reference: str) -> bool:
        return self.contains(reference)

    def __len__(self) -> int:
        """
        Get the number of files in the file store.
        """
        return len(list(self.keys()))


class MutableFileStore(FileStore):
    """
    A FileStore that supports adding and removing files.
    """

    @abstractmethod
    def add(self, file: str | Path | IO, type: Optional[str] = None) -> str:
        """
        Add a file to the file store.

        Args:
            file: The file to add to the file store.

        Returns:
            The reference to the file in the FileStore.
        """

    @abstractmethod
    def update(self, reference: str, file: str | Path | IO, type: Optional[str] = None) -> None:
        """
        Update a file in the file store.

        Args:
            reference: The id or path of the file in the FileStore.
            file: The file to update in the file store.

        Raises:
            FileNotFoundError: If the file is not found.
        """
        pass

    @abstractmethod
    def remove(self, reference: str) -> None:
        """
        Remove a file from the file store.

        Raises:
            FileNotFoundError: If the file is not found.

        Args:
            reference: The id or path of the file in the FileStore.
        """


class InPlaceFileStore(FileStore):
    def __init__(
        self,
        directory: str | Path,
    ):
        self.directory = Path(directory)
        if not (self.directory).exists():
            raise FileNotFoundError(f"File store directory {self.directory} is not a valid directory")

    def get(self, reference: str) -> FileResult:
        """
        Get a file from the file store by path relative to self.directory.

        Args:
            reference: The local path of the file relative to self.directory.

        Raises:
            FileNotFoundError: If the file is not found.

        Returns:
            A Path object.

        """
        path = self.directory / reference
        if not path.exists():
            raise FileNotFoundError(f"File {reference} not found in {self.directory}")
        return FileResult(file=path, type=path.suffix.lstrip("."), name=path.name)

    def contains(self, reference: str) -> bool:
        return (self.directory / reference).exists()

    def keys(self, folder: Optional[str] = None) -> Iterator[str]:
        """
        Iterate over all files in the file store.
        """
        folder: Path = self.directory if folder is None else self.directory / folder
        return (str(path.relative_to(self.directory)) for path in folder.iterdir() if path.is_file())


class LocalFileStore(MutableFileStore):
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        (self.directory / ".modaic").mkdir(parents=True, exist_ok=True)

    def get(self, reference: str) -> FileResult:
        """
        Get a file from the file store by path relative to self.directory.
        """
        path = self.directory / reference
        if not path.exists():
            raise FileNotFoundError(f"File {reference} not found in {self.directory}")
        return FileResult(file=path, type=path.suffix.lstrip("."), name=path.name)

    def contains(self, reference: str) -> bool:
        return (self.directory / reference).exists()

    def add(self, file: str | Path | IO) -> str:
        default_filename = str(uuid.uuid4())
        if isinstance(file, (Path, str)):
            original_path = Path(file)
            if is_in_dir(original_path, self.directory):
                path = original_path
            else:
                path = self.directory / ".modaic" / default_filename
                shutil.copy2(original_path, path)
        else:
            path = self.directory / ".modaic" / default_filename
            with open(path, "wb") as f:
                shutil.copyfileobj(file, f)
        return str(path.relative_to(self.directory))

    def update(self, reference: str, file: str | Path | IO) -> None:
        if isinstance(file, (Path, str)):
            original_path = Path(file)
            shutil.copy2(original_path, self.directory / reference)
        else:
            shutil.copyfileobj(file, self.directory / reference)

    def remove(self, reference: str) -> None:
        (self.directory / reference).unlink()

    def keys(self, folder: Optional[str] = None) -> Iterator[str]:
        base: Path = self.directory if folder is None else self.directory / folder
        return (str(path.relative_to(self.directory)) for path in base.iterdir() if path.is_file())


def is_in_dir(path: str | Path, directory: str | Path) -> bool:
    path = Path(path).resolve()  # follows symlinks
    directory = Path(directory).resolve()
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False
