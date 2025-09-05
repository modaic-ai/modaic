from .base import Context, HydratedAttr, requires_hydration
from typing import Callable, List, Iterable, Iterator, IO, Literal
from modaic.storage.file_store import FileStore
from pathlib import Path


class Text(Context):
    """
    Text context class.
    """

    text: str

    def chunk_text(
        self,
        chunk_fn: Callable[[str], Iterable[str | tuple[str, dict]]],
        kwargs: dict = None,
    ):
        def chunk_text_fn(text_context: "Text") -> Iterator["Text"]:
            for chunk in chunk_fn(text_context.text, **(kwargs or {})):
                yield Text(text=chunk)

        self.chunk_with(chunk_text_fn)

    @classmethod
    def from_file(
        cls, file: str | Path | IO, type: Literal["txt"] = "txt", params: dict = None
    ):
        """
        Load a LongText instance from a file.
        """
        if isinstance(file, (str, Path)):
            file = Path(file)
            text = file.read_text()
        elif isinstance(file, IO):
            text = file.read()
        return cls(text=text, **(params or {}))


class TextDocument(Context):
    """
    Text document context class.
    """

    _text: str = HydratedAttr()
    file_ref: str
    file_type: Literal["txt"] = "txt"

    def hydrate(self, file_store: FileStore) -> None:
        file = file_store.get(self.file_ref)
        if isinstance(file, Path):
            file = file.read_text()
        else:
            file = file.read()
        self._text = file

    @classmethod
    def from_file(
        cls,
        file: str,
        file_store: FileStore,
        file_type: Literal["txt"] = "txt",
        params: dict = None,
    ):
        """
        Load a TextDocument instance from a file.

        Args:
            file: The file to load.
            file_store: The file store to use.
            type: The type of file to load.
            params: The parameters to pass to the constructor.
        """
        file = file_store.get(file)
        instance = cls(file_ref=file, **(params or {}))
        instance.hydrate(file_store)
        return instance

    @requires_hydration
    def dump(self) -> None:
        return self._text

    @requires_hydration
    def chunk_text(
        self,
        chunk_fn: Callable[[str], List[str | tuple[str, dict]]],
        kwargs: dict = None,
    ):
        def chunk_text_fn(text_context: "TextDocument") -> List["Text"]:
            chunks = []
            for chunk in chunk_fn(text_context._text, **(kwargs or {})):
                chunks.append(Text(text=chunk))
            return chunks

        self.apply_to_chunks(chunk_text_fn)
