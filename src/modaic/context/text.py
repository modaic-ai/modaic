from .base import Context
from typing import Callable, List


class Text(Context):
    """
    Text context class.
    """

    text: str

    def chunk_text(
        self,
        chunk_fn: Callable[[str], List[str | tuple[str, dict]]],
        kwargs: dict = None,
    ):
        def chunk_text_fn(text_context: "Text") -> List["Text"]:
            chunks = []
            for chunk in chunk_fn(text_context.text, **(kwargs or {})):
                chunks.append(Text(text=chunk))
            return chunks

        self.apply_to_chunks(chunk_text_fn)

    @classmethod
    def from_file(cls, file: str, params: dict = None):
        """
        Load a LongText instance from a file.
        """
        with open(file, "r") as f:
            text = f.read()
        return cls(text=text, **(params or {}))
