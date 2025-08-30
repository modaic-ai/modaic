from .base import ContextSchema
from typing import Callable, List


class TextSchema(ContextSchema):
    """
    Schema for Text context.
    """

    text: str

    def chunk_text(
        self, chunk_fn: Callable[[str], List[str | tuple[str, dict]]], **kwargs
    ):
        def chunk_text_fn(text_context: "TextSchema") -> List["TextSchema"]:
            return chunk_fn(text_context.text)

    @classmethod
    def from_file(cls, file: str, *args, **kwargs):
        """
        Load a LongText instance from a file.
        """
        with open(file, "r") as f:
            text = f.read()
        return cls(text=text, *args, **kwargs)
