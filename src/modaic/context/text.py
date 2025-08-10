from .base import Molecular, Atomic, Source, SourceType, Context, SerializedContext
import requests
from typing import Callable, List, ClassVar, Type


class SerializedText(SerializedContext):
    context_class: ClassVar[str] = "Text"
    text: str


class Text(Atomic):
    schema: ClassVar[Type[SerializedContext]] = SerializedText

    def __init__(self, text: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text = text

    def embedme(self):
        return self.text

    def readme(self):
        return self.text


class SerializedLongText(SerializedContext):
    context_class: ClassVar[str] = "LongText"
    text: str
    chunks: List[SerializedText]


class LongText(Molecular):
    schema: ClassVar[Type[SerializedContext]] = SerializedLongText

    def __init__(self, text: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text = text
        self.chunks = []

    def chunk_text(
        self, chunk_fn: Callable[[str], List[str | tuple[str, dict]]]
    ) -> List[Text]:
        """
        Chunk the text into smaller Context objects.

        Args:
            chunk_fn: A function that takes in a string and returns a list of strings or string-metadata pairs.

        Returns:
            A list of Context objects.
        """

        # def context_chunk_fn(text: LongText) -> List[Text]:
        #     return [Text(text=text, source=self.source)]

        def context_chunk_fn(long_text: LongText) -> List[Text]:
            return [Text(text=t) for t in chunk_fn(long_text.text)]

        self.chunk_with(context_chunk_fn)

    def embedme(self):
        return self.text

    def readme(self):
        return self.text

    @classmethod
    def from_file(cls, file: str, *args, **kwargs):
        with open(file, "r") as f:
            text = f.read()
        return cls(text=text, *args, **kwargs)

    @classmethod
    def from_url(cls, url: str, *args, **kwargs):
        response = requests.get(url)
        return cls(text=response.text, *args, **kwargs)
