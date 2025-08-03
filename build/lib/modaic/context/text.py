from .types import Molecular, Atomic, Source, SourceType, Context
import requests
from typing import Callable, List


class Text(Atomic):
    def __init__(self, text: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text = text

    def embedme(self):
        return self.text

    def readme(self):
        return self.text


class LongText(Molecular):
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

        Note:
            If the chunk_fn returns a Context object the source will not be automatically set. You must set it manually.
        """
        chunks = chunk_fn(self.text)
        for i, chunk in enumerate(chunks):
            source = Source(
                type=self.source.type,
                origin=self.source.origin,
                metadata={"chunk_id": i},
            )
            if isinstance(chunk, str):
                self.chunks.append(Text(text=chunk, source=source))
            elif isinstance(chunk, tuple):
                self.chunks.append(
                    Text(text=chunk[0], source=source, metadata=chunk[1])
                )
            elif isinstance(chunk, Context):
                self.chunks.append(chunk)
            else:
                raise ValueError(
                    f"chunk_fn returned an invalid chunk type: {type(chunk)}"
                )
        return self.chunks

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
