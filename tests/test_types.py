from typing import Optional

from modaic.context.base import Context
from modaic.types import Array, String


class Email(Context):
    subject: String[100]
    content: str
    recipients: Array[str, 10]
    tags: list[str]
    priority: int
    score: float
    pinned: bool
    optional_summary: Optional[String[50]]
    optional_recipients: Optional[Array[int, 5]]


def test_types() -> None:
    pass
