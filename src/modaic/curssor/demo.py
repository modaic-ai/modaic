from __future__ import annotations

from typing import List

from modaic.context.text import TextSchema, LongTextSchema
from modaic.context.base import Source

from .adapter import contextschema_to_node, relationship_between


def build_demo_objects() -> List[object]:
    """Construct sample nodes and a relationship from ContextSchema instances.

    Returns:
        A list with [parent_node, child_node, relationship]
    """
    src = Source()
    lt = LongTextSchema(
        source=src,
        metadata={},
        text="Hello world",
        chunks=[
            TextSchema(source=src, metadata={}, text="Hello"),
            TextSchema(source=src, metadata={}, text="world"),
        ],
    )

    parent_node = contextschema_to_node(lt)
    child_node = contextschema_to_node(lt.chunks[0])

    start, rel, end = relationship_between(lt, lt.chunks[0], rel_type="HAS_CHUNK", properties={"order": 0})

    assert start.__class__ is parent_node.__class__
    assert end.__class__ is child_node.__class__

    return [parent_node, child_node, rel]


def main() -> None:
    objs = build_demo_objects()
    for obj in objs:
        print(type(obj).__name__, str(obj))


if __name__ == "__main__":
    main()

