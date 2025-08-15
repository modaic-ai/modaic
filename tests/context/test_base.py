import pytest
from typing import ClassVar, Type, List

from modaic.context.base import (
    ContextSchema,
    Atomic,
    Molecular,
    Context,
    Source,
    SourceType,
)


class MyAtomicSchema(ContextSchema):
    context_class: ClassVar[str] = "MyAtomic"
    value: int


class MyAtomic(Atomic):
    schema: ClassVar[Type[ContextSchema]] = MyAtomicSchema

    def __init__(
        self,
        value: int,
        id: str | None = None,
        source: Source | None = None,
        metadata: dict | None = None,
        **kwargs,
    ):
        super().__init__(source=source, metadata=metadata, **kwargs)
        self.value = value

    def embedme(self) -> str:
        return str(self.value)


class MyMolecularSchema(ContextSchema):
    context_class: ClassVar[str] = "MyMolecular"
    title: str


class MyMolecular(Molecular):
    schema: ClassVar[Type[ContextSchema]] = MyMolecularSchema

    def __init__(
        self,
        title: str,
        id: str | None = None,
        source: Source | None = None,
        metadata: dict | None = None,
        **kwargs,
    ):
        super().__init__(source=source, metadata=metadata, **kwargs)
        self.title = title

    def embedme(self) -> str:
        return self.title


def test_contextschema_metaclass_property_access():
    # Accessing declared fields on the schema class should yield Prop proxies
    prop = MyAtomicSchema.value
    from modaic.context.query_language import Prop

    assert isinstance(prop, Prop)
    with pytest.raises(AttributeError):
        _ = MyAtomicSchema.nonexistent


def test_atomic_extension_and_serialize_readme():
    src = Source(origin="/tmp/file.txt", type=SourceType.LOCAL_PATH, metadata={"a": 1})
    a = MyAtomic(value=42, source=src, metadata={"k": "v"})

    serialized = a.serialize()
    assert isinstance(serialized, MyAtomicSchema)
    assert serialized.value == 42
    assert isinstance(serialized.source, Source)
    assert serialized.metadata == {"k": "v"}

    # Default readme returns serialize()
    readme_obj = a.readme()
    assert isinstance(readme_obj, MyAtomicSchema)
    assert readme_obj.value == 42


def test_context_setters_and_metadata_updates():
    a = MyAtomic(value=1)
    # set_source without copy
    src = Source(origin="o", type=SourceType.URL, metadata={"m": 1})
    a.set_source(src)
    assert a.source is src

    # set_source with copy
    src2 = Source(origin="o2", type=SourceType.URL, metadata={"m": 2})
    a.set_source(src2, copy=True)
    assert a.source is not src2
    assert a.source.origin == "o2"

    # set_metadata
    a.set_metadata({"x": 1})
    assert a.metadata == {"x": 1}

    # add_metadata
    a.add_metadata({"y": 2})
    assert a.metadata == {"x": 1, "y": 2}


def test_molecular_chunk_with_sets_source_and_chunk_ids():
    parent = MyMolecular(
        title="Root",
        source=Source(
            origin="/root", type=SourceType.LOCAL_PATH, metadata={"chunk_id": 5}
        ),
    )

    def chunk_fn(ctx: Context) -> List[Context]:
        return [MyAtomic(value=i) for i in range(3)]

    parent.chunk_with(chunk_fn)
    assert len(parent.chunks) == 3

    for i, ch in enumerate(parent.chunks):
        # Source is set and parent weakref points back
        assert isinstance(ch.source, Source)
        assert ch.source.parent is parent
        # chunk_id was transformed from int to nested dict preserving previous id
        assert ch.source.metadata["chunk_id"] == {"id": 5, "chunk_id": i}
        # _parent should be excluded from dumps
        dumped = ch.source.model_dump()
        assert "_parent" not in dumped


def test_apply_to_chunks_mutation():
    m = MyMolecular(title="Root")

    def chunk_fn(ctx: Context) -> List[Context]:
        return [MyAtomic(value=0), MyAtomic(value=1)]

    m.chunk_with(chunk_fn)

    def bump(chunk: Context):
        assert isinstance(chunk, MyAtomic)
        chunk.value += 10

    m.apply_to_chunks(bump)
    assert [c.value for c in m.chunks] == [10, 11]


def test_update_chunk_id_variants():
    # no existing key
    md = {}
    Molecular.update_chunk_id(md, 7)
    assert md == {"chunk_id": 7}

    # existing int becomes dict
    md = {"chunk_id": 3}
    Molecular.update_chunk_id(md, 9)
    assert md == {"chunk_id": {"id": 3, "chunk_id": 9}}

    # existing dict nests deeper, preserving previous ids
    md = {"chunk_id": {"id": 1, "chunk_id": 2}}
    Molecular.update_chunk_id(md, 4)
    assert md == {"chunk_id": {"id": 1, "chunk_id": {"id": 2, "chunk_id": 4}}}


def test_deserialize_roundtrip_with_schema():
    src = Source(origin="/p", type=SourceType.LOCAL_PATH)
    a = MyAtomic(value=99, source=src, metadata={"z": True})
    schema = a.serialize()
    restored = MyAtomic.deserialize(schema)
    assert isinstance(restored, MyAtomic)
    assert restored.value == 99
    assert isinstance(restored.source, Source)
    assert restored.metadata == {"z": True}


@pytest.mark.xfail(
    reason="Context.from_dict depends on ContextSchema.from_dict which is not implemented"
)
def test_from_dict_not_implemented_yet():
    MyAtomic.from_dict({"value": 1, "source": {"origin": "o"}, "metadata": {}})


def test_str_and_repr_include_fields():
    a = MyAtomic(
        value=7,
        source=Source(origin="/a", type=SourceType.LOCAL_PATH),
        metadata={"k": "v"},
    )
    s = str(a)
    r = repr(a)
    assert "value=" in s and "source=" in s and "metadata=" in s
    assert r == s
