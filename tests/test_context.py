from modaic.context import Atomic, Molecular, serializable, Source, SourceType
import pytest


class AtomicContext(Atomic):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.visible = "hello"
        self._hidden = "you can't see me"

    @serializable
    def embedme(self):
        return "test: " + self._hidden

    @serializable
    def should_work(self, arg1=0, arg2=2):
        return arg1 + arg2

    def readme(self):
        return f"This is a test atomic context with a visible attribute: {self.visible}"


class CannotSerialize(Atomic):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.visible = "hello"

    @serializable
    def test_method(self, arg1, arg2):
        return "test"


class CanDeserialize(Molecular):
    def __init__(self, text: str, **kwargs):
        super().__init__(**kwargs)
        self.text = text
        self._chunks = []
        self._hidden_val = "you can't see me"

    @serializable
    def readme(self):
        return self.text

    @serializable
    def embedme(self):
        return self.text

    def complicated_func(self, arg1, arg2):
        return arg1 + arg2


class CustomInheritableContext(Atomic):
    def __init__(self, some_arg: int, some_other_arg: str, **kwargs):
        super().__init__(**kwargs)
        self.some_arg = some_arg
        self.some_other_arg = some_other_arg

    def readme(self):
        return (
            f"This is a test atomic context with a visible attribute: {self.some_arg}"
        )

    def embedme(self):
        return self.some_arg


class MolecularContext(Molecular):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.some_arg = 1

    def readme(self):
        return "hello"

    def embedme(self):
        return "hello"


def test_serialize():
    new_atomic = AtomicContext()
    serialized = new_atomic.serialize()
    assert serialized.context_class == "AtomicContext"
    assert serialized.visible == "hello"
    assert serialized.embedme == "test: you can't see me"
    assert serialized.should_work == 2
    assert serialized.source is None
    assert serialized.metadata == {}
    with pytest.raises(AttributeError):
        serialized._hidden
    with pytest.raises(AttributeError):
        serialized.readme
    with pytest.raises(TypeError):
        CannotSerialize().serialize()


def test_deserialize():
    serialized1 = CanDeserialize(
        text="hello",
        source=Source(
            type=SourceType.LOCAL_PATH, origin="test.txt", metadata={"chunk_id": 0}
        ),
        metadata={"md_chunk": "hello"},
    ).serialize()
    serialized2 = AtomicContext().serialize()
    assert serialized1.text == "hello"
    assert serialized1.source.origin == "test.txt"
    assert serialized1.source.type == SourceType.LOCAL_PATH
    assert serialized1.source.metadata == {"chunk_id": 0}
    assert serialized1.metadata == {"md_chunk": "hello"}
    assert serialized1.readme == "hello"
    with pytest.raises(AttributeError):
        serialized1._hidden_val
    with pytest.raises(AttributeError):
        serialized1._chunks
    with pytest.raises(AttributeError):
        serialized1.complicated_func
    with pytest.raises(ValueError):
        AtomicContext.deserialize(serialized2)

    deserialized = CanDeserialize.deserialize(serialized1)
    assert deserialized.text == "hello"
    assert deserialized._chunks == []
    assert deserialized._hidden_val == "you can't see me"
    assert deserialized.readme() == "hello"
    assert deserialized.complicated_func(1, 2) == 3


def test_custom_inheritable_context():
    custom_context = CustomInheritableContext(some_arg=1, some_other_arg="hello")
    assert custom_context.some_arg == 1
    assert custom_context.some_other_arg == "hello"
    assert (
        custom_context.readme()
        == "This is a test atomic context with a visible attribute: 1"
    )
    assert custom_context.embedme() == 1
    with pytest.raises(TypeError):
        CustomInheritableContext(some_arg=1, some_other_arg="hello", extra_arg=2)


class TestSource:
    @classmethod
    def setup_class(cls):
        cls.a = AtomicContext()

    def test_set_source(self):
        assert self.a.source is None
        msource = Source(type=SourceType.LOCAL_PATH, origin="test.txt")
        m = MolecularContext(source=msource)
        asource = Source(
            type=msource.type,
            origin=msource.origin,
            metadata={"chunk_id": 0},
            parent=m,
        )
        self.a.set_source(asource)
        assert self.a.source.parent is m
        assert m.source.parent is None
        assert self.a.source.origin == m.source.origin == "test.txt"
        assert m.source.type == m.source.type == SourceType.LOCAL_PATH
        assert m.source.metadata == {}
        assert self.a.source.metadata == {"chunk_id": 0}

    def test_parent_out_of_scope(self):
        assert self.a.source.type == SourceType.LOCAL_PATH
        assert self.a.source.origin == "test.txt"
        assert self.a.source.metadata == {"chunk_id": 0}
        assert self.a.source.parent is None

    def test_chunking(self):
        m = MolecularContext()
        m.chunk_with(
            lambda x: [MolecularContext(), MolecularContext(), MolecularContext()]
        )
        assert len(m.get_chunks()) == 3
        assert m.get_chunks()[0].source.metadata == {"chunk_id": 0}
        assert m.get_chunks()[1].source.metadata == {"chunk_id": 1}
        assert m.get_chunks()[2].source.metadata == {"chunk_id": 2}

        assert m.get_chunks()[0].source.parent is m
        assert m.get_chunks()[1].source.parent is m
        assert m.get_chunks()[2].source.parent is m
        mm = m.get_chunks()[0]
        mm.chunk_with(lambda x: [MolecularContext(), MolecularContext()])

        assert mm.get_chunks()[0].source.metadata == {
            "chunk_id": {"id": 0, "chunk_id": 0}
        }
        assert mm.get_chunks()[1].source.metadata == {
            "chunk_id": {"id": 0, "chunk_id": 1}
        }

        mm2 = m.get_chunks()[1]
        mm2.chunk_with(lambda x: [MolecularContext(), MolecularContext()])
        assert mm2.get_chunks()[0].source.metadata == {
            "chunk_id": {"id": 1, "chunk_id": 0}
        }
        assert mm2.get_chunks()[1].source.metadata == {
            "chunk_id": {"id": 1, "chunk_id": 1}
        }
