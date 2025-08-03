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
