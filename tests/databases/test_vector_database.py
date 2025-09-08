from modaic.context import Text
from modaic.databases import VectorDatabase

from ..testing_utils import DummyBackend


def test_vector_database():
    backend = DummyBackend()
    vdb = VectorDatabase(backend)
    vdb.create_collection("test", Text, exists_behavior="replace")
    vdb.add_records("test", [Text(text="test")])
    results = vdb.search("test", "test")
    assert len(results) == 1
    assert results[0][0].id == "1"
    assert results[0][0].context.text == "test"
