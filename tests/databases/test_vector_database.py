from typing import ClassVar, Optional

import dspy
import pytest

from modaic.databases.vector_database.vector_database import VectorDatabase


def test_custom_vectordb():
    class MyConfig(VectorDatabaseConfig):
        _module: ClassVar[str] = "modaic.databases.integrations.milvus"
        uri: str = "test.db"
        user: str = ""
        password: str = ""
        db_name: str = ""
        token: str = ""
        timeout: Optional[float] = None
        kwargs: dict = {}

    class MyNewDB(VectorDatabase):
        def __init__(self, config: MyConfig, **kwargs):
            super().__init__(config, **kwargs)

    x = MyNewDB(MyConfig(), embedder=dspy.Embedder(model="text-embedding-3-small"))
    assert x.config._module == "modaic.databases.integrations.milvus"
    assert x.config.uri == "test.db"

    with pytest.raises(AssertionError):

        class MyBadConfig(VectorDatabaseConfig):
            uri: str = "http://localhost:19530"
