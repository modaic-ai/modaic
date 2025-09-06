from modaic.databases import VectorDatabase, MilvusBackend, IndexConfig
from modaic.context import TableFile
from modaic.indexing import DummyEmbedder
from modaic.storage import InPlaceFileStore
import os
from pymilvus import DataType
from typing import List, Dict, Any
import numpy as np
# embedder = DummyEmbedder()
# index_config = IndexConfig(embedder=embedder)
# file_store = InPlaceFileStore("examples/TableRAG/dev_excel")
# try:
#     backend = MilvusBackend.from_local("test_db/index.db")
#     backend.create_collection("table_rag", TableFile.as_schema(), index_config)
#     file_ref = next(file_store.keys())
#     t = TableFile.from_file_store(file_ref, file_store)
#     print(t)
#     embedding = embedder(t.embedme())
#     record = backend.create_record({"vector": embedding}, t)
#     print("record", record)
#     backend.add_records("table_rag", [record])


# except Exception as e:
#     raise e
# finally:
#     os.remove("test_db/index.db")


def milvus_add_record(backend: MilvusBackend, embedding: np.ndarray, t: TableFile):
    record = backend.create_record({"vector": embedding}, t)
    print("record", record)
    backend.add_records("table_rag", [record])


def modaic_add_record(backend: MilvusBackend, embedding: np.ndarray, t: TableFile):
    pass


max_length = 65_535
max_length = 100
embedder = DummyEmbedder()
file_store = InPlaceFileStore("examples/TableRAG/dev_excel")
try:
    backend = MilvusBackend.from_local("test_db/index.db")
    milvus_schema = backend._client.create_schema(auto_id=False, enable_dynamic_field=True)
    milvus_schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=max_length)
    milvus_schema.add_field(field_name="parent", datatype=DataType.VARCHAR, max_length=max_length, nullable=True)
    milvus_schema.add_field(field_name="metadata", datatype=DataType.JSON)
    milvus_schema.add_field(field_name="file_ref", datatype=DataType.VARCHAR, max_length=max_length)
    milvus_schema.add_field(field_name="file_type", datatype=DataType.VARCHAR, max_length=max_length)
    milvus_schema.add_field(field_name="sheet_name", datatype=DataType.VARCHAR, max_length=max_length, is_nullable=True)
    milvus_schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=embedder.embedding_dim)

    # index_params = backend._client.prepare_index_params()
    # index_params.add_index(
    #     field_name="vector",
    #     index_name="vector_index",
    #     index_type="AUTOINDEX",
    #     metric_type="COSINE",
    # )

    backend._client.create_collection("table_rag", schema=milvus_schema)

    file_ref = next(file_store.keys())
    t = TableFile.from_file_store(file_ref, file_store)
    print(t)
    embedding = embedder(t.embedme())
    milvus_add_record(backend, embedding, t)


except Exception as e:
    raise e
finally:
    os.remove("test_db/index.db")
