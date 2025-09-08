from modaic.context import TableFile
from modaic.databases import MilvusBackend, VectorDatabase
from modaic.indexing import DummyEmbedder
from modaic.storage import InPlaceFileStore
import os
fs = InPlaceFileStore("examples/TableRAG/dev_excel")
embedder = DummyEmbedder()
vdb = VectorDatabase(MilvusBackend.from_local("index.db"), embedder=embedder, payload_class=TableFile)

# modaic_schema = TableFile.schema()
# print("modaic_schema", modaic_schema)
vdb.create_collection("table_rag", TableFile, exists_behavior="replace")

# print("number of files", len(fs))


def records_generator():
    for ref in os.listdir("examples/TableRAG/dev_excel"):
        table = TableFile.from_file_store(ref, fs)
        yield table


# # for table in records_generator():
# #     for field in table.__class__.model_fields:
# #         print(field)
# #     print("model_dump", table.model_dump(include=["id"]))
# #     break
vdb.add_records("table_rag", records_generator(), batch_size=2, tqdm_total=len(fs))


# class SpecialContext(Context):
#     a: int = Field(default=1, hidden=True)


# c = Context()
# for field in c.__class__.model_fields:
#     print(field)
#     if extra := getattr(field, "json_schema_extra", None):
#         hidden = extra.get("hidden", False)
#         print("hidden", hidden)
