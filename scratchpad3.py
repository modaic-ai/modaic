from modaic.databases import VectorDatabase, MilvusBackend
from modaic.storage import InPlaceFileStore
from modaic.context import TableFile
from modaic.indexing import DummyEmbedder

fs = InPlaceFileStore("examples/TableRAG/dev_excel")
embedder = DummyEmbedder()
vdb = VectorDatabase(MilvusBackend.from_local("index.db"), embedder=embedder)
vdb.create_collection("table_rag", TableFile.as_schema(), exists_behavior="replace")


def records_generator():
    for ref in fs.keys():
        table = TableFile.from_file_store(ref, fs)
        yield table


vdb.add_records("table_rag", records_generator(), batch_size=10000)
