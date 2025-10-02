import json
import os
from typing import Iterator, List, Literal, Optional

from agent.config import TableRAGConfig
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm  # auto picks the right frontend

from modaic import Indexer
from modaic.context import Context, Table, TableFile, Text
from modaic.databases import SearchResult, SQLDatabase, VectorDatabase
from modaic.indexing import Embedder, PineconeReranker
from modaic.storage import FileStore

load_dotenv()


class TableFileChunk(TableFile):
    content: str

    def embedme(self) -> str:
        return self.content


class TableRAGIndexer(Indexer):
    config: TableRAGConfig

    def __init__(
        self,
        *args,
        vector_db: VectorDatabase,
        file_store: FileStore,
        sql_db: SQLDatabase,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.embedder = Embedder(model="openai/text-embedding-3-small")
        self.vector_database = vector_db
        self.file_store = file_store
        self.sql_db = sql_db
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        self.reranker = PineconeReranker(model="bge-reranker-v2-m3", api_key=os.getenv("PINECONE_API_KEY"))
        self.last_query = None
        if self.vector_database.has_collection("table_rag"):
            self.vector_database.load_collection("table_rag", Text.schema(), embedder=self.embedder)
        else:
            self.vector_database.create_collection(
                "table_rag", Text.schema(), exists_behavior="replace", embedder=self.embedder
            )

        self.vector_database.create_collection("table_rag", Text.schema(), exists_behavior="append")

    def _ingest_files(self, files: Optional[List[str]] = None):
        records = []
        if files is None:
            files = self.file_store.iter_files()
        elif isinstance(files, str):
            files_results = self.file_store.values(files)

        with self.sql_db.begin():
            for file_result in tqdm(files_results, desc="Ingesting files", position=0):
                file = file_result.file
                file_type = file_result.file_type
                if file_type == "xlsx":
                    table = TableFile.from_file(file, file_store=self.file_store, file_type=file_type)
                    table.metadata["schema"] = table.schema_info()
                    table.chunk_with(self.chunk_table)
                    records.extend(table.chunks)
                elif file.endswith((".json")):
                    with open(file, "r", encoding="utf-8") as f:
                        data_split = json.load(f)
                    key_value_doc = ""
                    for key, item in data_split.items():
                        key_value_doc += f"{key} {item}\n"
                    text_document = Text(text=key_value_doc)
                    text_document.chunk_text(self.text_splitter.split_text)
                    text_document.apply_to_chunks(lambda chunk: chunk.add_metadata({"type": "text"}))
                    records.extend(text_document.chunks)
        self.vector_database.add_records("table_rag", records, batch_size=10000)

    def add():
        pass

    def delete():
        pass

    def retrieve(
        self,
        user_query: str,
        k_recall: int = 10,
        k_rerank: int = 10,
        type: Literal["table", "text", "all"] = "all",
    ) -> List[Context]:
        results = self.recall(user_query, k_recall, type)
        records = [
            (result["context_schema"].text, result["context_schema"])
            if result["context_schema"].context_class == "Text"
            else (
                result["context_schema"].metadata["md_chunk"],
                result["context_schema"],
            )
            for result in results
        ]

        results = self.reranker(user_query, records, k_rerank)
        results = [result[1] for result in results]
        return results

    def recall(
        self,
        user_query: str,
        k: int = 10,
        type: Literal["table", "text", "all"] = "all",
    ) -> List[SearchResult]:
        embedding = self.embedder([user_query])[0]
        if type == "table":
            filter = Text.metadata["type"] == "table"
        elif type == "text":
            filter = Text.metadata["type"] == "text"
        else:
            filter = None
        return self.vector_database.search("table_rag", embedding, k, filter)

    def chunk_table(self, table: TableFile) -> Iterator[TableFileChunk]:
        for text_chunk in self.text_splitter.split_text(table.markdown()):
            table_chunk = TableFileChunk(
                file_ref=table.file_ref,
                file_type=table.file_type,
                sheet_name=table.sheet_name,
                content=text_chunk,
                metadata={"type": "table", "schema": table.metadata["schema"]},
            )
            table_chunk._df = table._df
            yield table_chunk

    def sql_query(self, query: str) -> str:
        """
        Query the sql database and get the result as a string.
        Args:
            query: The sql query to execute.
        Returns:
            The result of the sql query as a string.
        """
        self.last_query = query
        try:
            return str(self.sql_db.fetchall(query))
        except Exception as e:
            return f"Error executing sql query: {e}"

    def get_table(self, table_id: str) -> Table:
        return self.sql_db.get_table(table_id)
