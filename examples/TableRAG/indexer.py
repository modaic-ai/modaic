from modaic.auto_agent import Indexer
from modaic.databases import VectorDatabase, MilvusVDBConfig
from typing import List, Literal
import dspy
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json
from modaic.databases import SQLDatabase, SQLiteConfig
from modaic.context import SerializedContext, Table, LongText, Text, Source, SourceType
from modaic.utils import PineconeReranker
from dotenv import load_dotenv
from tqdm import tqdm
import modaic

load_dotenv()


class TableRagIndexer(Indexer):
    def __init__(
        self, vdb_config: MilvusVDBConfig, sql_config: SQLiteConfig, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embedder = modaic.Embedder(model="openai/text-embedding-3-small")
        self.vector_database = VectorDatabase(
            config=vdb_config,
            embedder=self.embedder,
            payload_schema=Text.schema,
        )
        self.sql_db = SQLDatabase(config=sql_config)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        self.reranker = PineconeReranker(
            model="bge-reranker-v2-m3", api_key=os.getenv("PINECONE_API_KEY")
        )
        self.last_query = None

        self.vector_database.drop_collection("table_rag")
        self.vector_database.create_collection("table_rag", Text.schema)

    def ingest(self, files: List[str] | str, *args, **kwargs):
        if isinstance(files, str):
            files = [os.path.join(files, file) for file in os.listdir(files)]
        records = []
        with self.sql_db.connect_and_begin():
            for file in tqdm(files, desc="Ingesting files"):
                if file.endswith((".csv", ".xlsx", ".xls")):
                    if file.endswith(".csv"):
                        table = Table.from_csv(file)
                    elif file.endswith((".xlsx", ".xls")):
                        table = Table.from_excel(file)
                    # Add table to file system context store
                    table.metadata["schema"] = table.schema_info()
                    self.sql_db.add_table(table)
                    table.chunk_with(self.chunk_table)
                    records.extend(table.get_chunks())
                elif file.endswith((".json")):
                    with open(file, "r", encoding="utf-8") as f:
                        data_split = json.load(f)
                    key_value_doc = ""
                    for key, item in data_split.items():
                        key_value_doc += f"{key} {item}\n"
                    text_document = LongText(text=key_value_doc)
                    text_document.chunk_text(self.text_splitter.split_text)
                    text_document.apply_to_chunks(
                        lambda chunk: chunk.add_metadata({"type": "text"})
                    )
                    records.extend(text_document.get_chunks())
        print("Adding records to vector database")
        self.vector_database.add_records("table_rag", records)

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
    ) -> List[SerializedContext]:
        results = self.recall(user_query, k_recall, type)
        records = [
            result.text
            if result.context_class == "Text"
            else result.metadata["md_chunk"]
            for result in results
        ]
        results = self.reranker(user_query, records, k_rerank)
        return results

    def recall(
        self,
        user_query: str,
        k: int = 10,
        type: Literal["table", "text", "all"] = "all",
    ) -> List[SerializedContext]:
        embedding = self.embedder([user_query])[0]
        if type == "table":
            filter = {"metadata": {"type": "table"}}
        elif type == "text":
            filter = {"metadata": {"type": "text"}}
        else:
            filter = None
        return self.vector_database.search("table_rag", embedding, k, filter)

    def chunk_table(self, table: Table) -> List[Text]:
        table_md = LongText(text=table.markdown())
        table_md.chunk_text(self.text_splitter.split_text)
        table_md.apply_to_chunks(lambda chunk: chunk.add_metadata({"type": "table"}))
        return table_md.chunks

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


if __name__ == "__main__":
    indexer = TableRagIndexer(
        vdb_config=MilvusVDBConfig.from_local("index.db"),
        sql_config=SQLiteConfig(db_path="tables.db"),
    )
    excel_dir = "examples/TableRAG/dev_excel"
    excels = [os.path.join(excel_dir, file) for file in os.listdir(excel_dir)]
    excels = excels[:20]

    indexer.ingest(excels)
    x = indexer.recall("Who is the New Zealand Parliament Member for Canterbury")
    print(x)
