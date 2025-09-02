from modaic import Indexer
from modaic.databases import VectorDatabase, MilvusVDBConfig, SearchResult
from typing import List, Literal
import dspy
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json
from modaic.databases import SQLDatabase, SQLiteConfig
from modaic.context import (
    Context,
    Table,
    Text,
)
from modaic.indexing import PineconeReranker, Embedder
from dotenv import load_dotenv
from tqdm.auto import tqdm  # auto picks the right frontend
from modaic.context.query_language import Filter

load_dotenv()


class TableRagIndexer(Indexer):
    def __init__(
        self, vdb_config: MilvusVDBConfig, sql_config: SQLiteConfig, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embedder = Embedder(model="openai/text-embedding-3-small")
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
        # self.vector_database.load_collection("table_rag")

        self.vector_database.create_collection(
            "table_rag", Text.schema, exists_behavior="append"
        )

    def ingest(self, files: List[str] | str, *args, **kwargs):
        if isinstance(files, str):
            files = [os.path.join(files, file) for file in os.listdir(files)]
        records = []
        with self.sql_db.connect_and_begin():
            for file in tqdm(files, desc="Ingesting files", position=0):
                if file.endswith((".csv", ".xlsx", ".xls")):
                    if file.endswith(".csv"):
                        table = Table.from_csv(file)
                    elif file.endswith((".xlsx", ".xls")):
                        table = Table.from_excel(file)
                    # Add table to file system context store
                    table.metadata["schema"] = table.schema_info()
                    # print("TABLE NAME", table.name)
                    # print("TABLE SCHEMA\n", table.schema_info())
                    # print("TABLE METADATA\n", table.metadata)
                    # print()
                    # print()
                    # print(table.metadata["schema"])
                    self.sql_db.add_table(table)
                    table.chunk_with(self.chunk_table)
                    records.extend(table.chunks)
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
                    records.extend(text_document.chunks)
        print("Adding records to vector database")
        print("number of records", len(records))
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
        return self.vector_database.search("table_rag", embedding, k, Filter(filter))

    def chunk_table(self, table: Table) -> List[Text]:
        # if (
        #     table.name == "t_5th_new_zealand_parliament_0"
        #     or table.name == "france_at_the_2013_world_aquatics_championships_0"
        # ):
        #     print("CHUNKING TABLE", table.name)
        #     print("TABLE SCHEMA\n", table.schema_info())
        #     print("TABLE METADATA\n", table.metadata)
        #     print()
        #     print()
        #     raise Exception("Stop here")
        table_md = LongText(text=table.markdown())
        table_md.chunk_text(self.text_splitter.split_text)
        table_md.apply_to_chunks(
            lambda chunk: chunk.add_metadata(
                {"type": "table", "schema": table.metadata["schema"]}
            )
        )
        # raise Exception("Stop here")
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
        vdb_config=MilvusVDBConfig.from_local("examples/TableRAG/index2.db"),
        sql_config=SQLiteConfig(db_path="examples/TableRAG/tables.db"),
    )
    excel_dir = "examples/TableRAG/dev_excel"
    docs_dir = "examples/TableRAG/dev_doc"
    # excels = [os.path.join(excel_dir, file) for file in os.listdir(excel_dir)]
    # docs = [os.path.join(docs_dir, file) for file in os.listdir(docs_dir)]
    # all_files = excels + docs
    # NOTE: will get bad results because not using the entire dataset
    # excels = excels[:20]

    # indexer.ingest(docs)
    # indexer.ingest(excels)
    x = indexer.retrieve("Who is the New Zealand Parliament Member for Canterbury")
    print(x[0][1])
    # print(x[0][1].source)
    # print(x[0][1].metadata["schema"])

    # print(type(x[0][1]))
    # print(x[0][1].metadata["schema"])
    # x = indexer.sql_query("SELECT * FROM t_5th_new_zealand_parliament_0")
    # print(x)

    # print(type(x[0]["context_schema"]))
    # print(x[0]["context_schema"])
