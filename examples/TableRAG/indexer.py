from modaic.types import Indexer
from modaic.databases import VectorDatabase, MilvusVDBConfig
from typing import List, Literal
import dspy
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json
from modaic.databases import SQLDatabase, SQLiteConfig
from modaic.context import SerializedContext, Table, LongText
from modaic.utils import PineconeReranker
from dotenv import load_dotenv
load_dotenv()

class TableRagIndexer(Indexer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vdb_config = MilvusVDBConfig(
            host="index.db",
            collection_name="table_rag",
            embedding_dim=1536,
            index_type="FLAT",
        )
        self.embedder = dspy.Embedder(
            model="text-embedding-3-small"
        )
        self.vector_database = VectorDatabase(config=vdb_config, embedder=self.embedder)
        sql_config = SQLiteConfig(db_path="contexts/tables.db")
        self.sql_db = SQLDatabase(config=sql_config)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        self.reranker = PineconeReranker(
            model="bge-reranker-v2-m3",
            api_key=os.getenv("PINECONE_API_KEY")
        )
    
    def ingest(self, files: List[str] | str, *args, **kwargs):
        if isinstance(files, str):
            files = [os.path.join(files, file) for file in os.listdir(files)]
        records = []
        for file in files:
            if file.endswith((".csv", ".xlsx", ".xls")):
                if file.endswith(".csv"):
                    table = Table.from_csv(file)
                elif file.endswith((".xlsx", ".xls")):
                    table = Table.from_excel(file)
                # Add table to file system context store
                table.metadata["schema"] = table.schema_info()
                self.sql_db.add_table(table)
                # Get table markdown and chunk it
                table_md = LongText(text=table.to_markdown())
                table_md.chunk(self.text_splitter)
                # Serialize table to store as payload in vector database
                serialized_table = table.serialize()
                for chunk in table_md.chunks:
                    payload = serialized_table.copy()
                    payload.metadata["md_chunk"] = chunk.text
                    payload.metadata["type"] = "table"
                    records.append((chunk.text, payload))
                    
            elif file.endswith((".json")):
                with open(file, 'r', encoding="utf-8") as f:
                    data_split = json.load(f)
                key_value_doc = ''
                for key, item in data_split.items():
                    key_value_doc += f"{key} {item}\n"
                text_document = LongText(text=key_value_doc)
                text_document.chunk(self.text_splitter)
                text_document.apply_to_chunks(lambda chunk: chunk.add_metadata({"type": "text"}))
                records.extend([chunk for chunk in text_document.chunks])
            
        self.vector_database.add_records("table_rag", records)
    
    def retrieve(self, user_query: str, k_recall: int = 10, k_rerank: int = 10, type: Literal["table", "text", "all"] = "all") -> List[SerializedContext]:
        results = self.recall(user_query, k_recall, type)
        results = self.reranker(user_query, [result.payload for result in results], k_rerank)
        return results
    
    def recall(self, user_query: str, k: int = 10, type: Literal["table", "text", "all"] = "all") -> List[SerializedContext]:
        embedding = self.embedder([user_query])[0]
        if type == "table":
            filter = {"metadata":{"type": "table"}}
        elif type == "text":
            filter = {"metadata":{"type": "text"}}
        else:
            filter = None
        return self.vector_database.search("table_rag", embedding, k, filter)
        
        
                
            
        
    
