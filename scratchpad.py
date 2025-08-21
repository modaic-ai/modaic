from modaic import AutoAgent, AutoConfig, AutoIndexer
from modaic.hub import get_user_info
from modaic.databases import MilvusVDBConfig, SQLiteConfig
from dotenv import load_dotenv
import os
import sys

load_dotenv()

print("sys.path", sys.path)
# import agent
vdb_config = MilvusVDBConfig.from_local("examples/TableRAG/index2.db")
sql_config = SQLiteConfig(db_path="examples/TableRAG/tables.db")
indexer = AutoIndexer.from_precompiled(
    "swagginty/TableRAG",
    vdb_config=vdb_config,
    sql_config=sql_config,
)

print(indexer)
