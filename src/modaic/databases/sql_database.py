from sqlalchemy import create_engine, MetaData, inspect, Table as SQLTable, Column, String, Text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from ..context.table import Table
from typing import Optional, Literal
import pandas as pd
from dataclasses import dataclass
from ..context.types import Source, Context, SourceType    
import os
from tqdm import tqdm
from urllib.parse import urlencode
import json
from .database import ContextDatabase

@dataclass
class SQLDatabaseConfig:
    """
    Base class for SQL database configurations.
    Each subclass must implement the `url` property.
    """
    @property
    def url(self) -> str:
        raise NotImplementedError("Subclasses must implement this method")

@dataclass
class SQLServerConfig(SQLDatabaseConfig):
    """
    Configuration for a SQL served over a port or remote connection. (MySQL, PostgreSQL, etc.)
    
    Args:
        user: The username to connect to the database.
        password: The password to connect to the database.
        host: The host of the database.
        database: The name of the database.
        port: The port of the database.
    """
    user: str
    password: str
    host: str
    database: str
    port: Optional[str] = None
    dialect: str = "mysql"
    driver: Optional[str] = None
    query_params: Optional[dict] = None

    @property
    def url(self) -> str:
        port = f":{self.port}" if self.port else ""
        driver = f"+{self.driver}" if self.driver else ""
        query = f"?{urlencode(self.query_params)}" if self.query_params else ""
        return f"{self.dialect}{driver}://{self.user}:{self.password}@{self.host}{port}/{self.database}{query}"

@dataclass
class SQLiteConfig(SQLDatabaseConfig):
    """
    Configuration for a SQLite database.
    
    Args:
        db_path: Path to the SQLite database file.
        in_memory: Whether to create an in-memory SQLite database.
        query_params: Query parameters to pass to the database.
    """
    db_path: Optional[str] = None
    in_memory: bool = False
    query_params: Optional[dict] = None

    @property
    def url(self) -> str:
        base = "sqlite:///:memory:" if self.in_memory else f"sqlite:///{self.db_path}"
        query = f"?{urlencode(self.query_params)}" if self.query_params else ""
        return f"{base}{query}"


class SQLDatabase(ContextDatabase):
    def __init__(self, 
                 config: SQLDatabaseConfig | str,
                 engine_kwargs: dict = {}, # TODO: This may not be a smart idea, may want to enforce specific kwargs
                 session_kwargs: dict = {}, # TODO: This may not be a smart idea
                 ):
        self.url = config.url if isinstance(config, SQLDatabaseConfig) else config
        self.engine = create_engine(self.url, **engine_kwargs)
        self.metadata = MetaData()
        self.session = sessionmaker(bind=self.engine, **session_kwargs)
        self.inspector = inspect(self.engine)
        
        # Create metadata table to store table metadata
        self.metadata_table = SQLTable(
            'metadata', 
            self.metadata,
            Column('table_name', String(255), primary_key=True),
            Column('metadata_json', Text),
        )
        self.metadata.create_all(self.engine)
    
    def add_table(self, table: Table, if_exists: Literal['fail', 'replace', 'append'] = 'replace', schema: str = None):
        # TODO: support batch inserting for large dataframes
        table.df.to_sql(table.name, self.engine, if_exists=if_exists, index=False)
        
        # Store table metadata in the metadata table
        with self.engine.connect() as conn:
            # Remove existing metadata for this table if it exists
            conn.execute(self.metadata_table.delete().where(
                self.metadata_table.c.table_name == table.name
            ))
            
            # Insert new metadata
            conn.execute(self.metadata_table.insert().values(
                table_name=table.name,
                metadata_json=json.dumps(table.metadata),
                created_at=datetime.utcnow()
            ))
            conn.commit()
        
        table.set_source(Source(
            origin=self,
            type=SourceType.CONTEXT_OBJECT,
            metadata={
                "table_name": table.name,
                "sql_schema": schema
            }
        ))
    
    def drop_table(self, name: str):
        """
        Drop a table from the database and remove its metadata.
        
        Args:
            name: The name of the table to drop
        """
        with self.engine.connect() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {name}")
            # Also remove metadata for this table
            conn.execute(self.metadata_table.delete().where(
                self.metadata_table.c.table_name == name
            ))
            conn.commit()
    
    def list_tables(self):
        """
        List all tables currently in the database.
        
        Returns:
            List of table names in the database.
        """
        return self.inspector.get_table_names()
    
    def get_table(self, name: str) -> Table:
        df = pd.read_sql_table(name, self.engine)
            
        return Table(df, name=name, metadata=self.get_table_metadata(name))
    
    
    def get_table_schema(self, name: str):
        """
        Return column schema for a given table.
        
        Args:
            name: The name of the table to get schema for
            
        Returns:
            Column schema information for the table.
        """
        return self.inspector.get_columns(name)
    
    def get_table_metadata(self, name: str) -> dict:
        """
        Get metadata for a specific table.
        
        Args:
            name: The name of the table to get metadata for
            
        Returns:
            Dictionary containing the table's metadata, or empty dict if not found.
        """
        with self.engine.connect() as conn:
            result = conn.execute(
                self.metadata_table.select().where(
                    self.metadata_table.c.table_name == name
                )
            ).fetchone()
            
            if result:
                return json.loads(result.metadata_json)
            return {}
    
    def add_context(self, context: Context):
        pass
        
    
    @classmethod
    def from_dir(cls, dir_path: str, config: SQLDatabaseConfig): 
        # TODO: support batch inserting and parallel processing
        """
        Initializes a new SQLDatabase from a directory of files.
        
        Args:
            dir_path: Path to the directory containing files to load
            config: SQL database configuration
            
        Returns:
            New SQLDatabase instance loaded with data from the directory.
        """
        # TODO: make sure the loaded sql database is empty if not raise error and tell user to use __init__ for an already existing database
        instance = cls(config)
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"File not found: {dir_path}")
        for file_name in tqdm(os.listdir(dir_path), desc="Uploading files to SQL database"):
            full_path = os.path.join(dir_path, file_name)
            if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                table = Table.from_excel(full_path)
                instance.add_table(table, if_exists='fail')
            elif file_name.endswith(".csv"):
                table = Table.from_csv(full_path)
                instance.add_table(table, if_exists='fail')
        return instance

class MultiTenantSQLDatabase:
    raise NotImplementedError("Not implemented")
        
    
    


# def parse_excel_file_and_insert_to_db(excel_file_outer_dir: str):
#     if not os.path.exists(excel_file_outer_dir):
#         raise FileNotFoundError(f"File not found: {excel_file_outer_dir}")
    

#     for file_name in tqdm(os.listdir(excel_file_outer_dir)):
#         if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
#             full_path = os.path.join(excel_file_outer_dir, file_name)
#             df = pd.read_excel(full_path)
            
#             df_convert = df.apply(infer_and_convert)
#             df_convert = transfer_df_columns(df_convert)

#             schema_dict, table_name = generate_schema_info(df_convert, file_name)

#             # 确保目录存在
#             if not os.path.exists(SCHEMA_DIR):
#                 os.makedirs(SCHEMA_DIR)

#             with open(f"{SCHEMA_DIR}/{table_name}.json", 'w', encoding='utf-8') as f:
#                 json.dump(schema_dict, f, ensure_ascii=False)
            
#             sql_alchemy_helper.insert_dataframe_batch(df_convert, table_name)
