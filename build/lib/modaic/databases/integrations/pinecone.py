from typing import TYPE_CHECKING
from modaic.databases.vector_database import VectorDatabase, VectorDatabaseConfig

if TYPE_CHECKING:
    from pinecone import Pinecone

class PineconeVDBConfig(VectorDatabaseConfig):
    api_key: str
    environment: str
    index_name: str
    dimension: int
    metric: str
    pod_type: str
    

class PineconeVDB(VectorDatabase):
    def __init__(self, config: PineconeVDBConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        