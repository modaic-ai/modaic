from typing import Type, Any, ClassVar, Optional, List, Iterable
from pymilvus import DataType, MilvusClient
from pydantic import BaseModel
from ...databases.vector_database import VectorDatabaseConfig
from ...context.types import SerializedContext, Context
import uuid
import numpy as np
import dspy
from dataclasses import dataclass, field


@dataclass
class MilvusVDBConfig(VectorDatabaseConfig):
    _module: ClassVar[str] = "modaic.databases.integrations.milvus"

    uri: str = "http://localhost:19530"
    user: str = ""
    password: str = ""
    db_name: str = ""
    token: str = ""
    timeout: Optional[float] = None
    kwargs: dict = field(default_factory=dict)

    @staticmethod
    def from_local(file_path: str):
        return MilvusVDBConfig(uri=file_path)


def _init(config: MilvusVDBConfig):
    """
    Initialize a Milvus vector database.
    """
    return MilvusClient(
        uri=config.uri,
        user=config.user,
        password=config.password,
        db_name=config.db_name,
        token=config.token,
        timeout=config.timeout,
        **config.kwargs,
    )


def _create_record(embedding: np.ndarray, scontext: SerializedContext) -> Any:
    """
    Convert a SerializedContext to a record for Milvus.
    """
    record = {
        "id": str(uuid.uuid4()),
        "vector": embedding,
        "metadata": scontext.metadata,
        "source": scontext.source,
        "readme": scontext.readme,
        "embed_context": scontext.embed_context,
        "type": scontext.type.__name__,
    }
    return record


def add_records(client: MilvusClient, collection_name: str, records: List[Any]):
    """
    Add records to a Milvus collection.
    """
    client.insert(collection_name, records)


def drop_collection(client: MilvusClient, collection_name: str):
    """
    Drop a Milvus collection.
    """
    client.drop_collection(collection_name)


def create_collection(
    client: MilvusClient, collection_name: str, metadata_schema: Type[BaseModel]
) -> Any:
    """
    Create a Milvus collection.
    """
    schema = _convert_pydantic_to_milvus_schema(client, metadata_schema)
    client.create_collection(collection_name, schema)
    return schema


def has_collection(client: MilvusClient, collection_name: str) -> bool:
    """
    Check if a collection exists in Milvus.
    """
    return client.has_collection(collection_name)


def _convert_pydantic_to_milvus_schema(
    client: MilvusClient, pydantic_schema: Type[BaseModel]
) -> Any:
    """
    Convert a Pydantic BaseModel schema to a Milvus collection schema.
    """
    # Create a new Milvus schema
    milvus_schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

    # Get field information from the Pydantic model
    for field_name, field_info in pydantic_schema.model_fields.items():
        field_type = field_info.annotation

        # Map Python types to Milvus DataTypes
        milvus_data_type = _map_python_type_to_milvus(field_type, field_name)

        # Determine if this is a primary key field
        is_primary = getattr(field_info, "is_primary", field_name == "id")

        # Add field to schema
        if milvus_data_type == DataType.FLOAT_VECTOR:
            # For vector fields, we need to specify dimension
            # This assumes the dimension is stored in field metadata or defaults to embedder output size
            dim = getattr(field_info, "dim", getattr(self.embedder, "output_size", 384))
            milvus_schema.add_field(
                field_name=field_name,
                datatype=milvus_data_type,
                dim=dim,
                is_primary=is_primary,
            )
        elif milvus_data_type == DataType.VARCHAR:
            # For VARCHAR fields, specify max_length
            max_length = getattr(field_info, "max_length", 512)
            milvus_schema.add_field(
                field_name=field_name,
                datatype=milvus_data_type,
                max_length=max_length,
                is_primary=is_primary,
            )
        else:
            milvus_schema.add_field(
                field_name=field_name, datatype=milvus_data_type, is_primary=is_primary
            )

    return milvus_schema


def _map_python_type_to_milvus(self, python_type: Type, field_name: str) -> DataType:
    """
    Map Python types to Milvus DataTypes.
    """
    # Handle common type mappings
    type_mapping = {
        int: DataType.INT64,
        str: DataType.VARCHAR,
        float: DataType.FLOAT,
        bool: DataType.BOOL,
    }

    # Check for direct type mapping
    if python_type in type_mapping:
        return type_mapping[python_type]

    # Handle typing annotations
    if hasattr(python_type, "__origin__"):
        origin = python_type.__origin__
        if origin is list:
            # Assume list of floats is a vector field
            return DataType.FLOAT_VECTOR
        elif origin is dict:
            return DataType.JSON

    # Special handling for fields that might be vectors
    if "vector" in field_name.lower() or "embedding" in field_name.lower():
        return DataType.FLOAT_VECTOR

    # Default to VARCHAR for unknown types
    return DataType.VARCHAR


def search(
    client: MilvusClient,
    collection_name: str,
    vector: np.ndarray | List[int],
    k: int = 10,
    filter: Optional[dict] = None,
    **kwargs,
) -> List[SerializedContext]:
    """
    Retrieve records from the vector database.
    """
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    results = client.search(
        collection_name=collection_name, data=[vector], limit=k, filter=filter, **kwargs
    )
    context_list = []
    for result in results:
        match result:
            case {"id": id, "distance": distance, "entity": entity}:
                context_list.append(
                    {
                        "id": id,
                        "distance": distance,
                        "serialized_context": SerializedContext(**entity),
                    }
                )
            case _:
                raise ValueError(f"Invalid result format: {result}")
    return context_list


class MilvusVectorDatabase:
    def __init__(
        self,
        config: MilvusVDBConfig,
        embedder: dspy.Embedder,
        metadata_schema: Type[BaseModel] = None,
        **kwargs,
    ):
        self.config = config
        self.embedder = embedder
        self.metadata_schema = metadata_schema
        self.client = _init(config)
        self.schema = _convert_pydantic_to_milvus_schema(self.client, metadata_schema)

    def add_records(
        self,
        collection_name: str,
        records: Iterable[Context | SerializedContext],
        batch_size: Optional[int] = None,
    ):
        pass
