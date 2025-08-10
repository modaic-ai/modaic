from typing import (
    Type,
    Any,
    ClassVar,
    Optional,
    List,
    Iterable,
    get_args,
    get_origin,
    Dict,
)
from typing_extensions import Annotated
from pymilvus import DataType, MilvusClient
from pydantic import BaseModel
from ...databases.vector_database import VectorDatabaseConfig
from ...context.base import SerializedContext, Context
import uuid
import numpy as np
import dspy
from dataclasses import dataclass, field
from ...types import (
    int8,
    int16,
    int32,
    int64,
    float32,
    float64,
    double,
    Array,
    Vector,
    Float16Vector,
    BFloat16Vector,
    BinaryVector,
    String,
    SchemaField,
    Modaic_Type,
)
from collections.abc import Sequence, Mapping


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
    client: MilvusClient,
    collection_name: str,
    payload_schema: Dict[str, SchemaField],
    embedding_dim: int,
) -> Any:
    """
    Create a Milvus collection.
    """
    if "vector" not in payload_schema:
        payload_schema["vector"] = SchemaField(
            type="Vector",
            size=embedding_dim,
        )

    schema = _modaic_to_milvus_schema(client, payload_schema)
    client.create_collection(collection_name, schema=schema)
    return schema


def has_collection(client: MilvusClient, collection_name: str) -> bool:
    """
    Check if a collection exists in Milvus.

    Params:
        client: The Milvus client instance
        collection_name: The name of the collection to check

    Returns:
        bool: True if the collection exists, False otherwise
    """
    return client.has_collection(collection_name)


def _modaic_to_milvus_schema(
    client: MilvusClient,
    modaic_schema: Dict[str, SchemaField],
) -> Any:
    """
    Convert a Pydantic BaseModel schema to a Milvus collection schema.

    Params:
        client: The Milvus client instance
        modaic_schema: The Modaic schema to convert

    Returns:
        Any: The Milvus schema object
    """
    num_type_to_milvus: Mapping[Modaic_Type, DataType] = {
        "int8": DataType.INT8,
        "int16": DataType.INT16,
        "int32": DataType.INT32,
        "int64": DataType.INT64,
        "float32": DataType.FLOAT,
        "float64": DataType.DOUBLE,
        "bool": DataType.BOOL,
        "String": DataType.VARCHAR,
        "Mapping": DataType.JSON,
    }
    vector_type_to_milvus: Mapping[Modaic_Type, DataType] = {
        "Vector": DataType.FLOAT_VECTOR,
        "Float16Vector": DataType.FLOAT16_VECTOR,
        "BFloat16Vector": DataType.BFLOAT16_VECTOR,
        "BinaryVector": DataType.BINARY_VECTOR,
    }
    max_str_length = 65_535

    milvus_schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    for field_name, field_info in modaic_schema.items():
        field_type = field_info.type

        if field_name == "id":
            assert field_info.optional is False, "id field cannot be Optional"
            if (
                field_type == "int64" or field_type == "int32"
            ):  # CAVEAT: Milvus only accepts int64 for id
                milvus_schema.add_field(
                    field_name=field_name,
                    datatype=DataType.INT64,
                    is_primary=True,
                    auto_id=False,
                )
            elif field_type == "String":
                milvus_schema.add_field(
                    field_name=field_name,
                    datatype=DataType.VARCHAR,
                    max_length=field_info.size or max_str_length,
                    is_primary=True,
                    auto_id=False,
                )
            else:
                raise ValueError(f"Unsupported field type: {field_type}")
        elif field_type in num_type_to_milvus:
            milvus_data_type = num_type_to_milvus[field_type]
            milvus_schema.add_field(
                field_name=field_name,
                datatype=milvus_data_type,
                nullable=field_info["optional"],
            )
        elif isinstance(field_type, Vector):
            field_dim = field_info["dim"]
            milvus_schema.add_field(
                field_name=field_name,
                datatype=vector_type_to_milvus[field_type],
                dim=field_dim,
            )
        elif isinstance(field_type, Array):
            milvus_schema.add_field(
                field_name=field_name,
                datatype=DataType.ARRAY,
                element_type=DataType.INT64,
                max_capacity=field_info["max_size"],
                nullable=field_info["optional"],
            )
        elif isinstance(field_type, String):
            milvus_schema.add_field(
                field_name=field_name,
                datatype=DataType.VARCHAR,
                max_length=field_info["max_size"],
                nullable=field_info["optional"],
            )
        else:
            raise ValueError(f"Unsupported field type: {field_type}")
    return milvus_schema


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
        payload_schema: Type[SerializedContext] = None,
        **kwargs,
    ):
        self.config = config
        self.embedder = embedder
        self.payload_schema = payload_schema
        self.client = _init(config)
        self.schema = _convert_scontext_to_milvus_schema(
            self.client, payload_schema, embedder.embedding_dim
        )

    def add_records(
        self,
        collection_name: str,
        records: Iterable[Context | SerializedContext],
        batch_size: Optional[int] = None,
    ):
        pass
