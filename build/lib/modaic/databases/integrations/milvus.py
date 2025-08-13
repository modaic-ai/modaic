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
from ...databases.vector_database import (
    VectorDatabaseConfig,
    IndexType,
    Metric,
    IndexConfig,
    VectorType,
)
from ...context.base import ContextSchema, Context
import uuid
import numpy as np
import dspy
from dataclasses import dataclass, field
from ...types import SchemaField, Modaic_Type
from collections.abc import Sequence, Mapping
from ..vector_database import SearchResult


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


def _init(config: MilvusVDBConfig) -> MilvusClient:
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


def _create_record(
    embedding_map: Dict[str, np.ndarray], scontext: ContextSchema
) -> Any:
    """
    Convert a ContextSchema to a record for Milvus.
    """
    record = scontext.model_dump(mode="json")
    for index_name, embedding in embedding_map.items():
        record[index_name] = embedding
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
    index: List[IndexConfig] = IndexConfig(),
):
    """
    Create a Milvus collection.
    """

    schema = _modaic_to_milvus_schema(client, payload_schema)
    modaic_to_milvus_vector = {
        VectorType.FLOAT: DataType.FLOAT_VECTOR,
        VectorType.FLOAT16: DataType.FLOAT16_VECTOR,
        VectorType.BFLOAT16: DataType.BFLOAT16_VECTOR,
        VectorType.BINARY: DataType.BINARY_VECTOR,
        VectorType.FLOAT_SPARSE: DataType.SPARSE_FLOAT_VECTOR,
        # VectorType.INT8: DataType.INT8_VECTOR,
    }

    for index_config in index:
        try:
            vector_type = modaic_to_milvus_vector[index_config.vector_type]
        except KeyError:
            raise ValueError(
                f"Milvus does not support vector type: {index_config.vector_type}"
            )
        kwargs = {
            "field_name": index_config.name,
            "datatype": vector_type,
        }
        # sparse vectors don't have a dim in milvus
        if index_config.vector_type != VectorType.FLOAT_SPARSE:
            # sparse vectors don't have a dim in milvus
            kwargs["dim"] = index_config.embedder.embedding_dim
        schema.add_field(**kwargs)

    index_params = client.prepare_index_params()
    index_type = (
        index_config.index_type.name
        if index_config.index_type != IndexType.DEFAULT
        else "AUTOINDEX"
    )
    try:
        metric_type = index_config.metric.supported_libraries["milvus"]
    except KeyError:
        raise ValueError(f"Milvus does not support metric type: {index_config.metric}")
    index_params.add_index(
        field_name=index_config.name,
        index_name=f"{index_config.name}_index",
        index_type=index_type,
        metric_type=metric_type,
    )

    client.create_collection(collection_name, schema=schema, index_params=index_params)


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
    scalar_type_to_milvus: Mapping[Modaic_Type, DataType] = {
        "int8": DataType.INT8,
        "int16": DataType.INT16,
        "int32": DataType.INT32,
        "int64": DataType.INT64,
        "float32": DataType.FLOAT,
        "float64": DataType.DOUBLE,
        "bool": DataType.BOOL,
    }
    # vector_type_to_milvus: Mapping[Modaic_Type, DataType] = {
    #     "Vector": DataType.FLOAT_VECTOR,
    #     "Float16Vector": DataType.FLOAT16_VECTOR,
    #     "BFloat16Vector": DataType.BFLOAT16_VECTOR,
    #     "BinaryVector": DataType.BINARY_VECTOR,
    # }
    max_str_length = 65_535
    max_array_capacity = 4096

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
                raise ValueError(f"Milvus does not support id field type: {field_type}")
        elif field_type in scalar_type_to_milvus:
            milvus_data_type = scalar_type_to_milvus[field_type]
            milvus_schema.add_field(
                field_name=field_name,
                datatype=milvus_data_type,
                nullable=field_info["optional"],
            )
        elif field_type == "Array":
            inner_type = field_info.inner_type
            if inner_type == "String":
                milvus_schema.add_field(
                    field_name=field_name,
                    datatype=DataType.ARRAY,
                    element_type=DataType.VARCHAR,
                    max_length=field_info.size or max_str_length,
                    max_capacity=inner_type.size or max_array_capacity,
                    nullable=field_info.optional,
                )
            elif inner_type.type in scalar_type_to_milvus:
                milvus_schema.add_field(
                    field_name=field_name,
                    datatype=DataType.ARRAY,
                    element_type=scalar_type_to_milvus[inner_type.type],
                    max_capacity=inner_type.size or max_array_capacity,
                    nullable=field_info.optional,
                )
            else:
                raise ValueError(
                    f"Milvus does not support inner type {inner_type.type} for Array field: {field_name}"
                )
        elif field_type == "String":
            milvus_schema.add_field(
                field_name=field_name,
                datatype=DataType.VARCHAR,
                max_length=field_info.size or max_str_length,
            )
        # elif field_type in vector_type_to_milvus:
        #     milvus_schema.add_field(
        #         field_name=field_name,
        #         datatype=vector_type_to_milvus[field_type],
        #         dim=field_info.size,
        #     )
        elif field_type == "Mapping":
            milvus_schema.add_field(
                field_name=field_name,
                datatype=DataType.JSON,
                nullable=field_info.optional,
            )
        else:
            raise ValueError(
                f"Unsupported field type for Milvus - {field_name}: {field_type}"
            )
    return milvus_schema


def search(
    client: MilvusClient,
    collection_name: str,
    vector: np.ndarray | List[int],
    payload_schema: Type[BaseModel],
    k: int = 10,
    filter: Optional[dict] = None,
    index_name: Optional[str] = None,
) -> List[SearchResult]:
    """
    Retrieve records from the vector database.
    """
    if index_name is None:
        raise ValueError("Milvus requires an index_name to be specified for search")

    output_fields = [field_name for field_name in payload_schema.model_fields]

    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    results = client.search(
        collection_name=collection_name,
        data=[vector],
        limit=k,
        filter=filter,
        anns_field=index_name,
        output_fields=output_fields,
    )
    # print("search results", results)
    context_list = []
    # print("result type", type(results))
    # raise Exception("stop here")
    for result in results[0]:
        # print("result", result)
        match result:
            case {"id": id, "distance": distance, "entity": entity}:
                context_list.append(
                    {
                        "id": id,
                        "distance": distance,
                        "context_schema": payload_schema.model_validate(entity),
                    }
                )
            case _:
                raise ValueError(
                    f"Failed to parse search results to {payload_schema.__name__}: {result}"
                )
    return context_list


class MilvusVectorDatabase:
    def __init__(
        self,
        config: MilvusVDBConfig,
        embedder: dspy.Embedder,
        payload_schema: Type[ContextSchema] = None,
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
        records: Iterable[Context | ContextSchema],
        batch_size: Optional[int] = None,
    ):
        pass
