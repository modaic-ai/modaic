from collections.abc import Mapping
from typing import Any, ClassVar, Dict, List, Literal, Optional, Type, Union

import numpy as np
from langchain_community.query_constructors.milvus import MilvusTranslator as MilvusTranslator_
from langchain_core.structured_query import Comparator, Comparison, Visitor
from pymilvus import DataType, MilvusClient
from pymilvus.orm.collection import CollectionSchema

from ....context.base import Context
from ....exceptions import BackendCompatibilityError
from ....types import InnerField, Schema, SchemaField, float_format, int_format
from ..vector_database import DEFAULT_INDEX_NAME, IndexConfig, IndexType, SearchResult, VectorType

milvus_to_modaic_vector = {
    VectorType.FLOAT: DataType.FLOAT_VECTOR,
    VectorType.FLOAT16: DataType.FLOAT16_VECTOR,
    VectorType.BFLOAT16: DataType.BFLOAT16_VECTOR,
    VectorType.BINARY: DataType.BINARY_VECTOR,
    VectorType.FLOAT_SPARSE: DataType.SPARSE_FLOAT_VECTOR,
    # VectorType.INT8: DataType.INT8_VECTOR,
}

modaic_to_milvus_index = {
    IndexType.DEFAULT: "AUTOINDEX",
    IndexType.HNSW: "HNSW",
    IndexType.FLAT: "FLAT",
    IndexType.IVF_FLAT: "IVF_FLAT",
    IndexType.IVF_SQ8: "IVF_SQ8",
    IndexType.IVF_PQ: "IVF_PQ",
    IndexType.IVF_RABITQ: "IVF_RABITQ",
    IndexType.GPU_IVF_FLAT: "GPU_IVF_FLAT",
    IndexType.GPU_IVF_PQ: "GPU_IVF_PQ",
    IndexType.DISKANN: "DISKANN",
    IndexType.BIN_FLAT: "BIN_FLAT",
    IndexType.BIN_IVF_FLAT: "BIN_IVF_FLAT",
    IndexType.MINHASH_LSH: "MINHASH_LSH",
    IndexType.SPARSE_INVERTED_INDEX: "SPARSE_INVERTED_INDEX",
    IndexType.INVERTED: "INVERTED",
    IndexType.BITMAP: "BITMAP",
    IndexType.TRIE: "TRIE",
    IndexType.STL_SORT: "STL_SORT",
}

# Name for field that tracks which fields are null for a record (only used for milvus lite)
NULL_FIELD_NAME = "null_fields"


class MilvusTranslator(MilvusTranslator_):
    """
    Patch of langchain_community's MilvusTranslator to support lists of strings values.
    """

    def visit_comparison(self, comparison: Comparison) -> str:
        comparator = self._format_func(comparison.comparator)
        processed_value = process_value(comparison.value, comparison.comparator)
        attribute = comparison.attribute

        return "( " + attribute + " " + comparator + " " + processed_value + " )"


class MilvusBackend:
    _name: ClassVar[Literal["milvus"]] = "milvus"
    mql_translator: Visitor = MilvusTranslator()

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        user: str = "",
        password: str = "",
        db_name: str = "",
        token: str = "",
        timeout: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize a Milvus vector database.
        """

        if uri.startswith(("http://", "https://", "tcp://")):
            self.milvus_lite = False
        elif uri.endswith(".db"):
            self.milvus_lite = True
        else:
            raise ValueError(
                f"Invalid URI: {uri}, must start with http://, https://, or tcp:// for milvus server or end with .db for milvus lite"
            )
        self._client = MilvusClient(
            uri=uri,
            user=user,
            password=password,
            db_name=db_name,
            token=token,
            timeout=timeout,
            **kwargs,
        )

    def create_record(self, embedding_map: Dict[str, np.ndarray], context: Context) -> Any:
        """
        Convert a Context to a record for Milvus.
        """
        # CAVEAT: users can optionally hide fields from model_dump(). Use include_hidden=True to get all fields.
        record = context.model_dump(include_hidden=True)
        # NOTE: Track null values if using milvus lite since null values are not supported in milvus lite
        if self.milvus_lite:
            schema = context.schema().as_dict()
            null_fields = []
            for field_name, field_value in record.items():
                if field_value is None:
                    null_fields.append(field_name)
                    if schema[field_name].type == "string":
                        record[field_name] = ""
                    elif schema[field_name].type == "array":
                        record[field_name] = []
                    elif schema[field_name].type == "object":
                        record[field_name] = {}
                    elif schema[field_name].type == "number" or schema[field_name].type == "integer":
                        record[field_name] = 0
                    elif schema[field_name].type == "boolean":
                        record[field_name] = False

            record[NULL_FIELD_NAME] = null_fields

        for index_name, embedding in embedding_map.items():
            record[index_name] = embedding.tolist()
        return record

    def add_records(self, collection_name: str, records: List[Any]):
        """
        Add records to a Milvus collection.
        """
        self._client.insert(collection_name, records)

    def list_collections(self) -> List[str]:
        return self._client.list_collections()

    def drop_collection(self, collection_name: str):
        """
        Drop a Milvus collection.
        """
        self._client.drop_collection(collection_name)

    def create_collection(
        self,
        collection_name: str,
        payload_class: Type[Context],
        index: IndexConfig = IndexConfig(),  # noqa: B008
    ):
        """
        Create a Milvus collection.
        """
        if not issubclass(payload_class, Context):
            raise TypeError(f"Payload class {payload_class} is must be a subclass of Context")

        schema = _modaic_to_milvus_schema(self._client, payload_class.schema(), self.milvus_lite)
        modaic_to_milvus_vector = {
            VectorType.FLOAT: DataType.FLOAT_VECTOR,
            VectorType.FLOAT16: DataType.FLOAT16_VECTOR,
            VectorType.BFLOAT16: DataType.BFLOAT16_VECTOR,
            VectorType.BINARY: DataType.BINARY_VECTOR,
            VectorType.FLOAT_SPARSE: DataType.SPARSE_FLOAT_VECTOR,
            # VectorType.INT8: DataType.INT8_VECTOR,
        }

        try:
            vector_type = modaic_to_milvus_vector[index.vector_type]
        except KeyError:
            raise ValueError(f"Milvus does not support vector type: {index.vector_type}") from None
        kwargs = {
            "field_name": DEFAULT_INDEX_NAME,
            "datatype": vector_type,
        }
        # NOTE: sparse vectors don't have a dim in milvus
        if index.vector_type != VectorType.FLOAT_SPARSE:
            kwargs["dim"] = index.embedder.embedding_dim
        schema.add_field(**kwargs)

        index_params = self._client.prepare_index_params()
        index_type = modaic_to_milvus_index[index.index_type]
        try:
            metric_type = index.metric.supported_libraries["milvus"]
        except KeyError:
            raise ValueError(f"Milvus does not support metric type: {index.metric}") from None
        index_params.add_index(
            field_name=DEFAULT_INDEX_NAME,
            index_name=f"{DEFAULT_INDEX_NAME}_index",
            index_type=index_type,
            metric_type=metric_type,
        )

        self._client.create_collection(collection_name, schema=schema, index_params=index_params)

    def has_collection(self, collection_name: str) -> bool:
        """
        Check if a collection exists in Milvus.

        Args:
            client: The Milvus client instance
            collection_name: The name of the collection to check

        Returns:
            bool: True if the collection exists, False otherwise
        """
        return self._client.has_collection(collection_name)

    def search(
        self,
        collection_name: str,
        vectors: List[np.ndarray],
        payload_class: Type[Context],
        k: int = 10,
        filter: Optional[str] = None,
    ) -> List[List[SearchResult]]:
        """
        Retrieve records from the vector database.
        """
        if not issubclass(payload_class, Context):
            raise TypeError(f"Payload class {payload_class} is must be a subclass of Context")

        output_fields = [field_name for field_name in payload_class.model_fields]
        if self.milvus_lite:
            output_fields.append(NULL_FIELD_NAME)
        listified_vectors = [vector.tolist() for vector in vectors]

        searches = self._client.search(
            collection_name=collection_name,
            data=listified_vectors,
            limit=k,
            filter=filter,
            anns_field=DEFAULT_INDEX_NAME,  # Use the same field name as in create_collection
            output_fields=output_fields,
        )

        all_results = []
        for search in searches:
            context_list = []
            for result in search:
                match result:
                    case {"id": id, "distance": distance, "entity": entity}:
                        context_list.append(
                            SearchResult(
                                id=id, score=distance, context=payload_class.model_validate(self._process_null(entity))
                            )
                        )
                    case _:
                        raise ValueError(f"Failed to parse search results to {payload_class.__name__}: {result}")
            all_results.append(context_list)

        return all_results

    def get_records(self, collection_name: str, payload_class: Type[Context], record_ids: List[str]) -> List[Context]:
        output_fields = [field_name for field_name in payload_class.model_fields]
        if self.milvus_lite:
            output_fields.append(NULL_FIELD_NAME)
        records = self._client.get(collection_name=collection_name, ids=record_ids, output_fields=output_fields)
        return [payload_class.model_validate(self._process_null(record)) for record in records]

    @staticmethod
    def from_local(file_path: str) -> "MilvusBackend":
        return MilvusBackend(uri=file_path)

    def _process_null(self, record: dict) -> dict:
        if self.milvus_lite and NULL_FIELD_NAME in record:
            for field_name in record[NULL_FIELD_NAME]:
                record[field_name] = None
            del record[NULL_FIELD_NAME]
        return record


def _modaic_to_milvus_schema(client: MilvusClient, modaic_schema: Schema, milvus_lite: bool) -> CollectionSchema:
    """
    Convert a Pydantic BaseModel schema to a Milvus collection schema.

    Args:
        client: The Milvus client instance
        modaic_schema: The Modaic schema to convert
        milvus_lite: Whether the schema is for a milvus lite database

    Returns:
        Any: The Milvus schema object
    """
    # Maps types that can contain the 'format' keyword to the default milvus data type
    formatted_types: Mapping[Literal["integer", "number"], DataType] = {
        "integer": DataType.INT64,
        "number": DataType.DOUBLE,
    }
    # Maps types that do not contain the 'format' keyword to the milvus data type
    non_formatted_types: Mapping[Literal["string", "boolean"], DataType] = {
        "string": DataType.VARCHAR,
        "boolean": DataType.BOOL,
    }
    # Maps values for the 'format' keyword to the milvus data type
    format_to_milvus: Mapping[int_format | float_format, DataType] = {
        "int8": DataType.INT8,
        "int16": DataType.INT16,
        "int32": DataType.INT32,
        "int64": DataType.INT64,
        "float": DataType.FLOAT,
        "double": DataType.DOUBLE,
        "bool": DataType.BOOL,
    }

    MAX_STR_LENGTH = 65_535  # noqa: N806
    MAX_ARRAY_CAPACITY = 4096  # noqa: N806

    def get_milvus_type(sf: SchemaField | InnerField) -> DataType:
        type_ = sf.type
        format_ = sf.format
        if type_ in formatted_types and format_ in format_to_milvus:
            milvus_data_type = format_to_milvus[format_]
        elif type_ in formatted_types:
            milvus_data_type = formatted_types[type_]
        elif type_ in non_formatted_types:
            milvus_data_type = non_formatted_types[type_]
        else:
            raise ValueError(f"Milvus does not support field type: {type_}")
        return milvus_data_type

    def is_nullable(sf: SchemaField | InnerField) -> bool:
        if milvus_lite:
            return False
        return sf.optional

    milvus_schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    for field_name, schema_field in modaic_schema.as_dict().items():
        if schema_field.type == "array":
            if schema_field.inner_type.type == "string":
                milvus_schema.add_field(
                    field_name=field_name,
                    datatype=DataType.ARRAY,
                    nullable=is_nullable(schema_field),
                    element_type=DataType.VARCHAR,
                    max_capacity=schema_field.size or MAX_ARRAY_CAPACITY,
                    max_length=schema_field.inner_type.size or MAX_STR_LENGTH,
                )
            else:
                milvus_schema.add_field(
                    field_name=field_name,
                    datatype=DataType.ARRAY,
                    nullable=is_nullable(schema_field),
                    element_type=get_milvus_type(schema_field.inner_type),
                    max_capacity=schema_field.size or MAX_ARRAY_CAPACITY,
                )
        elif schema_field.type == "string":
            milvus_schema.add_field(
                field_name=field_name,
                datatype=DataType.VARCHAR,
                max_length=schema_field.size or MAX_STR_LENGTH,
                nullable=is_nullable(schema_field),
                is_primary=schema_field.is_id,
            )
        elif schema_field.type == "object":
            milvus_schema.add_field(
                field_name=field_name,
                datatype=DataType.JSON,
                nullable=is_nullable(schema_field),
            )
        else:
            milvus_data_type = get_milvus_type(schema_field)
            milvus_schema.add_field(
                field_name=field_name,
                datatype=milvus_data_type,
                nullable=is_nullable(schema_field),
            )

    if milvus_lite:
        if NULL_FIELD_NAME in milvus_schema.fields:
            raise BackendCompatibilityError(
                f"Milvus lite vector databases reserve the field '{NULL_FIELD_NAME}' for tracking null values"
            )
        else:
            milvus_schema.add_field(
                field_name=NULL_FIELD_NAME,
                datatype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=len(modaic_schema.as_dict()),
                max_length=255,
            )
    return milvus_schema


def process_value(value: Union[int, float, str], comparator: Comparator) -> str:
    """Convert a value to a string and add double quotes if it is a string.

    It required for comparators involving strings.

    Args:
        value: The value to convert.
        comparator: The comparator.

    Returns:
        The converted value as a string.
    """
    #
    if isinstance(value, str):
        if comparator is Comparator.LIKE:
            # If the comparator is LIKE, add a percent sign after it for prefix matching
            # and add double quotes
            return f'"{value}%"'
        else:
            # If the value is already a string, add double quotes
            return f'"{value}"'
    elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
        inside = ", ".join(f'"{v}"' for v in value)
        return f"[{inside}]"
    else:
        # If the value is not a string, convert it to a string without double quotes
        return str(value)
