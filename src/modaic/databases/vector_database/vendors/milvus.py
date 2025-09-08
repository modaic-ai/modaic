from collections.abc import Mapping
from typing import Any, ClassVar, Dict, List, Literal, Optional, Type

import numpy as np
from pydantic import BaseModel
from pymilvus import DataType, MilvusClient
from pymilvus.orm.collection import CollectionSchema

from ....context.base import Context
from ....exceptions import BackendCompatibilityError
from ....types import InnerField, Schema, SchemaField, float_format, int_format
from ..vector_database import IndexConfig, IndexType, SearchResult, VectorType

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


class MilvusBackend:
    _name: ClassVar[Literal["milvus"]] = "milvus"

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

            record["null"] = null_fields

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
            "field_name": index.name,
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
            field_name=index.name,
            index_name=f"{index.name}_index",
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
        filter: Optional[dict] = None,
    ) -> List[List[SearchResult]]:
        """
        Retrieve records from the vector database.
        """
        if not issubclass(payload_class, Context):
            raise TypeError(f"Payload class {payload_class} is must be a subclass of Context")

        output_fields = [field_name for field_name in payload_class.model_fields]
        listified_vectors = [vector.tolist() for vector in vectors]
        # Convert dict filter (MQL) to Milvus string expression if provided
        if isinstance(filter, dict):
            filter = mql_to_milvus(filter)

        searches = self._client.search(
            collection_name=collection_name,
            data=listified_vectors,
            limit=k,
            filter=filter,
            anns_field="vector",
            output_fields=output_fields,
        )
        all_results = []
        for search in searches:
            context_list = []
            for result in search:
                match result:
                    case {"id": id, "distance": distance, "entity": entity}:
                        context_list.append(
                            {
                                "id": id,
                                "distance": distance,
                                "context": payload_class.model_validate(self._process_null(entity)),
                            }
                        )
                    case _:
                        raise ValueError(f"Failed to parse search results to {payload_class.__name__}: {result}")
            all_results.append(context_list)
        return all_results

    def get_records(self, collection_name: str, payload_class: Type[Context], record_ids: List[str]) -> Context:
        output_fields = [field_name for field_name in payload_class.model_fields]
        records = self._client.get(collection_name=collection_name, ids=record_ids, output_fields=output_fields)
        return [payload_class.model_validate(self._process_null(record)) for record in records]

    @staticmethod
    def from_local(file_path: str) -> "MilvusBackend":
        return MilvusBackend(uri=file_path)

    def _process_null(self, record: dict) -> dict:
        if self.milvus_lite and "null" in record:
            for field_name in record["null"]:
                record[field_name] = None
            del record["null"]
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
        if "null" in milvus_schema.fields:
            raise BackendCompatibilityError(
                "Milvus lite vector databases reserve the field 'null' for tracking null values"
            )
        else:
            milvus_schema.add_field(
                field_name="null",
                datatype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=len(modaic_schema.as_dict()),
                max_length=255,
            )
    return milvus_schema


def mql_to_milvus(mql: Dict[str, Any]) -> str:
    """
    Convert a Modaic Query Language (MQL) filter into a Milvus boolean expression string.

    Args:
        mql: A dictionary representing the MQL filter. Supports logical operators
            like `$and`, `$or`, `$not` and comparison operators like `$eq`, `$ne`,
            `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$like`, `$exists`.
            JSON nested fields can be expressed using dot notation, e.g.,
            `product.model`.

    Returns:
        The equivalent Milvus filter expression string.
    """

    def format_identifier(identifier: str) -> str:
        """Format an identifier for Milvus. Supports JSON path using dot notation.

        Args:
            identifier: Field identifier. Dotted paths (e.g., "product.model") are
                converted to Milvus JSON accessors (e.g., product["model"]).

        Returns:
            The Milvus-compatible identifier string.
        """
        if "." not in identifier:
            return identifier
        head, *rest = identifier.split(".")
        json_path = head
        for key in rest:
            json_path += f'["{key}"]'
        return json_path

    def format_value(value: Any) -> str:
        """Format a Python value into a Milvus filter literal.

        Args:
            value: The value to format.

        Returns:
            A string literal usable in a Milvus filter expression.
        """
        if isinstance(value, str):
            return f'"{value}"'
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "NULL"
        if isinstance(value, (list, tuple)):
            inner = ", ".join(format_value(v) for v in value)
            return f"[{inner}]"
        return str(value)

    def join_with(op: str, parts: List[str]) -> str:
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return " (" + f" {op} ".join(parts) + ") "

    def parse_expr(node: Any) -> str:
        """Parse an $expr expression tree into a Milvus expression string.

        Supports arithmetic: $add, $sub, $mul, $div, $mod, $pow, $neg
        Supports comparisons: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $like
        Field references should be strings beginning with '$'.
        """
        # Field reference like "$field" or "$product.model"
        if isinstance(node, str) and node.startswith("$"):
            return format_identifier(node[1:])
        # Primitive literal
        if not isinstance(node, (dict, list, tuple)):
            return format_value(node)
        # List literal (e.g., in RHS of $in)
        if isinstance(node, (list, tuple)):
            return "[" + ", ".join(parse_expr(x) for x in node) + "]"

        if isinstance(node, dict):
            if len(node) != 1:
                # Combine multiple nodes with AND by default
                return join_with("AND", [parse_expr({k: v}) for k, v in node.items()])
            ((op, val),) = node.items()
            # Arithmetic operators may take 1..n args
            if op in ("$add", "$sub", "$mul", "$div", "$mod", "$pow"):
                args = val if isinstance(val, (list, tuple)) else [val]
                parsed = [parse_expr(a) for a in args]
                if op == "$add":
                    return "(" + " + ".join(parsed) + ")"
                if op == "$mul":
                    return "(" + " * ".join(parsed) + ")"
                # Binary-only
                if len(parsed) != 2:
                    raise ValueError(f"Operator {op} expects 2 arguments")
                a, b = parsed
                if op == "$sub":
                    return f"({a} - {b})"
                if op == "$div":
                    return f"({a} / {b})"
                if op == "$mod":
                    return f"({a} % {b})"
                if op == "$pow":
                    return f"({a} ** {b})"
            if op == "$neg":
                inner = parse_expr(val)
                return f"-({inner})"

            # Comparison operators inside $expr
            if op in ("$eq", "$ne", "$gt", "$gte", "$lt", "$lte"):
                if not isinstance(val, (list, tuple)) or len(val) != 2:
                    raise ValueError(f"{op} expects two operands in $expr")
                left = parse_expr(val[0])
                right = parse_expr(val[1])
                ops_map = {
                    "$eq": "==",
                    "$ne": "!=",
                    "$gt": ">",
                    "$gte": ">=",
                    "$lt": "<",
                    "$lte": "<=",
                }
                return f"({left} {ops_map[op]} {right})"
            if op == "$in":
                if not isinstance(val, (list, tuple)) or len(val) != 2:
                    raise ValueError("$in expects [expr, array] in $expr")
                left = parse_expr(val[0])
                right = parse_expr(val[1])
                return f"({left} in {right})"
            if op == "$nin":
                if not isinstance(val, (list, tuple)) or len(val) != 2:
                    raise ValueError("$nin expects [expr, array] in $expr")
                left = parse_expr(val[0])
                right = parse_expr(val[1])
                return f"NOT ({left} in {right})"
            if op == "$like":
                if not isinstance(val, (list, tuple)) or len(val) != 2:
                    raise ValueError("$like expects [expr, pattern] in $expr")
                left = parse_expr(val[0])
                right = parse_expr(val[1])
                return f"({left} like {right})"
            # Fallback: treat as field-op style
            return parse_expr(val)

        raise ValueError("Invalid $expr node")

    def parse_field(field: str, condition: Any) -> str:
        field_expr = format_identifier(field)
        if isinstance(condition, dict):
            subparts: List[str] = []
            for op, val in condition.items():
                match op:
                    case "$eq":
                        rhs = parse_expr(val) if isinstance(val, (dict, list, tuple)) else format_value(val)
                        subparts.append(f"{field_expr} == {rhs}")
                    case "$ne":
                        rhs = parse_expr(val) if isinstance(val, (dict, list, tuple)) else format_value(val)
                        subparts.append(f"{field_expr} != {rhs}")
                    case "$gt":
                        rhs = parse_expr(val) if isinstance(val, (dict, list, tuple)) else format_value(val)
                        subparts.append(f"{field_expr} > {rhs}")
                    case "$gte":
                        rhs = parse_expr(val) if isinstance(val, (dict, list, tuple)) else format_value(val)
                        subparts.append(f"{field_expr} >= {rhs}")
                    case "$lt":
                        rhs = parse_expr(val) if isinstance(val, (dict, list, tuple)) else format_value(val)
                        subparts.append(f"{field_expr} < {rhs}")
                    case "$lte":
                        rhs = parse_expr(val) if isinstance(val, (dict, list, tuple)) else format_value(val)
                        subparts.append(f"{field_expr} <= {rhs}")
                    case "$in":
                        rhs = parse_expr(val) if isinstance(val, (dict, list, tuple)) else format_value(val)
                        subparts.append(f"{field_expr} in {rhs}")
                    case "$nin":
                        rhs = parse_expr(val) if isinstance(val, (dict, list, tuple)) else format_value(val)
                        subparts.append(f"NOT ({field_expr} in {rhs})")
                    case "$like":
                        rhs = parse_expr(val) if isinstance(val, (dict, list, tuple)) else format_value(val)
                        subparts.append(f"{field_expr} like {rhs}")
                    case "$exists":
                        if bool(val):
                            subparts.append(f"{field_expr} IS NOT NULL")
                        else:
                            subparts.append(f"{field_expr} IS NULL")
                    case _:
                        raise ValueError(f"Unsupported MQL operator: {op}")
            return join_with("AND", subparts).strip()
        # Implicit equality
        return f"{field_expr} == {format_value(condition)}"

    def parse_mql(node: Any) -> str:
        if isinstance(node, dict):
            parts: List[str] = []
            for key, val in node.items():
                match key:
                    case "$and":
                        if not isinstance(val, list):
                            raise ValueError("$and expects a list")
                        parts.append(join_with("AND", [parse_mql(v) for v in val]).strip())
                    case "$or":
                        if not isinstance(val, list):
                            raise ValueError("$or expects a list")
                        parts.append(join_with("OR", [parse_mql(v) for v in val]).strip())
                    case "$not":
                        parts.append("NOT (" + parse_mql(val).strip() + ")")
                    case "$expr":
                        parts.append(parse_expr(val))
                    case _:
                        parts.append(parse_field(key, val))
            return join_with("AND", parts).strip()
        elif isinstance(node, list):
            return join_with("AND", [parse_mql(v) for v in node]).strip()
        else:
            raise ValueError("Invalid MQL structure: expected dict or list at root")

    return parse_mql(mql)
