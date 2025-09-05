from typing import (
    Type,
    Any,
    ClassVar,
    Optional,
    List,
    Dict,
    Literal,
)
from pymilvus import DataType, MilvusClient
from pydantic import BaseModel
from ..vector_database import (
    IndexType,
    IndexConfig,
    VectorType,
)
from ....context.base import Context
import numpy as np
from dataclasses import dataclass, field
from ....types import SchemaField, Modaic_Type
from collections.abc import Mapping
from ..vector_database import SearchResult


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
        self._client = MilvusClient(
            uri=uri,
            user=user,
            password=password,
            db_name=db_name,
            token=token,
            timeout=timeout,
            **kwargs,
        )

    def create_record(self, embedding_map: Dict[str, np.ndarray], scontext: Context) -> Any:
        """
        Convert a Context to a record for Milvus.
        """
        record = scontext.dump_all(mode="json")
        for index_name, embedding in embedding_map.items():
            record[index_name] = embedding
        return record

    def add_records(self, collection_name: str, records: List[Any]):
        """
        Add records to a Milvus collection.
        """
        self._client.insert(collection_name, records)

    def drop_collection(self, collection_name: str):
        """
        Drop a Milvus collection.
        """
        self._client.drop_collection(collection_name)

    def create_collection(
        self,
        collection_name: str,
        payload_schema: Dict[str, SchemaField],
        index: List[IndexConfig] = IndexConfig(),
    ):
        """
        Create a Milvus collection.
        """

        schema = _modaic_to_milvus_schema(self._client, payload_schema)
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
                raise ValueError(f"Milvus does not support vector type: {index_config.vector_type}")
            kwargs = {
                "field_name": index_config.name,
                "datatype": vector_type,
            }
            # sparse vectors don't have a dim in milvus
            if index_config.vector_type != VectorType.FLOAT_SPARSE:
                # sparse vectors don't have a dim in milvus
                kwargs["dim"] = index_config.embedder.embedding_dim
            schema.add_field(**kwargs)

        index_params = self._client.prepare_index_params()
        index_type = index_config.index_type.name if index_config.index_type != IndexType.DEFAULT else "AUTOINDEX"
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

        self._client.create_collection(collection_name, schema=schema, index_params=index_params)

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

    def search(
        self,
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
        # Convert dict filter (MQL) to Milvus string expression if provided
        if isinstance(filter, dict):
            filter = mql_to_milvus(filter)

        results = self._client.search(
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
                    raise ValueError(f"Failed to parse search results to {payload_schema.__name__}: {result}")
        return context_list

    @staticmethod
    def from_local(file_path: str):
        return MilvusBackend(uri=file_path)


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
            if field_type == "int64" or field_type == "int32":  # CAVEAT: Milvus only accepts int64 for id
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
                raise ValueError(f"Milvus does not support inner type {inner_type.type} for Array field: {field_name}")
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
            raise ValueError(f"Unsupported field type for Milvus - {field_name}: {field_type}")
    return milvus_schema


def mql_to_milvus(mql: Dict[str, Any]) -> str:
    """
    Convert a Modaic Query Language (MQL) filter into a Milvus boolean expression string.

    Params:
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

        Params:
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

        Params:
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
