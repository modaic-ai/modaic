import json # Still needed
from collections.abc import Mapping
from typing import Any, ClassVar, Dict, List, Literal, Optional, Type, Union

import numpy as np
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)
import weaviate
from weaviate.classes.config import Configure, Property, DataType as WeaviateDataType, VectorDistances
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.query import Filter, MetadataQuery

from ....context.base import Context
from ....exceptions import BackendCompatibilityError
from ....types import InnerField, Schema, SchemaField, float_format, int_format
from ..vector_database import DEFAULT_INDEX_NAME, IndexConfig, IndexType, SearchResult, VectorType


modaic_to_weaviate_vector = {
    VectorType.FLOAT: "float32",  # Weaviate's default
    VectorType.FLOAT16: "float16",
    VectorType.BFLOAT16: "bfloat16",
}

modaic_to_weaviate_index = {
    IndexType.DEFAULT: "hnsw",
    IndexType.HNSW: "hnsw",
    IndexType.FLAT: "flat",
}

modaic_metric_to_weaviate_enum = {
    "cosine": VectorDistances.COSINE,
    "l2": VectorDistances.L2_SQUARED,
    "ip": VectorDistances.DOT,
    "dot": VectorDistances.DOT,
    "euclidean": VectorDistances.L2_SQUARED,
}


class WeaviateTranslator(Visitor):
    """
    Translator to convert structured queries to Weaviate filters.
    """

    def visit_operation(self, operation: Operation) -> Filter:
        """Convert an operation (AND, OR, NOT) to a Weaviate filter."""
        args = [arg.accept(self) for arg in operation.arguments]
        
        if operation.operator == Operator.AND:
            return Filter.all_of(args)
        elif operation.operator == Operator.OR:
            return Filter.any_of(args)
        elif operation.operator == Operator.NOT:
            return Filter.not_(args[0])
        else:
            raise ValueError(f"Unsupported operator: {operation.operator}")

    def visit_comparison(self, comparison: Comparison) -> Filter:
        """Convert a comparison to a Weaviate filter."""
        attribute = comparison.attribute
        comparator = comparison.comparator
        value = comparison.value

        if comparator == Comparator.EQ:
            return Filter.by_property(attribute).equal(value)
        elif comparator == Comparator.NE:
            return Filter.by_property(attribute).not_equal(value)
        elif comparator == Comparator.GT:
            return Filter.by_property(attribute).greater_than(value)
        elif comparator == Comparator.GTE:
            return Filter.by_property(attribute).greater_or_equal(value)
        elif comparator == Comparator.LT:
            return Filter.by_property(attribute).less_than(value)
        elif comparator == Comparator.LTE:
            return Filter.by_property(attribute).less_or_equal(value)
        elif comparator == Comparator.IN:
            return Filter.by_property(attribute).contains_any(value)
        elif comparator == Comparator.CONTAIN:
            return Filter.by_property(attribute).contains_any([value])
        elif comparator == Comparator.LIKE:
            return Filter.by_property(attribute).like(f"*{value}*")
        else:
            raise ValueError(f"Unsupported comparator: {comparator}")

    def visit_structured_query(self, structured_query: StructuredQuery) -> Filter:
        """Convert a structured query to a Weaviate filter."""
        if structured_query.filter is None:
            raise ValueError("Structured query has no filter")
        return structured_query.filter.accept(self)


class WeaviateBackend:
    _name: ClassVar[Literal["weaviate"]] = "weaviate"
    mql_translator: Visitor = WeaviateTranslator()

    def __init__(
        self,
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        additional_config = None
        if timeout is not None:
            additional_config = AdditionalConfig(
                timeout=Timeout(init=timeout, query=timeout, insert=timeout)
            )
        
        if 'additional_config' in kwargs:
            pass
        elif additional_config is not None:
            kwargs['additional_config'] = additional_config
        
        if api_key:
            self._client = weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=weaviate.auth.AuthApiKey(api_key),
                **kwargs,
            )
        else:
            url_clean = url.replace("http://", "").replace("https://", "")
            if ":" in url_clean:
                host, port_str = url_clean.split(":", 1)
                port = int(port_str)
            else:
                host = url_clean
                port = 8080
            
            self._client = weaviate.connect_to_local(
                host=host,
                port=port,
                **kwargs,
            )

    def __del__(self):
        """Ensure client is closed on deletion."""
        if hasattr(self, '_client'):
            self._client.close()

    def create_record(self, embedding_map: Dict[str, np.ndarray], context: Context) -> Any:
        """
        Convert a Context to a record for Weaviate.
        
        This function serializes nested objects (like dicts or other Context models)
        into JSON strings to be stored in Weaviate's TEXT fields.
        """
        record_data = context.model_dump(include_hidden=True)
        record_id = record_data.pop('id')
        
        properties = {}
        for field_name, value in record_data.items():
            if isinstance(value, dict):
                properties[field_name] = json.dumps(value)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                properties[field_name] = [json.dumps(item) for item in value]
            else:
                properties[field_name] = value
        
        return {
            "properties": properties,
            "vector": embedding_map.get(DEFAULT_INDEX_NAME, list(embedding_map.values())[0]).tolist(),
            "uuid": record_id
        }

    def add_records(self, collection_name: str, records: List[Any]):
        collection = self._client.collections.get(collection_name)
        
        with collection.batch.dynamic() as batch:
            for record in records:
                batch.add_object(
                    properties=record["properties"],
                    vector=record["vector"],
                    uuid=record["uuid"]
                )

    def list_collections(self) -> List[str]:
        return list(self._client.collections.list_all().keys())

    def drop_collection(self, collection_name: str):
        self._client.collections.delete(collection_name)

    def create_collection(
        self,
        collection_name: str,
        payload_class: Type[Context],
        index: IndexConfig = IndexConfig(),
    ):
        if not issubclass(payload_class, Context):
            raise TypeError(f"Payload class {payload_class} must be a subclass of Context")

        properties = _modaic_to_weaviate_properties(payload_class.schema())

        try:
            vectorizer_config = None
            index_type = modaic_to_weaviate_index.get(index.index_type, "hnsw")
            
            if isinstance(index.metric, str):
                metric_name = index.metric.lower()
            else:
                metric_name = getattr(index.metric, 'name', 'cosine').lower()
            
            metric_enum = modaic_metric_to_weaviate_enum.get(metric_name, VectorDistances.COSINE)
            
            if index_type == "flat":
                vector_index_config = Configure.VectorIndex.flat(
                    distance_metric=metric_enum,
                )
            elif index_type == "dynamic":
                vector_index_config = Configure.VectorIndex.dynamic(
                    distance_metric=metric_enum,
                    threshold=10000,
                )
            else:
                vector_index_config = Configure.VectorIndex.hnsw(
                    distance_metric=metric_enum,
                )

        except (KeyError, AttributeError) as e:
            raise ValueError(f"Weaviate does not support the specified configuration: {e}") from None

        self._client.collections.create(
            name=collection_name,
            properties=properties,
            vectorizer_config=vectorizer_config,
            vector_index_config=vector_index_config,
        )

    def has_collection(self, collection_name: str) -> bool:
        return collection_name in self.list_collections()

    def _deserialize_properties(self, properties: Dict[str, Any], payload_class: Type[Context]) -> Dict[str, Any]:
        """
        Converts Weaviate properties back into a Pydantic-valid dictionary.
        - Parses JSON strings back into dictionaries/objects.
        """
        deserialized_props = dict(properties) 
        
        # Get the schema to check types
        schema_dict = payload_class.schema().as_dict()

        for field_name, value in deserialized_props.items():
            if field_name not in schema_dict:
                continue
            
            schema_field = schema_dict[field_name]
            
            # Check if it's a string that should be an object (e.g., dict or nested model)
            if isinstance(value, str) and schema_field.type == "object":
                try:
                    deserialized_props[field_name] = json.loads(value)
                except json.JSONDecodeError:
                    pass # Keep original string if not valid JSON
            
            # Check if it's a list of strings that should be a list of objects
            elif (isinstance(value, list) and 
                  schema_field.type == "array" and 
                  schema_field.inner_type.type == "object"):
                
                new_list = []
                for item in value:
                    if isinstance(item, str):
                        try:
                            new_list.append(json.loads(item))
                        except json.JSONDecodeError:
                            new_list.append(item) # Keep original
                    else:
                        new_list.append(item) # Already processed?
                deserialized_props[field_name] = new_list
        
        return deserialized_props

    def search(
        self,
        collection_name: str,
        vectors: List[np.ndarray],
        payload_class: Type[Context],
        k: int = 10,
        filter: Optional[Filter] = None,
    ) -> List[List[SearchResult]]:
        if not issubclass(payload_class, Context):
            raise TypeError(f"Payload class {payload_class} must be a subclass of Context")

        collection = self._client.collections.get(collection_name)
        all_results = []

        for vector in vectors:
            response = collection.query.near_vector(
                near_vector=vector.tolist(),
                limit=k,
                filters=filter,
                return_metadata=MetadataQuery(distance=True),
            )

            context_list = []
            for obj in response.objects:
                properties = self._deserialize_properties(obj.properties, payload_class)
                properties['id'] = str(obj.uuid) # Cast UUID to string

                context = payload_class.model_validate(properties)
                
                score = 1.0 - obj.metadata.distance
                
                context_list.append(SearchResult(id=context.id, context=context, score=score))

            all_results.append(context_list)

        return all_results

    def get_records(self, collection_name: str, payload_class: Type[Context], record_ids: List[str]) -> List[Context]:
        collection = self._client.collections.get(collection_name)
        records = []
        
        for record_id in record_ids:
            obj = collection.query.fetch_object_by_id(record_id)
            if obj:
                properties = self._deserialize_properties(obj.properties, payload_class)
                properties['id'] = str(obj.uuid) # Cast UUID to string
                
                records.append(payload_class.model_validate(properties))
        
        return records

    @staticmethod
    def from_local(host: str = "localhost", port: int = 8080) -> "WeaviateBackend":
        return WeaviateBackend(url=f"http://{host}:{port}")


def _modaic_to_weaviate_properties(modaic_schema: Schema) -> List[Property]:
    """
    Convert a Modaic schema to Weaviate properties.
    """
    type_mapping: Mapping[str, WeaviateDataType] = {
        "string": WeaviateDataType.TEXT,
        "integer": WeaviateDataType.INT,
        "number": WeaviateDataType.NUMBER,
        "boolean": WeaviateDataType.BOOL,
    }
    
    format_mapping: Mapping[int_format | float_format, WeaviateDataType] = {
        "int8": WeaviateDataType.INT,
        "int16": WeaviateDataType.INT,
        "int32": WeaviateDataType.INT,
        "int64": WeaviateDataType.INT,
        "float": WeaviateDataType.NUMBER,
        "double": WeaviateDataType.NUMBER,
    }

    properties = []
    
    for field_name, schema_field in modaic_schema.as_dict().items():
        if schema_field.is_id:
            continue
            
        index_searchable = False
            
        if schema_field.type == "array":
            if schema_field.inner_type.type == "string":
                data_type = WeaviateDataType.TEXT_ARRAY
                index_searchable = True
            elif schema_field.inner_type.type == "integer":
                data_type = WeaviateDataType.INT_ARRAY
            elif schema_field.inner_type.type == "number":
                data_type = WeaviateDataType.NUMBER_ARRAY
            elif schema_field.inner_type.type == "boolean":
                data_type = WeaviateDataType.BOOL_ARRAY
            elif schema_field.inner_type.type == "object":
                data_type = WeaviateDataType.TEXT_ARRAY # Store as array of JSON strings
            else:
                raise ValueError(f"Weaviate does not support array of type: {schema_field.inner_type.type}")

        elif schema_field.type == "object":
            data_type = WeaviateDataType.TEXT # Store as JSON string
            index_searchable = False 
        
        elif schema_field.format in format_mapping:
            data_type = format_mapping[schema_field.format]
        
        elif schema_field.type in type_mapping:
            data_type = type_mapping[schema_field.type]
            if data_type == WeaviateDataType.TEXT:
                index_searchable = True
        
        else:
            raise ValueError(f"Weaviate does not support field type: {schema_field.type}")
        
        properties.append(
            Property(
                name=field_name,
                data_type=data_type,
                skip_vectorization=True,
                index_filterable=True,
                index_searchable=index_searchable,
            )
        )
    
    return properties