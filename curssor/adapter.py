from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

from pydantic import BaseModel

from .registry import get_or_create_node_class, get_or_create_relationship_class


def _flatten_pydantic(model: BaseModel) -> Dict[str, Any]:
    """Flatten a Pydantic model to a dict suitable for Node properties.

    - Scalars stay as-is
    - Nested BaseModel -> dict
    - Lists of BaseModels -> list of dicts
    - Exclude large binary/image types (left to caller)

    Params:
        model: Pydantic BaseModel instance

    Returns:
        A plain dictionary of properties
    """
    def convert(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return {k: convert(v) for k, v in value.model_dump().items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    return {k: convert(v) for k, v in model.model_dump().items()}


def contextschema_to_node(schema: BaseModel) -> Any:
    """Create a GQLAlchemy Node from a ContextSchema instance.

    Params:
        schema: ContextSchema (Pydantic BaseModel) instance

    Returns:
        GQLAlchemy Node instance with label derived from schema.context_class
    """
    node_cls = get_or_create_node_class(type(schema))
    props = _flatten_pydantic(schema)
    # Remove fields that GQLAlchemy uses internally or that are non-serializable
    props.pop("_id", None)
    props.pop("_labels", None)
    return node_cls(**props)


def relationship_between(
    start_schema: BaseModel,
    end_schema: BaseModel,
    rel_type: str,
    properties: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any, Any]:
    """Create a start Node, end Node, and Relationship between them.

    Params:
        start_schema: Source ContextSchema instance
        end_schema: Target ContextSchema instance
        rel_type: Relationship type (e.g., "HAS_CHUNK")
        properties: Optional relationship properties

    Returns:
        (start_node, relationship, end_node)
    """
    start_node = contextschema_to_node(start_schema)
    end_node = contextschema_to_node(end_schema)

    start_label = getattr(type(start_schema), "context_class", type(start_schema).__name__)
    end_label = getattr(type(end_schema), "context_class", type(end_schema).__name__)
    rel_cls = get_or_create_relationship_class(rel_type, start_label, end_label)

    rel_props = {} if properties is None else dict(properties)
    relationship = rel_cls(**rel_props)
    # Attach start/end IDs later once persisted; caller/DB layer will wire IDs
    return start_node, relationship, end_node

