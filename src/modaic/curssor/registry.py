from __future__ import annotations

from typing import Dict, Tuple, Type

from gqlalchemy.models import Node as GQLNode, Relationship as GQLRelationship
from pydantic import BaseModel


_node_cache: Dict[Type[BaseModel], Type[GQLNode]] = {}
_rel_cache: Dict[Tuple[str, str, str], Type[GQLRelationship]] = {}


def get_or_create_node_class(model_cls: Type[BaseModel]) -> Type[GQLNode]:
    """Return a dynamic GQLAlchemy Node subclass for a ContextSchema class.

    Params:
        model_cls: Pydantic BaseModel subclass (ContextSchema subclass)
    """
    from gqlalchemy.models import Node as _Node

    if model_cls in _node_cache:
        return _node_cache[model_cls]

    label = getattr(model_cls, "context_class", model_cls.__name__)

    class DynamicNode(_Node, label=label):  # type: ignore[misc]
        pass

    _node_cache[model_cls] = DynamicNode
    return DynamicNode


def get_or_create_relationship_class(
    type_name: str,
    start_label: str,
    end_label: str,
) -> Type[GQLRelationship]:
    """Return a dynamic GQLAlchemy Relationship subclass for a relation type.

    Params:
        type_name: Relationship type string, e.g., "HAS_CHUNK"
        start_label: Start node label
        end_label: End node label
    """
    from gqlalchemy.models import Relationship as _Relationship

    key = (type_name, start_label, end_label)
    if key in _rel_cache:
        return _rel_cache[key]

    class DynamicRel(_Relationship, type=type_name):  # type: ignore[misc]
        pass

    _rel_cache[key] = DynamicRel
    return DynamicRel

