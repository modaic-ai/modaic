### GQLAlchemy integration: design, changes, and examples

This directory contains a thin, optional integration layer that adapts `modaic` `ContextSchema` objects (Pydantic models defined under `modaic.context`) to `gqlalchemy` graph primitives (`Node`, `Relationship`). The integration aims to make any `ContextSchema` usable as a graph node and allow easy creation of relationships between schemas, without coupling `modaic` core to a specific graph backend.

What changed:

- Added `gqlalchemy` dependency to the project using `uv add gqlalchemy`.
- Created a new package at `src/modaic/curssor/` with:
  - `registry.py`: dynamic class factories and caches for `Node` and `Relationship` subclasses.
  - `adapter.py`: helpers to convert `ContextSchema` instances into `gqlalchemy` `Node` objects and to create `Relationship` instances between two schemas.
  - `demo.py`: a small, DB-free demo that maps `LongTextSchema` and `TextSchema` to nodes and builds a `HAS_CHUNK` relationship.
  - `__init__.py`: re-exports for ergonomic imports.

How it works (high-level):

- Dynamic Node classes: for any `ContextSchema` subclass, we create a corresponding `gqlalchemy.Node` subclass on the fly. The `label` is set to `schema.context_class` (if present) or the Pydantic class name.
- Dynamic Relationship classes: for any edge type (e.g., `HAS_CHUNK`), we create a `gqlalchemy.Relationship` subclass on demand.
- Property mapping: a schema instance is flattened into a plain dict (`pydantic.BaseModel.model_dump()` plus recursive conversion of nested BaseModels and lists). The result is used as Node properties. This keeps the adapter generic and faithful to the serialized form you already use elsewhere.
- No DB required: the adapter returns in-memory `Node` and `Relationship` objects; you can optionally persist them with a running Memgraph or Neo4j via `gqlalchemy`.

Design details and rationale:

- Separation of concerns: `modaic` types remain unchanged. The integration layer converts them at the boundary to graph objects.
- Dynamic vs static classes: using runtime generation avoids boilerplate and keeps things aligned as you add new `ContextSchema` types.
- Extensibility: future enhancements like index/constraint hints can be based on Pydantic field metadata or decorators without modifying existing schemas.
- Safety: we flatten nested models so graph properties remain JSON-like types. Large or binary payloads should be excluded at the caller level if needed.

Key files:

- `src/modaic/curssor/adapter.py`:
  - `contextschema_to_node(schema) -> Node`
  - `relationship_between(start_schema, end_schema, rel_type, properties=None) -> (start_node, relationship, end_node)`
- `src/modaic/curssor/registry.py`:
  - `get_or_create_node_class(model_cls)`
  - `get_or_create_relationship_class(type_name, start_label, end_label)`
- `src/modaic/curssor/demo.py`: runnable example using `LongTextSchema`/`TextSchema`.

Run the demo (no DB):

```bash
cd modaic
uv run python -m modaic.curssor.demo
```

You should see a `DynamicNode` for `LongText`, a `DynamicNode` for `Text`, and a `DynamicRel` for `HAS_CHUNK` printed with their properties.

Examples

1. Convert any ContextSchema into a Node

```python
from modaic.curssor import contextschema_to_node
from modaic.context.text import TextSchema
from modaic.context.base import Source

schema = TextSchema(source=Source(), metadata={}, text="hello")
node = contextschema_to_node(schema)
print(type(node).__name__, node)
```

2. Create a relationship between two schemas

```python
from modaic.curssor import relationship_between
from modaic.context.text import LongTextSchema, TextSchema
from modaic.context.base import Source

src = Source()
parent = LongTextSchema(source=src, metadata={}, text="Hello world", chunks=[])
child = TextSchema(source=src, metadata={}, text="Hello")

start_node, rel, end_node = relationship_between(parent, child, "HAS_CHUNK", {"order": 0})
print(type(start_node).__name__, rel.type, type(end_node).__name__)
```

3. Optional: Persist nodes and relationships to a running Memgraph instance

```python
from gqlalchemy import Memgraph
from gqlalchemy.models import Node, Relationship
from modaic.curssor import contextschema_to_node, relationship_between
from modaic.context.text import LongTextSchema, TextSchema
from modaic.context.base import Source

# 1) Start Memgraph separately (e.g., docker run memgraph/memgraph)
# 2) Connect
db = Memgraph(host="127.0.0.1", port=7687, username="", password="")

src = Source()
parent = LongTextSchema(source=src, metadata={}, text="Hello world", chunks=[])
child = TextSchema(source=src, metadata={}, text="Hello")

# Convert to in-memory graph objects
parent_node = contextschema_to_node(parent)
child_node = contextschema_to_node(child)

# Persist nodes; Memgraph returns stored nodes with IDs
saved_parent = db.save_node(parent_node)
saved_child = db.save_node(child_node)

# Create relationship type and persist
start_node, rel, end_node = relationship_between(parent, child, "HAS_CHUNK", {"order": 0})
# Wire IDs
rel._start_node_id = saved_parent._id
rel._end_node_id = saved_child._id
db.save_relationship(rel)
```

4. Integrating with existing Modaic flows (TableRAG example idea)

- Chunking produces `Context` objects with `ContextSchema` payloads.
- Use `contextschema_to_node` on each chunk’s schema to construct nodes.
- Create edges to represent containment (`HAS_CHUNK`) or semantic links (`RELATED_TO`).
- Persist to your graph for downstream traversal, neighborhood expansion, or graph-based RAG.

Planned enhancements

- Index/constraint hints from Pydantic field metadata (e.g., mark unique or indexed fields)
- Configurable property filtering/coercion for non-primitive types
- Helpers for bulk export: serialize a `Context` tree/graph directly to `Node`/`Relationship` batches
- Convenience functions for round-trip conversions (graph -> `ContextSchema`)

Notes

- The adapter uses each schema’s serialized shape as properties; by default this includes `id`, `source`, and `metadata`. Adjust or filter at call-site if needed.
- Large/binary/image payloads should be stored externally or referenced, not embedded in graph properties.
