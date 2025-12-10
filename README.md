[![Docs](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://docs.modaic.dev)
[![PyPI](https://img.shields.io/pypi/v/modaic)](https://pypi.org/project/modaic/)


# Modaic 🐙
**Mod**ular **A**gent **I**nfrastructure **C**ollection, a Python framework for maintaining DSPy applications.

## Overview

Modaic provides a comprehensive toolkit for creating intelligent DSPY pipelines that can work with diverse data sources including tables, documents, and databases. Built on top of DSPy, it offers a way to share and manage DSPY pipelines with integrated vector, SQL, and graph database support.

## Key Features

- **Hub Support**: Load and share precompiled DSPY programs from Modaic Hub
- **Context Management**: Structured handling of molecular and atomic context types
- **Database Integration**: Support for Vector (Milvus, Pinecone, Qdrant), SQL (SQLite, MySQL, PostgreSQL), and Graph (Memgraph, Neo4j)
- **Program Framework**: Precompiled and auto-loading DSPY programs
- **Table Processing**: Advanced Excel/CSV processing with SQL querying capabilities


## Installation

### Using uv (recommended)

```bash
uv add modaic
```

Optional (for hub operations):

```bash
export MODAIC_TOKEN="<your-token>"
```

### Using pip
Please note that you will not be able to push DSPY programs to the Modaic Hub with pip.
```bash
pip install modaic
```
## Quick Start

### Creating a Simple Program

```python
from modaic import PrecompiledProgram, PrecompiledConfig

class WeatherConfig(PrecompiledConfig):
    weather: str = "sunny"

class WeatherProgram(PrecompiledProgram):
    config: WeatherConfig

    def __init__(self, config: WeatherConfig, **kwargs):
        super().__init__(config, **kwargs)

    def forward(self, query: str) -> str:
        return f"The weather in {query} is {self.config.weather}."

weather_program = WeatherProgram(WeatherConfig())
print(weather_program(query="Tokyo"))
```

Save and load locally:

```python
weather_program.save_precompiled("./my-weather")

from modaic import AutoProgram, AutoConfig

cfg = AutoConfig.from_precompiled("./my-weather", local=True)
loaded = AutoProgram.from_precompiled("./my-weather", local=True)
print(loaded(query="Kyoto"))
```

### Working with Tables

```python
from pathlib import Path
from modaic.context import Table, TableFile
import pandas as pd

# Load from Excel/CSV
excel = TableFile.from_file(
    file_ref="employees.xlsx",
    file=Path("employees.xlsx"),
    file_type="xlsx",
)
csv = TableFile.from_file(
    file_ref="data.csv",
    file=Path("data.csv"),
    file_type="csv",
)

# Create from DataFrame
df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
table = Table(df=df, name="my_table")

# Query with SQL (refer to in-memory table as `this`)
result = table.query("SELECT * FROM this WHERE col1 > 1")

# Convert to markdown
markdown = table.markdown()
```

### Database Integration

#### SQL Database
```python
from modaic.databases import SQLDatabase, SQLiteBackend

# Configure and connect
backend = SQLiteBackend(db_path="my_database.db")
db = SQLDatabase(backend)

# Add table
db.add_table(table)

# Query
rows = db.fetchall("SELECT * FROM my_table")
```

#### Vector Database
#### Graph Database
```python
from modaic.context import Context, Relation
from modaic.databases import GraphDatabase, MemgraphConfig, Neo4jConfig

# Configure backend (choose one)
mg = GraphDatabase(MemgraphConfig())
# or
neo = GraphDatabase(Neo4jConfig())

# Define nodes
class Person(Context):
    name: str
    age: int

class KNOWS(Relation):
    since: int

alice = Person(name="Alice", age=30)
bob = Person(name="Bob", age=28)

# Save nodes
alice.save(mg)
bob.save(mg)

# Create relationship (Alice)-[KNOWS]->(Bob)
rel = (alice >> KNOWS(since=2020) >> bob)
rel.save(mg)

# Query
rows = mg.execute_and_fetch("MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a, r, b LIMIT 5")
```
```python
from modaic import Embedder
from modaic.context import Text
from modaic.databases import VectorDatabase, MilvusBackend

# Setup embedder and backend
embedder = Embedder("openai/text-embedding-3-small")
backend = MilvusBackend.from_local("vector.db")  # milvus lite

# Initialize database
vdb = VectorDatabase(backend=backend, embedder=embedder, payload_class=Text)

# Create collection and add records
vdb.create_collection("my_collection", payload_class=Text)
vdb.add_records("my_collection", [Text(text="hello world"), Text(text="modaic makes sharing DSPY programs easy")])

# Search
results = vdb.search("my_collection", query="hello", k=3)
top_hit_text = results[0][0].context.text
```

## Architecture
### Program Types

1. **PrecompiledProgram**: Statically defined programs with explicit configuration
2. **AutoProgram**: Dynamically loaded programs from Modaic Hub or local repositories

### Database Support

| Database Type | Providers | Use Case |
|---------------|-----------|----------|
| **Vector** | Milvus | Semantic search, RAG |
| **SQL** | SQLite, MySQL, PostgreSQL | Structured queries, table storage |

## Examples

### TableRAG Example

The TableRAG example demonstrates a complete RAG pipeline for table-based question answering:

```python
from modaic import PrecompiledConfig, PrecompiledProgram
from modaic.context import Table
from modaic.databases import VectorDatabase, SQLDatabase
from modaic.types import Indexer

class TableRAGConfig(PrecompiledConfig):
    k_recall: int = 50
    k_rerank: int = 5

class TableRAGProgram(PrecompiledProgram):
    config: TableRAGConfig # ! Important: config must be annotated with the config class

    def __init__(self, config: TableRAGConfig, indexer: Indexer, **kwargs):
        super().__init__(config, **kwargs)
        self.indexer = indexer
        # Initialize DSPy modules for reasoning

    def forward(self, user_query: str) -> str:
        # Retrieve relevant tables
        # Generate SQL queries
        # Combine results and provide answer
        pass
```

## Support

For issues and questions:
- GitHub Issues: `https://github.com/modaic-ai/modaic/issues`
- Docs: `https://docs.modaic.dev`
