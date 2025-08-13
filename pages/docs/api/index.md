# Modaic

A Python framework for building AI agents with structured context management, database integration, and retrieval-augmented generation (RAG) capabilities.

## Overview

Modaic provides a comprehensive toolkit for creating intelligent agents that can work with diverse data sources including tables, documents, and databases. Built on top of DSPy, it offers both precompiled and auto-loading agent architectures with integrated vector and SQL database support.

## Key Features

- **Context Management**: Structured handling of molecular and atomic context types
- **Database Integration**: Support for Vector (Milvus, Pinecone, Qdrant) and SQL databases (SQLite, MySQL, PostgreSQL)
- **Agent Framework**: Precompiled and auto-loading agent architectures
- **Table Processing**: Advanced Excel/CSV processing with SQL querying capabilities
- **RAG Pipeline**: Built-in retrieval-augmented generation with reranking
- **Hub Support**: Load and share precompiled agents from Modaic Hub

## Installation

```bash
pip install modaic
```

### Development Installation

```bash
git clone <repository-url>
cd modaic
pip install -e .
```

### Dependencies

**Core Dependencies:**
- `dspy>=2.6.27` - Framework foundation
- `duckdb>=1.3.2` - In-memory SQL processing
- `openpyxl>=3.1.5` - Excel file support
- `pillow>=11.3.0` - Image processing
- `pymilvus>=2.5.14` - Vector database support

**Development Dependencies:**
- `pytest>=8.4.1` - Testing framework
- `ruff>=0.12.7` - Code linting
- `mkdocs-material>=9.6.16` - Documentation

## Quick Start

### Creating a Simple Agent

```python
from modaic import PrecompiledAgent, PrecompiledConfig

class WeatherConfig(PrecompiledConfig):
    agent_type = "WeatherAgent"

class WeatherAgent(PrecompiledAgent):
    config_class = WeatherConfig
    
    def __init__(self, config: WeatherConfig, **kwargs):
        super().__init__(config, **kwargs)

    def forward(self, query: str) -> str:
        return f"Weather information for: {query}"

# Create and use the agent
config = WeatherConfig()
agent = WeatherAgent(config)
result = agent.forward("What's the weather in Tokyo?")
```

### Working with Tables

```python
from modaic.context import Table
import pandas as pd

# Load from Excel/CSV
table = Table.from_excel("data.xlsx")
table = Table.from_csv("data.csv")

# Create from DataFrame
df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
table = Table(df, name="my_table")

# Query with SQL
result = table.query("SELECT * FROM this WHERE col1 > 1")

# Convert to markdown
markdown = table.markdown()
```

### Database Integration

#### SQL Database
```python
from modaic.databases import SQLDatabase, SQLiteConfig

# Configure and connect
config = SQLiteConfig(db_path="my_database.db")
db = SQLDatabase(config)

# Add table
db.add_table(table)

# Query
result = db.fetchall("SELECT * FROM my_table")
```

#### Vector Database
```python
from modaic.databases import VectorDatabase, MilvusVDBConfig
from modaic.context import Text
import dspy

# Setup embedder and config
embedder = dspy.Embedder(model="openai/text-embedding-3-small")
config = MilvusVDBConfig.from_local("vector.db")

# Initialize database
vdb = VectorDatabase(config, embedder, Text.serialized_context_class)

# Create collection and add records
vdb.create_collection("my_collection", Text.serialized_context_class)
# Add your text records here
```

## Architecture

### Context Types

Modaic organizes data into two main context types:

- **Molecular Context**: Complex data structures (Tables, MultiTabbedTables)
- **Atomic Context**: Simple text units that can be embedded and retrieved

### Agent Types

1. **PrecompiledAgent**: Statically defined agents with explicit configuration
2. **AutoAgent**: Dynamically loaded agents from Modaic Hub or local repositories

### Database Support

| Database Type | Providers | Use Case |
|---------------|-----------|----------|
| **Vector** | Milvus, Pinecone, Qdrant | Semantic search, RAG |
| **SQL** | SQLite, MySQL, PostgreSQL | Structured queries, table storage |

## Examples

### TableRAG Example

The TableRAG example demonstrates a complete RAG pipeline for table-based question answering:

```python
from modaic.precompiled_agent import PrecompiledConfig, PrecompiledAgent
from modaic.context import Table
from modaic.databases import VectorDatabase, SQLDatabase
from modaic.types import Indexer

class TableRAGConfig(PrecompiledConfig):
    agent_type = "TableRAGAgent"
    k_recall: int = 50
    k_rerank: int = 5

class TableRAGAgent(PrecompiledAgent):
    config_class = TableRAGConfig
    
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

## Testing

Run the test suite:

```bash
pytest tests/
```

Key test modules:
- `test_table.py` - Table context functionality
- `test_sql_database.py` - SQL database operations  
- `test_vectordb.py` - Vector database operations
- `test_autoagent.py` - Agent loading and execution

## Documentation

Build and serve documentation locally:

```bash
mkdocs serve
```

## Development

### Code Quality

The project uses Ruff for linting:

```bash
ruff check src/
ruff format src/
```

### Project Structure

```
modaic/
├── src/modaic/           # Main package
│   ├── context/          # Context management (Table, Text, etc.)
│   ├── databases/        # Database integrations
│   ├── utils/            # Utilities (reranker, etc.)
│   ├── storage/          # Context storage
│   ├── auto_agent.py     # Dynamic agent loading
│   └── precompiled_agent.py  # Static agent framework
├── examples/             # Usage examples
├── tests/                # Test suite
└── docs/                 # Documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[License information to be added]

## Documentation Deployment

This project includes automated documentation deployment. See [README_DEPLOYMENT.md](README_DEPLOYMENT.md) for details on:
- GitHub workflow configuration
- External repository integration
- Manual and automated triggers
- Troubleshooting deployment issues

## Support

For issues and questions:
- GitHub Issues: [Link to issues]
- Documentation: [Link to docs]