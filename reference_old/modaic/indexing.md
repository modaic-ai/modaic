# Table of Contents

* [indexing](#modaic.indexing)
  * [Reranker](#modaic.indexing.Reranker)
    * [\_\_call\_\_](#modaic.indexing.Reranker.__call__)
  * [Embedder](#modaic.indexing.Embedder)

---
sidebar_label: indexing
title: modaic.indexing
---

## Reranker Objects

```python
class Reranker(ABC)
```

#### \_\_call\_\_

```python
def __call__(query: str,
             options: List[Context | Tuple[str, Context | ContextSchema]],
             k: int = 10,
             **kwargs) -> List[Tuple[float, Context | ContextSchema]]
```

Reranks the options based on the query.

**Arguments**:

- `query` - The query to rerank the options for.
- `options` - The options to rerank. Each option is a Context or tuple of (embedme_string, Context/ContextSchema).
- `k` - The number of options to return.
- `**kwargs` - Additional keyword arguments to pass to the reranker.
  

**Returns**:

  A list of tuples, where each tuple is (Context | ContextSchema, score). The Context or ContextSchema type depends on whichever was passed as an option for that index.

## Embedder Objects

```python
class Embedder(dspy.Embedder)
```

A wrapper around dspy.Embedder that automatically determines the output size of the model.

