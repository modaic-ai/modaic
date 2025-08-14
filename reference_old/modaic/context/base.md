# Table of Contents

* [base](#modaic.context.base)
  * [Source](#modaic.context.base.Source)
    * [model\_dump](#modaic.context.base.Source.model_dump)
    * [model\_dump\_json](#modaic.context.base.Source.model_dump_json)
  * [ContextSchema](#modaic.context.base.ContextSchema)
  * [Context](#modaic.context.base.Context)
    * [\_\_init\_\_](#modaic.context.base.Context.__init__)
    * [embedme](#modaic.context.base.Context.embedme)
    * [readme](#modaic.context.base.Context.readme)
    * [serialize](#modaic.context.base.Context.serialize)
    * [deserialize](#modaic.context.base.Context.deserialize)
    * [set\_source](#modaic.context.base.Context.set_source)
    * [set\_metadata](#modaic.context.base.Context.set_metadata)
    * [add\_metadata](#modaic.context.base.Context.add_metadata)
    * [from\_dict](#modaic.context.base.Context.from_dict)
  * [Atomic](#modaic.context.base.Atomic)
  * [Molecular](#modaic.context.base.Molecular)
    * [chunk\_with](#modaic.context.base.Molecular.chunk_with)
    * [apply\_to\_chunks](#modaic.context.base.Molecular.apply_to_chunks)

---
sidebar_label: base
title: modaic.context.base
---

## Source Objects

```python
class Source(BaseModel)
```

#### model\_dump

```python
def model_dump(**kwargs)
```

Override model_dump method to exclude _parent field

#### model\_dump\_json

```python
def model_dump_json(**kwargs)
```

Override model_dump_json method to exclude _parent field

## ContextSchema Objects

```python
class ContextSchema(BaseModel, metaclass=ContextSchemaMeta)
```

Base class used to define the schema of a context object when they are serialized.

**Attributes**:

- `context_class` - The class of the context object that this serialized context is for.
- `id` - The id of the serialized context.
- `source` - The source of the context object.
- `metadata` - The metadata of the context object.
  

**Example**:

  In this example, `CaptionedImageSchema` stores the caption and the caption embedding while `CaptionedImage` is the `Context` class that is used to store the context object.
  Note that the image is loaded dynamically in the `CaptionedImage` class and is not serialized to `CaptionedImageSchema`.
    ```python
    from modaic.context import ContextSchema
    from modaic.types import String, Vector, Float16Vector

    class CaptionedImageSchema(ContextSchema):
        caption: String[100]
        caption_embedding: Float16Vector[384]
        image_path: String[100]

    class CaptionedImage(Atomic):
        schema = CaptionedImageSchema

        def __init__(self, image_path: str, caption: str, caption_embedding: np.ndarray, **kwargs):
            super().__init__(**kwargs)
            self.caption = caption
            self.caption_embedding = caption_embedding
            self.image_path = image_path
            self.image = PIL.Image.open(image_path)

        def embedme(self) -> PIL.Image.Image:
            return self.image
    ```

## Context Objects

```python
class Context(ABC)
```

#### \_\_init\_\_

```python
def __init__(source: Optional[Source] = None, metadata: Optional[dict] = None)
```

**Arguments**:

- `source` - The source of the context.
- `metadata` - The metadata of the context. If None, an empty dict is created

#### embedme

```python
@abstractmethod
def embedme() -> str | PIL.Image.Image
```

Abstract method defined by all subclasses of `Context` to define how embedding modeles should embed the context.

**Returns**:

  The string or image that should be used to embed the context.

#### readme

```python
def readme() -> str | pydantic.BaseModel
```

How LLMs should read the context. By default returns self.serialize()

**Returns**:

  LLM readable format of the context.

#### serialize

```python
def serialize() -> ContextSchema
```

Serializes the context object into its associated `ContextSchema` object. Defined at self.schema.

**Returns**:

  The serialized context object.

#### deserialize

```python
@classmethod
def deserialize(cls, serialized: ContextSchema | dict, **kwargs)
```

Deserializes a `ContextSchema` object into a `Context` object.

**Arguments**:

- `serialized` - The serialized context object or a dict.
- `**kwargs` - Additional keyword arguments to pass to the Context object&#x27;s constructor. (will overide any attributes set in the ContextSchema object)
  

**Returns**:

  The deserialized context object.

#### set\_source

```python
def set_source(source: Source, copy: bool = False)
```

Sets the source of the context object.

**Arguments**:

- `source` - Source - The source of the context object.
- `copy` - bool - Whether to copy the source object to make it safe to mutate.

#### set\_metadata

```python
def set_metadata(metadata: dict, copy: bool = False)
```

Sets the metadata of the context object.

**Arguments**:

- `metadata` - The metadata of the context object.
- `copy` - Whether to copy the metadata object to make it safe to mutate.

#### add\_metadata

```python
def add_metadata(metadata: dict)
```

Adds metadata to the context object.

**Arguments**:

- `metadata` - The metadata to add to the context object.

#### from\_dict

```python
@classmethod
def from_dict(cls, d: dict, **kwargs)
```

Deserializes a dict into a `Context` object.

**Arguments**:

- `d` - The dict to deserialize.
- `**kwargs` - Additional keyword arguments to pass to the Context object&#x27;s constructor. (will overide any attributes set in the dict)
  

**Returns**:

  The deserialized context object.

## Atomic Objects

```python
class Atomic(Context)
```

Base class for all Atomic Context objects. Atomic objects represent context at its finest granularity and are not chunkable.

**Example**:

  In this example, `CaptionedImage` is an `Atomic` context object that stores the caption and the caption embedding.
    ```python
    from modaic.context import ContextSchema
    from modaic.types import String, Vector, Float16Vector

    class CaptionImageSchema(ContextSchema):
        caption: String[100]
        caption_embedding: Float16Vector[384]
        image_path: String[100]

    class CaptionedImage(Atomic):
        schema = CaptionImageSchema

        def __init__(self, image_path: str, caption: str, caption_embedding: np.ndarray, **kwargs):
            super().__init__(**kwargs)
            self.caption = caption
            self.caption_embedding = caption_embedding
            self.image_path = image_path
            self.image = PIL.Image.open(image_path)

        def embedme(self) -> PIL.Image.Image:
            return self.image
    ```

## Molecular Objects

```python
class Molecular(Context)
```

Base class for all `Molecular` Context objects. `Molecular` context objects represent context that can be chunked into smaller `Molecular` or `Atomic` context objects.

**Example**:

  In this example, `MarkdownDoc` is a `Molecular` context object that stores a markdown document.
    ```python
    from modaic.context import Molecular
    from modaic.types import String, Vector, Float16Vector
    from langchain_text_splitters import MarkdownTextSplitter
    from modaic.context import Text

    class MarkdownDocSchema(ContextSchema):
        markdown: String

    class MarkdownDoc(Molecular):
        schema = MarkdownDocSchema

        def chunk(self):
            # Split the markdown into chunks of 1000 characters
            splitter = MarkdownTextSplitter()
            chunk_fn = lambda mdoc: [Text(text=t) for t in splitter.split_text(mdoc.markdown)]
            self.chunk_with(chunk_fn)

        def __init__(self, markdown: str, **kwargs):
            super().__init__(**kwargs)
            self.markdown = markdown
    ```

#### chunk\_with

```python
def chunk_with(chunk_fn: str | Callable[[Context], List[Context]],
               set_source: bool = True,
               **kwargs)
```

Chunk the context object into smaller Context objects.

**Arguments**:

- `chunk_fn` - The function to use to chunk the context object. The function should take in a specific type of Context object and return a list of Context objects.
- `set_source` - bool - Whether to automatically set the source of the chunks using the Context object. (sets chunk.source to self.source, sets chunk.source.parent to self, and updates the chunk.source.metadata with the chunk_id)
- `**kwargs` - dict - Additional keyword arguments to pass to the chunking function.

#### apply\_to\_chunks

```python
def apply_to_chunks(apply_fn: Callable[[Context], None], **kwargs)
```

Applies apply_fn to each chunk in chunks.

**Arguments**:

- `apply_fn` - The function to apply to each chunk. Function should take in a Context object and mutate it.
- `**kwargs` - Additional keyword arguments to pass to apply_fn.

