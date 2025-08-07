# Context Engineering with Modaic

Modaic comes with a powerful context engineering toolkit that is designed to be extremely flexible while still being portable and easy to use. In our SDK you have a range of context types to choose from and you can also easily define your own context class with custom behavior. From simple context classes like text and images to more complex context classes like tables, code bases, and user profiles.

## Molecular and Atomic Context

Wait... why are we talking about chemistry? Good question. In Modaic we split context into two parts. _Molecular_ and _Atomic_. `Atomic` context, as the name suggests, is atomic. It cannot be broken down into further parts. Some examples include text, a single image, or a single website element. `Molecular` context is a context which can be chunked into smaller parts of either `Atomic` **or** `Molecular` context. Like a markdown doc, a pdf, or a website. When you create a custom context class you will extend from one of these two base classes.

## SerializedContext

The `SerializedContext` class helps define how our context objects will be serialized and deserialized when they are stored in vector and graph databases. Every context class must define a class attribute called `serialized_context_class` which points to its to a child class of `SerializedContext` and defines what fields from your context class will be serialized and how. `SerializedContext` under the hood is a `pydantic.BaseModel` so you can use all the features of pydantic to define your serialized context class.

## Source Tracking

All context classes automatically track their source. Throughout your entire framework. The `Source` class which is a `pydantic.BaseModel` that contains the following fields:

- `origin: Optional[str]`: The filename, url, or hostname the context originates from
- `type: Optional[SourceType]`: The type of origin can be one of the following:
  - `SourceType.LOCAL_PATH`: The context is a local file path
  - `SourceType.URL`: The context is a url
  - `SourceType.SQL_DB`: The context is a SQL database
- `metadata: dict`: Source metadata that can be used to identify the context in the origin. For example, a chunk id, row id, or table id.
- `parent: Optional[Context]`: A special property that is not serialized but contains a weakref to the context's parent context. 

!!! warning
    `parent` is a weakref to the parent context object. When the parent context goes out of scope, `source.parent` will be `None`. Also, `parent` is **never** serialized.

## Methods
### Predefined Methods

All `Context` classes come with 2 main methods:
```python
def serialize(self) -> SerializedContext:
    """Serialize the context into a `SerializedContext` object."""
```
```python
@classmethod
def deserialize(cls, serialized: SerializedContext | dict, **kwargs) -> Context:
    """Deserialize the context from a `SerializedContext` object. Can use kwargs to pass in additional fields to the Context constructor. 
    For example, fields that were not serialized but are needed to initialize the context. Or fields that you would like to override from the serialized context."""
```

Additionally, the context classes that extend from `Molecular` have a `chunk_with` method that can be used to chunk the context into smaller parts. It has the following signature:

```python
def chunk_with(self, chunk_fn: str | Callable[[Context], List[Context]], set_source: bool = True, **kwargs) -> bool:
    """
     Chunks a Molecular context into smaller parts of either `Atomic` or `Molecular` context. Chunks will be stored in the `.chunks` attribute.
     Args:
     - chunk_fn: A function that takes a context and returns a list of context objects.
     - set_source: Whether to automatically set the source of the chunked context objects based on the parent context.
     - **kwargs: Additional keyword arguments to pass to the chunk function.
    """
```
!!! tip
    All the above methods are automatically implemented for you when you extend from `Atomic` or `Molecular`. No need to override them. Unless you want to of course :wink:  

### Abstract Methods
The below methods are abstract methods that you must override when you extend from `Atomic` or `Molecular`.

```python
    @abstractmethod
    def embedme(self) -> str | PIL.Image.Image:
        """
        Abstract method defined by all subclasses of `Context` to define embedding behavior for embedding models.
        Returns:
            The string or image that should be used to embed the context.
        """
```
```python
    @abstractmethod
    def readme(self) -> str:
        """
        Abstract method defined by all subclasses of `Context` to define readme behavior for LLMs.
        Returns:
            The readme string that should be read by LLMs.
        """
```

## Storage
