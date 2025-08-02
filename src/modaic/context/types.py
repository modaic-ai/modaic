from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, Callable, List
from enum import auto
from abc import ABC, abstractmethod
import copy

class SourceType(Enum):
    FILE_OBJECT = auto()
    LOCAL_PATH = auto()
    URL = auto()
    CONTEXT_OBJECT = auto()

    
@dataclass
class Source:
    origin: Any
    type: SourceType
    metadata: dict = field(default_factory=dict)

class SerializedContext:
    def __init__(self, metadata: dict = {}, source: Source = None, **kwargs):
        self.metadata: dict = metadata
        self.source: Source = source
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __str__(self):
        vals = '\n\t'.join([f'{k}={v}' for k,v in self.__dict__.items() if not k.startswith('_')])
        return f"{self.__class__.__name__}(\n\t{vals}\n)"
    
    @classmethod
    def from_dict(cls, d: dict):
        new_d = {}
        match d:
            case {"source": source_dict, "metadata": metadata, **rest_of_dict}:
                match source_dict:
                    case {"type": source_type, "origin": origin_dict, "metadata": metadata_dict}:
                        type = SourceType[source_type]
                        new_source = Source(type=type, origin=origin_dict, metadata=metadata_dict)
                        new_d = {**rest_of_dict, "source": new_source, "metadata": metadata}
                        return cls(**new_d)
                    
        raise ValueError(f"Invalid SerializedContext dictionary format: {d}")
    
    def to_dict(self):
        match self.__dict__:
            case {"source": source, "metadata": metadata, **rest_of_dict}:
                match source:
                    case Source(type=source_type, origin=origin, metadata=source_metadata):
                        source_dict = {"type": source_type.name, "origin": origin, "metadata": source_metadata}
                        return {"source": source_dict, "metadata": metadata, **rest_of_dict}

        raise ValueError(f"Invalid SerializedContext: {self}")
    
    def copy(self):
        return copy.deepcopy(self)
    
    
        

class Context(ABC):
    def __init__(self, source: Optional[Source] = None, metadata: dict = {}):
        self.source = source
        self.metadata = metadata
    
    @abstractmethod
    def embedme(self) -> str:
        pass
    
    @abstractmethod
    def readme(self):
        pass
    
    def serialize(self) -> SerializedContext:
        d = {}
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                d[k] = v
        return SerializedContext(**d)
    
    @classmethod
    def deserialize(cls, serialized: SerializedContext, **kwargs):
        try:
            match serialized.to_dict():
                case {"source": source, **rest_of_dict}:
                    match source:
                        case {"type": source_type, "origin": origin, "metadata": source_metadata}:
                            type = SourceType[source_type]
                            source = Source(type=type, origin=origin, metadata=source_metadata)
                            return cls(source=source, **rest_of_dict)
        except: # noqa
            return None
        
    def set_source(self, source: Source):
        self.source = source
    
    def set_metadata(self, metadata: dict):
        self.metadata = metadata
    
    def add_metadata(self, metadata: dict):
        self.metadata.update(metadata)
    
    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return cls.deserialize(SerializedContext.from_dict(d), **kwargs)
    
    def resolve_source(self) -> str:
        """
        Get the top level origin of the context.
        
        Returns:
            The top level origin of the context, or None if no source is set.
        """
        if self.source is None:
            return None
        elif self.source.type == SourceType.CONTEXT_OBJECT:
            return self.source.origin.resolve_source()
        else:
            return self.source.origin
    
    def __str__(self):
        return f"{self.__class__.__name__}(source={self.source}, metadata={self.metadata})"
    
    def __repr__(self):
        return self.__str__()
    
        
    
class Atomic(Context):
    """
    Atomic Context is a single piece of information that can be embedded and used for retrieval.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    
    
#TODO add support for PIL.Image.Image and Video embed types we'll need to replace dspy.Embedder with a more general embedder
class Molecular(Context):
    """
    Molecular context objects can be chunked into smaller Context objects.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chunks: List[Context] = []
        
    def chunk(self, chunk_fn: str|Callable, **kwargs) -> bool:
        """
        Chunk the context object into smaller Context objects.
        Args:
            chunk_fn: str | Callable - The name of the chunking function to use. If a string, the function must be defined on the class. If a callable, the callable will be used to chunk the context object.
            **kwargs: dict - Additional keyword arguments to pass to the chunking function.
        Returns:
            bool - True if the chunking function was found, False otherwise.
        """
        if isinstance(chunk_fn, str):
            chunk_fn_attr = f"chunk_{chunk_fn}"
            if hasattr(self, chunk_fn_attr):
                chunk_callable = getattr(self, chunk_fn_attr)
                chunk_callable(**kwargs)
                return True
            else:
                raise ValueError(f"Chunk function {chunk_fn_attr} not found for {self.__class__.__name__}")
        return False
    
    def apply_to_chunks(self, apply_fn: Callable[[Context], None], **kwargs):
        """
        Applies apply_fn to each chunk in chunks.
        
        Args:
            apply_fn: The function to apply to each chunk. Function should take in a Context object and mutate it.
            **kwargs: Additional keyword arguments to pass to apply_fn.
        """
        for chunk in self.chunks:
            apply_fn(chunk, **kwargs)
