"""
Auto-Resolution Context System - Base Classes

This module provides the core infrastructure for automatic type resolution
of serialized context objects, similar to HuggingFace's AutoModel system.
"""

from dataclasses import dataclass
from typing import Type, ClassVar, Dict, Any
import importlib
from collections import OrderedDict
from abc import ABC, abstractmethod


# Context type registry - similar to HuggingFace's CONFIG_MAPPING
CONTEXT_MAPPING = OrderedDict()


class AutoContextRegistry:
    """
    Registry for automatically resolving context types similar to HuggingFace's AutoModel.
    """
    
    @classmethod
    def register(cls, context_type: str, context_class: Type['Context']):
        """
        Register a context class with a given type identifier.
        
        Args:
            context_type: String identifier for the context type (e.g., "text", "image", "document")
            context_class: The Context class to associate with this type
        """
        CONTEXT_MAPPING[context_type] = context_class
    
    @classmethod
    def get_context_class(cls, context_type: str) -> Type['Context']:
        """
        Get the context class for a given type identifier.
        
        Args:
            context_type: String identifier for the context type
            
        Returns:
            The Context class associated with the type
        """
        if context_type in CONTEXT_MAPPING:
            return CONTEXT_MAPPING[context_type]
        raise ValueError(f"Unknown context type: {context_type}. Available types: {list(CONTEXT_MAPPING.keys())}")
    
    @classmethod
    def resolve_from_string(cls, type_string: str) -> Type['Context']:
        """
        Resolve a context class from either a registered type or a full module path.
        
        Args:
            type_string: Either a registered type identifier or full module.ClassName path
            
        Returns:
            The resolved Context class
        """
        # First try to resolve from registry
        if type_string in CONTEXT_MAPPING:
            return CONTEXT_MAPPING[type_string]
        
        # Fall back to importlib resolution for full module paths
        try:
            if '.' in type_string:
                module_path, class_name = type_string.rsplit('.', 1)
                module = importlib.import_module(module_path)
                return getattr(module, class_name)
            else:
                # Try to import as a module name
                return importlib.import_module(type_string)
        except (ValueError, ModuleNotFoundError, AttributeError) as e:
            raise ImportError(f"Could not resolve context type '{type_string}'. Not found in registry {list(CONTEXT_MAPPING.keys())} and failed to import: {e}")


@dataclass
class SerializedContext:
    """
    Container for serialized context data with automatic type resolution.
    """
    type: Type['Context']
    metadata: dict
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        """
        Create a SerializedContext from a dictionary with automatic type resolution.
        
        Args:
            d: Dictionary containing 'type' and other fields
            
        Returns:
            SerializedContext instance with resolved type
        """
        match d:
            case {"type": type_name, **rest}:
                # Use AutoContextRegistry for automatic resolution
                resolved_type = AutoContextRegistry.resolve_from_string(type_name)
                return cls(type=resolved_type, **rest)
            case _:
                raise ValueError(f"Invalid serialized context: {d}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert SerializedContext to dictionary format.
        
        Returns:
            Dictionary representation with type as string identifier
        """
        # Try to find a registered type name first
        type_name = None
        for registered_type, registered_class in CONTEXT_MAPPING.items():
            if registered_class == self.type:
                type_name = registered_type
                break
        
        # Fall back to full module path if not found in registry
        if type_name is None:
            type_name = self.type.__module__ + "." + self.type.__qualname__
        print(f"$$$$type_name: {type_name}")
            
        return {"type": type_name, "metadata": self.metadata}


class Context(ABC):
    """
    Base class for all context types with auto-registration support.
    """
    # Class variable to store the context type for auto-registration
    context_type: ClassVar[str] = None
    
    def __init__(self, **kwargs):
        """
        Initialize the base context.
        
        Args:
            **kwargs: Additional metadata for the context
        """
        self.metadata = kwargs
    
    @classmethod
    def register_for_auto_context(cls, context_type: str = None):
        """
        Register this context class for automatic resolution.
        Similar to HuggingFace's register_for_auto_class method.
        
        Args:
            context_type: Optional type identifier. If not provided, uses cls.context_type
        """
        type_id = context_type or cls.context_type
        if type_id is None:
            raise ValueError(f"No context_type specified for {cls.__name__}. Either set cls.context_type or provide context_type parameter.")
        
        AutoContextRegistry.register(type_id, cls)
    
    @abstractmethod
    def serialize(self) -> SerializedContext:
        """
        Serialize this context instance to a SerializedContext.
        Subclasses must implement this method.
        
        Returns:
            SerializedContext instance containing this context's data
        """
        pass
    
    @staticmethod
    @abstractmethod
    def deserialize(serialized: SerializedContext):
        """
        Deserialize a SerializedContext back to a Context instance.
        Subclasses must implement this method.
        
        Args:
            serialized: SerializedContext instance
            
        Returns:
            Context instance of the appropriate type
        """
        pass
    
    @staticmethod
    def from_serialized(serialized: SerializedContext):
        """
        Create a Context instance from a SerializedContext using automatic type resolution.
        
        Args:
            serialized: SerializedContext instance
            
        Returns:
            Context instance of the appropriate type
        """
        cls = serialized.type
        try:
            return cls.deserialize(serialized)
        except NotImplementedError:
            try:
                return cls(**serialized.metadata)
            except Exception as e:
                raise ValueError(f"Cannot deserialize {cls} from {serialized}: {e}")
