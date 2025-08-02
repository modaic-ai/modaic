"""
Text Context Implementation

This module provides the TextContext class for handling text-based content
with automatic serialization and deserialization support.
"""

from context import Context, SerializedContext


class TextContext(Context):
    """
    Context class for text-based content with auto-registration support.
    """
    context_type = "text"  # Auto-registration identifier
    
    def __init__(self, text: str, **kwargs):
        """
        Initialize a TextContext with text content.
        
        Args:
            text: The text content to store
            **kwargs: Additional metadata for the context
        """
        super().__init__(**kwargs)
        self.text = text
        # Store text in metadata for serialization
        self.metadata['text'] = text
    
    def serialize(self) -> SerializedContext:
        """
        Serialize this TextContext to a SerializedContext.
        
        Returns:
            SerializedContext instance containing this context's data
        """
        return SerializedContext(
            type=TextContext,
            metadata=self.metadata
        )
    
    @staticmethod
    def deserialize(serialized: SerializedContext):
        """
        Deserialize a SerializedContext to TextContext.
        
        Args:
            serialized: SerializedContext instance
            
        Returns:
            TextContext instance
        """
        text = serialized.metadata.get("text", "")
        # Remove 'text' from metadata copy to avoid duplication
        other_metadata = {k: v for k, v in serialized.metadata.items() if k != 'text'}
        return TextContext(text=text, **other_metadata)
    
    def get_text(self) -> str:
        """
        Get the text content.
        
        Returns:
            The text content of this context
        """
        return self.text
    
    def set_text(self, text: str):
        """
        Update the text content.
        
        Args:
            text: New text content
        """
        self.text = text
        self.metadata['text'] = text
    
    def __str__(self):
        return f"TextContext(text='{self.text[:50]}{'...' if len(self.text) > 50 else ''}')"
    
    def __repr__(self):
        return f"TextContext(text={repr(self.text)}, metadata={self.metadata})"


# Auto-register the TextContext class
TextContext.register_for_auto_context()
