from typing import List, Tuple, Optional, Any, Dict
from abc import ABC, abstractmethod
from ..context.types import SerializedContext, Context
from pinecone import Pinecone
import os

class Reranker(ABC):
    def __init__(self, *args, **kwargs):
        pass

    
    def __call__(self, 
                 query: str, 
                 options: List[Context | Tuple[str,  Context | SerializedContext]], 
                 k: int = 10, 
                 **kwargs
                 ) -> List[Tuple[Context | SerializedContext, float]]:
        """
        Reranks the options based on the query.
        
        Args:
            query: The query to rerank the options for.
            options: The options to rerank. Each option is a Context or tuple of (embedme_string, Context/SerializedContext).
            k: The number of options to return.
            **kwargs: Additional keyword arguments to pass to the reranker.
            
        Returns:
            A list of tuples, where each tuple is (Context | SerializedContext, score). The Context or SerializedContext type depends on whichever was passed as an option for that index.
        """
        embedmes = []
        payloads = []
        for option in options:
            if isinstance(option, Context):
                embedmes.append(option.embedme())
                payloads.append(option)
            else:
                embedmes.append(option[0])
                payloads.append(option[1])
        
        results = self._rerank(query, embedmes, k, **kwargs)
        
        return [(payloads[idx], score) for idx, score in results]
    
    @abstractmethod
    def _rerank(self, query: str, options: List[str], k: int = 10, **kwargs) -> List[Tuple[int, float]]:
        """
        Reranks the options based on the query.
        
        Args:
            query: The query to rerank the options for.
            options: The options to rerank. Each option is a string.
            k: The number of options to return.
            **kwargs: Additional keyword arguments to pass to the reranker.
            
        Returns:
            A list of tuples, where each tuple is (index, score).
        """
        pass
    

class PineconeReranker(Reranker):
    def __init__(self, model: str, api_key: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        if api_key is None:
            self.pinecone = Pinecone(os.getenv("PINECONE_API_KEY"))
        else:
            self.pinecone = Pinecone(api_key)
    def _rerank(self, query: str, options: List[str], k: int = 10, parameters: Optional[Dict[str, Any]] = None) -> List[Tuple[int, float]]:
        results = self.pinecone.inference.rerank(
            model=self.model,
            query=query,
            documents=options,
            top_n=k,
            return_documents=False,
            parameters=parameters,
        )
        return [(result.index, result.score) for result in results.data]