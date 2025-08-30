from typing import List, Tuple, Optional, Any, Dict
from abc import ABC, abstractmethod
from .context.base import ContextSchema
from pinecone import Pinecone
import os
import dspy


class Reranker(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(
        self,
        query: str,
        options: List[ContextSchema | Tuple[str, ContextSchema]],
        k: int = 10,
        **kwargs,
    ) -> List[Tuple[float, ContextSchema]]:
        """
        Reranks the options based on the query.

        Args:
            query: The query to rerank the options for.
            options: The options to rerank. Each option is a ContextSchema or tuple of (embedme_string, ContextSchema).
            k: The number of options to return.
            **kwargs: Additional keyword arguments to pass to the reranker.

        Returns:
            A list of tuples, where each tuple is (ContextSchema, score).
        """
        embedmes = []
        payloads = []
        for option in options:
            if isinstance(option, ContextSchema):
                embedmes.append(option.embedme())
                payloads.append(option)
            elif isinstance(option, Tuple):
                assert isinstance(option[0], str) and isinstance(
                    option[1], ContextSchema
                ), "options provided to rerank must be ContextSchema objects"
                embedmes.append(option[0])
                payloads.append(option[1])
            else:
                raise ValueError(
                    f"Invalid option type: {type(option)}. Must be ContextSchema or Tuple[str, ContextSchema]"
                )

        results = self._rerank(query, embedmes, k, **kwargs)

        return [(score, payloads[idx]) for idx, score in results]

    @abstractmethod
    def _rerank(
        self, query: str, options: List[str], k: int = 10, **kwargs
    ) -> List[Tuple[int, float]]:
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

    def _rerank(
        self,
        query: str,
        options: List[str],
        k: int = 10,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[int, float]]:
        results = self.pinecone.inference.rerank(
            model=self.model,
            query=query,
            documents=options,
            top_n=k,
            return_documents=False,
            parameters=parameters,
        )
        return [(result.index, result.score) for result in results.data]


class Embedder(dspy.Embedder):
    """
    A wrapper around dspy.Embedder that automatically determines the output size of the model.
    """

    def __init__(self, *args, embedding_dim: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim

        if self.embedding_dim is None:
            output = self("hello")
            self.embedding_dim = output.shape[0]
