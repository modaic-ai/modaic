from typing import List, Tuple, Optional, Any, Dict
from abc import ABC, abstractmethod
from .context.base import ContextSchema, Context
from pinecone import Pinecone
import os
import dspy


class Indexer:
    def __init__(self, *args, **kwargs):
        pass

    # @abstractmethod
    def ingest(self, contexts: List[Context], *args, **kwargs):
        pass


class Reranker(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(
        self,
        query: str,
        options: List[Context | Tuple[str, Context | ContextSchema]],
        k: int = 10,
        **kwargs,
    ) -> List[Tuple[Context | ContextSchema, float]]:
        """
        Reranks the options based on the query.

        Args:
            query: The query to rerank the options for.
            options: The options to rerank. Each option is a Context or tuple of (embedme_string, Context/ContextSchema).
            k: The number of options to return.
            **kwargs: Additional keyword arguments to pass to the reranker.

        Returns:
            A list of tuples, where each tuple is (Context | ContextSchema, score). The Context or ContextSchema type depends on whichever was passed as an option for that index.
        """
        embedmes = []
        payloads = []
        for option in options:
            if isinstance(option, Context):
                embedmes.append(option.embedme())
                payloads.append(option)
            elif isinstance(option, Tuple):
                assert isinstance(option[0], str) and isinstance(
                    option[1], Context | ContextSchema
                ), (
                    "options provided to rerank must be Context objects or serialized context objects"
                )
                embedmes.append(option[0])
                payloads.append(option[1])
            else:
                raise ValueError(
                    f"Invalid option type: {type(option)}. Must be Context or Tuple[str, Context | ContextSchema]"
                )

        results = self._rerank(query, embedmes, k, **kwargs)

        return [(payloads[idx], score) for idx, score in results]

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
