import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import dspy
import numpy as np

from .context.base import Context
from .observability import Trackable, track_modaic_obj


class Reranker(ABC, Trackable):
    def __init__(self, *args, **kwargs):
        ABC.__init__(self)
        Trackable.__init__(self, **kwargs)

    @track_modaic_obj
    def __call__(
        self,
        query: str,
        options: List[Context | Tuple[str, Context]],
        k: int = 10,
        **kwargs,
    ) -> List[Tuple[float, Context]]:
        """
        Reranks the options based on the query.

        Args:
            query: The query to rerank the options for.
            options: The options to rerank. Each option is a Context or tuple of (embedme_string, Context).
            k: The number of options to return.
            **kwargs: Additional keyword arguments to pass to the reranker.

        Returns:
            A list of tuples, where each tuple is (Context, score).
        """
        embedmes = []
        payloads = []
        for option in options:
            if isinstance(option, Context):
                embedmes.append(option.embedme())
                payloads.append(option)
            elif isinstance(option, Tuple):
                assert isinstance(option[0], str) and isinstance(option[1], Context), (
                    "options provided to rerank must be Context objects"
                )
                embedmes.append(option[0])
                payloads.append(option[1])
            else:
                raise ValueError(f"Invalid option type: {type(option)}. Must be Context or Tuple[str, Context]")

        results = self._rerank(query, embedmes, k, **kwargs)

        return [(score, payloads[idx]) for idx, score in results]

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
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError("Pinecone is not installed. Please install it with `uv add pinecone`") from None

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


class DummyEmbedder(Embedder):
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim

    def __call__(self, text: str | List[str]) -> np.ndarray:
        if isinstance(text, str):
            return np.random.rand(self.embedding_dim)
        else:
            return np.random.rand(len(text), self.embedding_dim)
