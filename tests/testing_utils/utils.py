import math
import os
import random
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Type

import numpy as np
import requests

from modaic.context import Context
from modaic.databases import SearchResult
from modaic.databases.vector_database import Metric
from modaic.indexing import DummyEmbedder, Embedder


class Membedder(DummyEmbedder):
    """
    An embedder that generates random embeddings. And remembers the embeddings it has generated.
    """

    def __init__(self, embedding_dim: int = 512):
        super().__init__(embedding_dim)
        self.embeddings = {}

    def __call__(self, text: str | List[str]) -> np.ndarray:
        embedding = super().__call__(text)
        self.embeddings[text] = embedding
        return embedding


DummyIndex = Dict[str, tuple]  # context_id -> embedding

DEFAULT_INDEX_NAME = "vector"


@dataclass
class DummyCollection:
    records: Dict[str, Context]  # context_id -> context
    indexes: Dict[str, DummyIndex]  # index_name -> index
    payload_class: Type[Context]
    metrics: Dict[str, Metric]  # index_name -> metric


def dot_product(a: tuple, b: tuple) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def euclidean_distance(a: tuple, b: tuple) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=False)))


def manhattan_distance(a: tuple, b: tuple) -> float:
    return sum(abs(x - y) for x, y in zip(a, b, strict=False))


def hamming_distance(a: tuple, b: tuple) -> int:
    return sum(x != y for x, y in zip(a, b, strict=False))


def jaccard_similarity(a: tuple, b: tuple) -> float:
    set_a, set_b = set(a), set(b)
    return len(set_a & set_b) / len(set_a | set_b)


def minhash_signature(s: tuple, num_hashes: int = 100) -> list[int]:
    max_val = 2**32 - 1
    hashes = []
    for _ in range(num_hashes):
        a, b = random.randint(1, max_val), random.randint(0, max_val)
        h = min(((a * hash(x) + b) % max_val) for x in s)
        hashes.append(h)
    return hashes


def mhjaccard_similarity(a: tuple, b: tuple, num_hashes: int = 100) -> float:
    sig_a = minhash_signature(set(a), num_hashes)
    sig_b = minhash_signature(set(b), num_hashes)
    return sum(x == y for x, y in zip(sig_a, sig_b, strict=False)) / num_hashes


metric_to_fn = {
    Metric.DOT_PRODUCT: dot_product,
    Metric.EUCLIDEAN: euclidean_distance,
    Metric.MANHATTAN: manhattan_distance,
    Metric.HAMMING: hamming_distance,
    Metric.JACCARD: jaccard_similarity,
    Metric.MHJACCARD: mhjaccard_similarity,
}


class DummyBackend:
    _name: ClassVar[str] = "dummy"
    _client: None
    collections: Dict[str, DummyCollection]

    def __init__(self):
        self._client = None

    def create_record(self, embedding_map: Dict[str, np.ndarray], context: Context) -> Any:
        record = {
            "context": Context,
            **{index: tuple(embedding.tolist()) for index, embedding in embedding_map.items()},
        }
        return record

    def add_records(self, collection_name: str, records: List[Any]) -> None:
        self.collections[collection_name].records.update(
            {str(record["context"].id): record["context"] for record in records}
        )
        for index in self.collections[collection_name].indexes:
            self.collections[collection_name].indexes[index].update(
                {record["context"].id: record[index] for record in records}
            )

    def drop_collection(self, collection_name: str) -> None:
        del self.collections[collection_name]

    def create_collection(self, collection_name: str, payload_class: Type[Context], metric: Metric) -> None:
        empty_index = {}
        self.collections[collection_name] = DummyCollection(
            records={},
            indexes={DEFAULT_INDEX_NAME: empty_index},
            payload_class=payload_class,
            metrics={DEFAULT_INDEX_NAME: metric},
        )

    def list_collections(self) -> List[str]:
        return list(self.collections.keys())

    def has_collection(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def search(
        self,
        collection_name: str,
        vector: np.ndarray,
        payload_class: Type[Context],
        k: int,
        filter: Optional[dict] = None,
    ) -> List[SearchResult]:
        if filter is not None:
            raise NotImplementedError("Filtering is not implemented for DummyBackend")
        distances = []
        for id_, embedding in self.collections[collection_name].indexes[DEFAULT_INDEX_NAME].items():
            distance = metric_to_fn[self.collections[collection_name].metrics[DEFAULT_INDEX_NAME]](embedding, vector)
            distances.append((id_, distance))
        distances.sort(key=lambda x: x[1])
        return [
            SearchResult(id=id_, score=distance, context=self.collections[collection_name].records[id_])
            for id_, distance in distances[:k]
        ]


class HardcodedEmbedder(Embedder):
    text_to_embedding: Dict[str, np.ndarray]

    def __init__(self, embedding_dim: int = 3):
        # Don't call super().__init__() to avoid model requirement
        self.embedding_dim = embedding_dim

        self.text_to_embedding = {}

    def __call__(
        self,
        text: str | List[str],
        embedding: Optional[np.ndarray | List[np.ndarray]] = None,
    ) -> np.ndarray:
        if isinstance(text, str):
            if embedding is not None:
                self.text_to_embedding[text] = embedding
                return embedding
            else:
                return self.text_to_embedding[text]
        elif isinstance(text, list):
            if embedding is not None:
                for t in text:
                    self.text_to_embedding[t] = embedding
                return embedding
            else:
                return np.vstack([self.text_to_embedding[t] for t in text])
        raise ValueError(f"Invalid text type: {type(text)}")


def delete_program_repo(
    username: str,
    program_name: str,
    base_url: str = None,
    bearer_token: Optional[str] = None,
    stytch_session: Optional[str] = None,
) -> requests.Response:
    """
    Delete a program repository.

    Params:
        base_url (str): API base URL (e.g., http://localhost:8000).
        username (str): Owner username.
        program_name (str): Repository name.
        bearer_token (Optional[str]): Bearer token for Authorization header.
        stytch_session (Optional[str]): Session token for 'stytch_session' cookie.

    Returns:
        requests.Response: HTTP response object.
    """
    if base_url is None:
        base_url = os.getenv("MODAIC_API_URL")
    if bearer_token is None:
        bearer_token = os.getenv("MODAIC_TOKEN")
    url = f"{base_url}/api/v1/agents/delete/owner/{username}/agent/{program_name}" #TODO: endpoints need to be update to fit program nomenclature
    headers = {"Authorization": f"token {bearer_token}"}
    cookies = {"stytch_session": stytch_session} if stytch_session else {}
    resp = requests.delete(url, headers=headers, cookies=cookies, timeout=30)
    return resp
