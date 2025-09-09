import math
import random
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Type, TypedDict

import numpy as np

from modaic.context import Context
from modaic.databases import SearchResult
from modaic.databases.vector_database import Metric, MilvusBackend, VectorDatabase
from modaic.databases.vector_database.vector_database import VectorDBBackend
from modaic.indexing import DummyEmbedder


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


def minhash_signature(s, num_hashes=100):
    max_val = 2**32 - 1
    hashes = []
    for i in range(num_hashes):
        a, b = random.randint(1, max_val), random.randint(0, max_val)
        h = min(((a * hash(x) + b) % max_val) for x in s)
        hashes.append(h)
    return hashes


def mhjaccard_similarity(a: tuple, b: tuple, num_hashes=100) -> float:
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
            SearchResult(id=id_, distance=distance, context=self.collections[collection_name].records[id_])
            for id_, distance in distances[:k]
        ]


class HardcodedEmbedder(Embedder):
    text_to_embedding: Dict[str, np.ndarray]

    def __init__(self):
        self.id_to_embedding = {}

    def __call__(self, text: str | List[str], embedding: Optional[np.ndarray | List[np.ndarray]] = None) -> np.ndarray:
        if embedding is not None:
            self.text_to_embedding[text] = embedding
        if isinstance(text, str):
            return self.id_to_embedding[text]
        else:
            return [self.id_to_embedding[t] for t in text]
        return self.id_to_embedding[text]
