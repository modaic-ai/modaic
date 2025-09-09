from typing import TYPE_CHECKING, Dict, Iterable, List, Literal, Optional, Tuple

import immutables
import numpy as np
from PIL import Image
from tqdm import tqdm

from modaic.context.base import Context, Embeddable

from .common import _add_item_embedme, _items_have_multiple_embedmes

if TYPE_CHECKING:
    from modaic.databases.vector_database import VectorDatabase


def has_collection(vdb: "VectorDatabase", collection_name: str) -> bool:
    """
    Check if a collection exists in the vector database.

    Args:
        collection_name: The name of the collection to check

    Returns:
        True if the collection exists, False otherwise
    """
    return vdb.ext.backend.has_collection(collection_name)


def _embed_and_add_records(
    vdb: "VectorDatabase",
    collection_name: str,
    embedmes: Dict[str, List[str | Image.Image]] | Dict[None, List[str | Image.Image]],
    contexts: List[Context],
):
    # TODO: could add functionality for multiple embedmes per context (e.g. you want to embed both an image and a text description of an image)
    all_embeddings = {}
    if collection_name not in vdb.collections:
        raise ValueError(
            f"Collection {collection_name} not found in VectorDatabase's indexes, Please use VectorDatabase.create_collection() to create a collection first. Alternatively, you can use VectorDatabase.load_collection() to add records to an existing collection."
        )
    try:
        first_index = next(iter(vdb.collections[collection_name].indexes.keys()))
        # NOTE: get embeddings for each index
        for index_name, index_config in vdb.collections[collection_name].indexes.items():
            embeddings = index_config.embedder(embedmes)
            # NOTE: Ensure embeddings is a 2D array (DSPy returns 1D for single strings, 2D for lists)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            # NOTE: If index_name is None use the only index for the collection
            all_embeddings[index_name or first_index] = embeddings
    except Exception as e:
        raise ValueError(f"Failed to create embeddings for index: {index_name}") from e

    data_to_insert: List[immutables.Map[str, np.ndarray]] = []
    # FIXME Probably should add type checking to ensure context matches schema, not sure how to do this efficiently
    for i, item in enumerate(contexts):
        embedding_map: dict[str, np.ndarray] = {}
        for index_name, embedding in all_embeddings.items():
            embedding_map[index_name] = embedding[i]

        # Create a record with embedding and validated metadata
        record = vdb.ext.backend.create_record(embedding_map, item)
        data_to_insert.append(record)

    vdb.ext.backend.add_records(collection_name, data_to_insert)
    del data_to_insert


def add_records(
    vdb: "VectorDatabase",
    collection_name: str,
    records: Iterable[Embeddable | Tuple[str | Image.Image, Context]],
    batch_size: Optional[int] = None,
    embedme_scope: Literal["auto", "context", "index"] = "auto",
):
    """
    Add items to a collection in the vector database.
    Uses the Context's get_embed_context() method and the embedder to create embeddings.

    Args:
        collection_name: The name of the collection to add records to
        records: The records to add to the collection
        batch_size: Optional batch size for processing records
    """
    if not records:
        return

    # NOTE: Make embedmes compatible with the ext's hybrid search function
    if embedme_scope == "auto":
        if _items_have_multiple_embedmes(records):
            embedme_scope = "index"
        else:
            embedme_scope = "context"

    if embedme_scope == "index":
        embedmes: Dict[str, List[str | Image.Image]] = {k: [] for k in vdb.collections[collection_name].indexes.keys()}
    else:
        # CAVEAT: We make embedmes a dict with None as opposed to a list so we don't have to type check it
        embedmes: Dict[None, List[str | Image.Image]] = {None: []}

    serialized_contexts = []
    # TODO: add multi-processing/multi-threading here, just ensure that the backend is thread-safe. Maybe we add a class level parameter to check if the vendor is thread-safe. Embedding will still need to happen on a single thread
    with tqdm(total=len(records), desc="Adding records to vector database", position=0) as pbar:
        for _, item in tqdm(
            enumerate(records),
            desc="Adding records to vector database",
            position=0,
            leave=False,
        ):
            _add_item_embedme(embedmes, item)
            serialized_contexts.append(item)

            if batch_size is not None and len(serialized_contexts) == batch_size:
                _embed_and_add_records(vdb, collection_name, embedmes, serialized_contexts)
                if embedme_scope == "index":
                    embedmes = {k: [] for k in embedmes.keys()}
                else:
                    embedmes = {None: []}
                serialized_contexts = []
                pbar.update(batch_size)

    if serialized_contexts:
        _embed_and_add_records(vdb, collection_name, embedmes, serialized_contexts)
