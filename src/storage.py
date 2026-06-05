"""Vector storage module for theGist pipeline.

This module handles storing extracted insights into a local ChromaDB
vector database using sentence transformer embeddings. It also provides
semantic search functionality allowing insights to be retrieved by
meaning rather than exact keyword matching.

Typical usage:
    >>> from src.storage import store_insights, query_insights
    >>> store_insights(insights, "Example_Video_Title")
    >>> results = query_insights("how to counter cavalry units")
"""

import json
import logging
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from typing import Optional

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
    CHROMA_N_RESULTS,
    EMBEDDING_MODEL,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
)

# ---------------------------------------------------------------------------
# Logger Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module Level Singletons
# ---------------------------------------------------------------------------

# Embedding model and ChromaDB client are initialised once at module level
# to avoid reloading the model or reconnecting on every function call.
_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def _get_embedding_model() -> SentenceTransformer:
    """Returns the singleton sentence transformer embedding model.

    Loads the model from disk on first call and returns the cached
    instance on all subsequent calls to avoid repeated loading overhead.

    Returns:
        The initialised SentenceTransformer embedding model instance.

    Example:
        >>> model = _get_embedding_model()
        >>> embedding = model.encode("cavalry counters archers")
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully.")
    return _embedding_model


def _get_collection() -> chromadb.Collection:
    """Returns the singleton ChromaDB collection instance.

    Initialises the persistent ChromaDB client and retrieves or creates
    the configured collection on first call. Returns the cached instance
    on all subsequent calls.

    Returns:
        The ChromaDB collection instance for storing and querying insights.

    Example:
        >>> collection = _get_collection()
        >>> print(collection.name)
        'thegist_insights'
    """
    global _chroma_client, _collection
    if _collection is None:
        logger.info(f"Connecting to ChromaDB at: {CHROMA_DB_DIR}")
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR),
        )
        _collection = _chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Connected to collection: {CHROMA_COLLECTION_NAME} "
            f"({_collection.count()} existing insights)"
        )
    return _collection


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generates vector embeddings for a list of text strings.

    Uses the singleton sentence transformer model to encode each text
    string into a fixed length vector representation suitable for
    semantic similarity search.

    Args:
        texts: A list of strings to generate embeddings for.

    Returns:
        A list of embedding vectors where each vector is a list of
        floats corresponding to the encoded text at the same index.

    Example:
        >>> embeddings = _generate_embeddings(["cavalry counters archers"])
        >>> print(len(embeddings[0]))
        384
    """
    model = _get_embedding_model()
    logger.info(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def _build_document_ids(insights: list[str], source_name: str) -> list[str]:
    """Generates unique document IDs for a list of insights.

    Creates a deterministic ID for each insight based on the source
    name and its index position, ensuring IDs are consistent and
    traceable back to their origin video.

    Args:
        insights: The list of insight strings to generate IDs for.
        source_name: The stem name of the source transcript file.

    Returns:
        A list of unique string IDs with one ID per insight in the
        same order as the input list.

    Example:
        >>> ids = _build_document_ids(["cavalry tip"], "Example_Video")
        >>> print(ids[0])
        'Example_Video_0'
    """
    return [f"{source_name}_{i}" for i in range(len(insights))]


def _build_metadata(insights: list[str], source_name: str) -> list[dict]:
    """Builds metadata dictionaries for each insight to store in ChromaDB.

    Metadata is stored alongside each insight vector and can be used
    for filtering queries by source or retrieving provenance information.

    Args:
        insights: The list of insight strings to build metadata for.
        source_name: The stem name of the source transcript file.

    Returns:
        A list of metadata dictionaries with one dictionary per insight
        containing source and character length information.

    Example:
        >>> meta = _build_metadata(["cavalry tip"], "Example_Video")
        >>> print(meta[0])
        {'source': 'Example_Video', 'char_length': 11}
    """
    return [
        {
            "source": source_name,
            "char_length": len(insight),
        }
        for insight in insights
    ]


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

def store_insights(insights: list[str], source_name: str) -> int:
    """Stores a list of extracted insights in the ChromaDB vector database.

    Generates embeddings for each insight using the local sentence
    transformer model and upserts them into the ChromaDB collection
    with associated metadata. Uses upsert to safely handle re-runs
    without creating duplicate entries.

    Args:
        insights: The list of insight strings to store, typically the
            output of extraction.extract_insights().
        source_name: The stem name of the source transcript file used
            for metadata and document ID generation.

    Returns:
        The total number of insights successfully stored in the
        collection after the upsert operation.

    Raises:
        ValueError: If the insights list is empty.

    Example:
        >>> count = store_insights(insights, "Example_Video_Title")
        >>> print(f"Stored {count} insights")
        Stored 40 insights
    """
    if not insights:
        raise ValueError("Cannot store an empty insights list.")

    logger.info(f"Storing {len(insights)} insights for: {source_name}")

    collection = _get_collection()
    embeddings = _generate_embeddings(insights)
    ids = _build_document_ids(insights, source_name)
    metadata = _build_metadata(insights, source_name)

    collection.upsert(
        ids=ids,
        documents=insights,
        embeddings=embeddings,
        metadatas=metadata,
    )

    total = collection.count()
    logger.info(
        f"Upsert complete. {len(insights)} insights stored for {source_name}. "
        f"Total insights in collection: {total}"
    )
    return total


def query_insights(
    query: str,
    source_name: Optional[str] = None,
    n_results: int = CHROMA_N_RESULTS,
) -> list[dict]:
    """Retrieves the most semantically similar insights for a given query.

    Encodes the query string into a vector embedding and searches the
    ChromaDB collection for the closest matching insights using cosine
    similarity. Optionally filters results to a specific source video.

    Args:
        query: The natural language query string to search for.
        source_name: Optional stem name of a source transcript to filter
            results to insights from a specific video only. If None,
            searches across all stored insights.
        n_results: The number of results to return. Defaults to the
            value configured in config.py.

    Returns:
        A list of result dictionaries sorted by relevance, where each
        dictionary contains the following keys:
            - insight: The matched insight string.
            - source: The source video the insight came from.
            - distance: The cosine distance score (lower is more similar).

    Raises:
        ValueError: If the query string is empty or whitespace only.

    Example:
        >>> results = query_insights("how to counter cavalry units")
        >>> for r in results:
        ...     print(r["insight"])
        'Spearmen and halberdiers are effective against cavalry units.'
    """
    if not query.strip():
        raise ValueError("Query string cannot be empty.")

    logger.info(f"Querying insights for: '{query}'")

    collection = _get_collection()
    model = _get_embedding_model()
    query_embedding = model.encode(query).tolist()

    where_filter = {"source": source_name} if source_name else None

    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        results.append({
            "insight": doc,
            "source": meta.get("source", "unknown"),
            "distance": round(dist, 4),
        })

    logger.info(f"Query returned {len(results)} results.")
    return results


def get_all_insights(source_name: Optional[str] = None) -> list[str]:
    """Retrieves all stored insights from the ChromaDB collection.

    Fetches every insight currently stored in the collection,
    optionally filtered to a specific source video. Primarily used
    by the learning layer to build quiz question pools.

    Args:
        source_name: Optional stem name of a source transcript to
            filter results to a specific video only. If None, returns
            all insights across all sources.

    Returns:
        A list of all insight strings matching the filter criteria.

    Example:
        >>> all_insights = get_all_insights("Example_Video_Title")
        >>> print(f"Total: {len(all_insights)}")
        Total: 40
    """
    collection = _get_collection()
    where_filter = {"source": source_name} if source_name else None

    results = collection.get(
        where=where_filter,
        include=["documents"],
    )

    insights = results.get("documents", [])
    logger.info(
        f"Retrieved {len(insights)} insights"
        f"{f' for {source_name}' if source_name else ' across all sources'}."
    )
    return insights