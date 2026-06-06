"""Study topics module for theGist pipeline.

This module provides functionality for creating, managing, and refreshing
user defined study topics. A study topic aggregates semantically related
insights from across all ingested videos into a single curated collection
that can be reviewed and quizzed on independently.

Topics are persisted as JSON files in the configured topics directory
defined in config.py and can be refreshed as new video content is added
to the knowledge base.

Typical usage:
    >>> from src.topics import create_topic, get_topic, list_topics
    >>> topic = create_topic("Celts early game strategy", "Celts early game feudal age rush")
    >>> print(f"Created topic with {len(topic['insights'])} insights")
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import (
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
    TOPIC_INSIGHT_COUNT,
    TOPICS_DIR,
)
from src.storage import query_insights

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
# Internal Helpers
# ---------------------------------------------------------------------------

def _sanitize_topic_name(name: str) -> str:
    """Converts a topic name into a safe filename string.

    Removes or replaces characters that are invalid in filenames and
    normalises whitespace to underscores for consistent file naming.

    Args:
        name: The raw topic name string to sanitize.

    Returns:
        A sanitized string safe for use as a filename, with a maximum
        length of 80 characters.

    Example:
        >>> _sanitize_topic_name("Celts: Early Game Strategy!")
        'Celts_Early_Game_Strategy'
    """
    sanitized = re.sub(r"[^\w\s-]", "", name)
    sanitized = re.sub(r"\s+", "_", sanitized).strip("_")
    return sanitized[:80]


def _topic_path(topic_name: str) -> Path:
    """Returns the file path for a given topic name.

    Args:
        topic_name: The raw topic name string to derive the path from.

    Returns:
        A Path object pointing to the topic JSON file location.

    Example:
        >>> path = _topic_path("Celts Early Game")
        >>> print(path.name)
        'Celts_Early_Game.json'
    """
    return TOPICS_DIR / f"{_sanitize_topic_name(topic_name)}.json"


def _load_topic_file(path: Path) -> dict:
    """Loads and parses a topic JSON file from disk.

    Args:
        path: The Path to the topic JSON file to load.

    Returns:
        The parsed topic dictionary.

    Raises:
        FileNotFoundError: If no topic file exists at the given path.
    """
    if not path.exists():
        raise FileNotFoundError(f"Topic file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_topic_file(topic: dict) -> Path:
    """Saves a topic dictionary to disk as a JSON file.

    Args:
        topic: The topic dictionary to persist. Must contain a 'name' key.

    Returns:
        A Path object pointing to the saved topic JSON file.
    """
    path = _topic_path(topic["name"])
    path.write_text(
        json.dumps(topic, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Topic saved: {path.name}")
    return path


def _fetch_insights_for_topic(
    query: str,
    source_name: Optional[str],
    insight_count: int,
) -> list[dict]:
    """Fetches semantically relevant insights for a topic query.

    Queries ChromaDB using the topic query string and returns the top
    matching insights with their source and distance metadata.

    Args:
        query: The natural language query string defining the topic.
        source_name: Optional source filter to restrict results to a
            specific video. If None searches across all sources.
        insight_count: The number of insights to retrieve.

    Returns:
        A list of result dictionaries each containing 'insight',
        'source', and 'distance' keys.
    """
    return query_insights(
        query,
        source_name=source_name,
        n_results=insight_count,
    )


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

def create_topic(
    name: str,
    query: str,
    source_name: Optional[str] = None,
    insight_count: int = TOPIC_INSIGHT_COUNT,
) -> dict:
    """Creates a new study topic and populates it with relevant insights.

    Queries the ChromaDB knowledge base using the provided query string
    to find the most semantically relevant insights across all ingested
    videos, or optionally filtered to a specific source. The resulting
    topic is persisted to disk and returned.

    Args:
        name: The display name for the study topic. Used as the topic
            identifier and derived filename.
        query: The natural language query used to find relevant insights.
            Should describe the topic clearly for best semantic matching.
        source_name: Optional stem name of a source transcript to restrict
            insights to a specific video only. If None, aggregates across
            all ingested sources.
        insight_count: The number of insights to include in the topic.
            Defaults to TOPIC_INSIGHT_COUNT from config.py.

    Returns:
        The newly created topic dictionary containing the following keys:
            - name: The topic display name.
            - query: The query used to populate the topic.
            - source_filter: The source filter applied, or None.
            - created_at: ISO format timestamp of topic creation.
            - refreshed_at: ISO format timestamp of last refresh.
            - insight_count: Number of insights in the topic.
            - insights: List of insight dictionaries with text and metadata.

    Raises:
        ValueError: If a topic with the same name already exists on disk
            or if no insights are found for the given query.

    Example:
        >>> topic = create_topic(
        ...     "Celts early game",
        ...     "Celts feudal age rush strategy"
        ... )
        >>> print(f"Created with {topic['insight_count']} insights")
        Created with 20 insights
    """
    path = _topic_path(name)
    if path.exists():
        raise ValueError(
            f"Topic '{name}' already exists. "
            f"Use refresh_topic() to update it or delete it first."
        )

    logger.info(f"Creating topic: '{name}' | query: '{query}'")
    results = _fetch_insights_for_topic(query, source_name, insight_count)

    if not results:
        raise ValueError(
            f"No insights found for query: '{query}'. "
            f"Ingest more videos related to this topic first."
        )

    now = datetime.now().isoformat()
    topic = {
        "name": name,
        "query": query,
        "source_filter": source_name,
        "created_at": now,
        "refreshed_at": now,
        "insight_count": len(results),
        "insights": results,
    }

    _save_topic_file(topic)
    logger.info(
        f"Topic '{name}' created with {len(results)} insights."
    )
    return topic


def get_topic(name: str) -> dict:
    """Retrieves a study topic by name from disk.

    Args:
        name: The display name of the topic to retrieve.

    Returns:
        The topic dictionary loaded from its JSON file.

    Raises:
        FileNotFoundError: If no topic with the given name exists.

    Example:
        >>> topic = get_topic("Celts early game")
        >>> print(topic["insight_count"])
        20
    """
    path = _topic_path(name)
    try:
        topic = _load_topic_file(path)
        logger.info(
            f"Loaded topic: '{name}' "
            f"({topic['insight_count']} insights)"
        )
        return topic
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Topic '{name}' not found. "
            f"Use create_topic() to create it first."
        )


def refresh_topic(name: str) -> dict:
    """Refreshes a study topic by re-running its query against the knowledge base.

    Re-queries ChromaDB using the topic's original query and source filter
    to incorporate any new insights added since the topic was last created
    or refreshed. Updates the topic file on disk with the new results.

    Args:
        name: The display name of the topic to refresh.

    Returns:
        The updated topic dictionary with refreshed insights and an
        updated refreshed_at timestamp.

    Raises:
        FileNotFoundError: If no topic with the given name exists.

    Example:
        >>> topic = refresh_topic("Celts early game")
        >>> print(f"Refreshed with {topic['insight_count']} insights")
        Refreshed with 23 insights
    """
    existing = get_topic(name)
    logger.info(f"Refreshing topic: '{name}'")

    results = _fetch_insights_for_topic(
        existing["query"],
        existing["source_filter"],
        TOPIC_INSIGHT_COUNT,
    )

    existing["insights"] = results
    existing["insight_count"] = len(results)
    existing["refreshed_at"] = datetime.now().isoformat()

    _save_topic_file(existing)
    logger.info(
        f"Topic '{name}' refreshed with {len(results)} insights."
    )
    return existing


def delete_topic(name: str) -> None:
    """Deletes a study topic file from disk.

    Args:
        name: The display name of the topic to delete.

    Raises:
        FileNotFoundError: If no topic with the given name exists.

    Example:
        >>> delete_topic("Celts early game")
    """
    path = _topic_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Topic '{name}' not found.")
    path.unlink()
    logger.info(f"Topic deleted: '{name}'")


def list_topics() -> list[dict]:
    """Returns a summary list of all saved study topics.

    Scans the topics directory and returns a lightweight summary of
    each topic without loading the full insight lists, for efficient
    display in the UI and CLI.

    Returns:
        A list of summary dictionaries sorted alphabetically by name,
        each containing 'name', 'query', 'insight_count',
        'created_at', and 'refreshed_at' keys. Returns an empty list
        if no topics have been created yet.

    Example:
        >>> topics = list_topics()
        >>> for t in topics:
        ...     print(t["name"], t["insight_count"])
        Celts early game 20
        Resource management 18
    """
    topic_files = sorted(TOPICS_DIR.glob("*.json"))
    summaries = []

    for f in topic_files:
        try:
            topic = json.loads(f.read_text(encoding="utf-8"))
            summaries.append({
                "name": topic["name"],
                "query": topic["query"],
                "insight_count": topic["insight_count"],
                "source_filter": topic.get("source_filter"),
                "created_at": topic["created_at"],
                "refreshed_at": topic["refreshed_at"],
            })
        except Exception as e:
            logger.warning(f"Could not load topic file {f.name}: {e}")

    return summaries