"""Database module for theGist.

This module handles all SQLite database interactions for theGist.
It manages two tables — records and ideas — and initializes the
database automatically on first import.

Public interface:
    insert_record(title, channel, url, video_id, subject, transcript) -> dict
    get_record(record_id) -> dict
    get_record_by_video_id(video_id) -> dict | None
    list_records(subject) -> list[dict]
    insert_ideas(record_id, texts) -> list[dict]
    get_ideas(record_id) -> list[dict]
    get_ideas_by_subject(subject) -> list[dict]
    delete_idea(idea_id) -> None
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from thegist.config import DATABASE_PATH


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    channel     TEXT NOT NULL,
    url         TEXT NOT NULL,
    video_id    TEXT NOT NULL UNIQUE,
    subject     TEXT NOT NULL,
    transcript  TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ideas (
    id          TEXT PRIMARY KEY,
    record_id   TEXT NOT NULL,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (record_id) REFERENCES records (id) ON DELETE CASCADE
);
"""


# ---------------------------------------------------------------------------
# Connection Management
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    """Opens and returns a connection to the theGist SQLite database.

    Enables foreign key enforcement and row factory so rows are
    returned as dictionaries rather than tuples.

    Returns:
        A configured sqlite3 Connection object.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _init_db() -> None:
    """Creates the database tables if they do not already exist.

    Called automatically when this module is first imported. Safe
    to call multiple times due to IF NOT EXISTS clauses.
    """
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(_SCHEMA)


# Run on import
_init_db()


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    """Returns the current UTC time as an ISO format string.

    Returns:
        The current UTC datetime as an ISO 8601 string.
    """
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Converts a sqlite3 Row object to a plain dictionary.

    Args:
        row: A sqlite3 Row object from a query result.

    Returns:
        A plain dictionary with column names as keys.
    """
    return dict(row)


# ---------------------------------------------------------------------------
# Public Interface — Records
# ---------------------------------------------------------------------------

def insert_record(
    title: str,
    channel: str,
    url: str,
    video_id: str,
    subject: str,
    transcript: str,
) -> dict:
    """Inserts a new video record into the database.

    Args:
        title: The YouTube video title.
        channel: The YouTube channel name.
        url: The full YouTube video URL.
        video_id: The normalized YouTube video ID used for deduplication.
        subject: The subject this video belongs to.
        transcript: The full cleaned transcript text.

    Returns:
        The newly created record as a dictionary.

    Raises:
        ValueError: If a record with the same video ID already exists.

    Example:
        >>> record = insert_record(
        ...     title="Muisca vs Celtas",
        ...     channel="Hera",
        ...     url="https://youtube.com/watch?v=abc",
        ...     video_id="abc",
        ...     subject="Muisca strategy",
        ...     transcript="Today we are playing...",
        ... )
    """
    record = {
        "id": str(uuid.uuid4()),
        "title": title,
        "channel": channel,
        "url": url,
        "video_id": video_id,
        "subject": subject,
        "transcript": transcript,
        "created_at": _now(),
    }

    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO records
                    (id, title, channel, url, video_id,
                     subject, transcript, created_at)
                VALUES
                    (:id, :title, :channel, :url, :video_id,
                     :subject, :transcript, :created_at)
                """,
                record,
            )
    except sqlite3.IntegrityError:
        raise ValueError(
            f"A record with this video ID already exists: {video_id}"
        )

    return record


def get_record(record_id: str) -> dict:
    """Retrieves a single record by its ID.

    Args:
        record_id: The unique identifier of the record.

    Returns:
        The record as a dictionary.

    Raises:
        ValueError: If no record with the given ID exists.

    Example:
        >>> record = get_record("a1b2c3d4-...")
        >>> print(record["title"])
        'Muisca vs Celtas'
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM records WHERE id = ?",
            (record_id,),
        ).fetchone()

    if row is None:
        raise ValueError(f"No record found with id: {record_id}")

    return _row_to_dict(row)


def get_record_by_video_id(video_id: str) -> dict | None:
    """Retrieves a record by its YouTube video ID.

    Used by the fetch command to check for duplicates before
    attempting to fetch a transcript regardless of URL format.

    Args:
        video_id: The YouTube video ID to look up.

    Returns:
        The record as a dictionary if found, or None if no record
        exists for the given video ID.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM records WHERE video_id = ?",
            (video_id,),
        ).fetchone()

    return _row_to_dict(row) if row else None


def list_records(subject: str | None = None) -> list[dict]:
    """Returns a list of all records, optionally filtered by subject.

    Args:
        subject: Optional subject name to filter by. If None returns
            all records across all subjects.

    Returns:
        A list of record dictionaries ordered by created_at descending.

    Example:
        >>> records = list_records("Muisca strategy")
        >>> print(len(records))
        5
    """
    with _connect() as conn:
        if subject:
            rows = conn.execute(
                "SELECT * FROM records WHERE subject = ? ORDER BY created_at DESC",
                (subject,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM records ORDER BY created_at DESC"
            ).fetchall()

    return [_row_to_dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Public Interface — Ideas
# ---------------------------------------------------------------------------

def insert_ideas(record_id: str, texts: list[str]) -> list[dict]:
    """Inserts a list of key ideas for a given record.

    Args:
        record_id: The ID of the record to attach ideas to.
        texts: A list of idea text strings to insert. Empty strings
            are ignored automatically.

    Returns:
        A list of the newly created idea dictionaries.

    Raises:
        ValueError: If record_id does not exist or texts is empty
            after filtering blank strings.

    Example:
        >>> ideas = insert_ideas(record_id, ["Cavalry archers have extra range"])
        >>> print(ideas[0]["text"])
        'Cavalry archers have extra range'
    """
    filtered = [t.strip() for t in texts if t.strip()]

    if not filtered:
        raise ValueError("No valid idea texts provided.")

    now = _now()
    ideas = [
        {
            "id": str(uuid.uuid4()),
            "record_id": record_id,
            "text": text,
            "created_at": now,
        }
        for text in filtered
    ]

    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO ideas (id, record_id, text, created_at)
            VALUES (:id, :record_id, :text, :created_at)
            """,
            ideas,
        )

    return ideas


def get_ideas(record_id: str) -> list[dict]:
    """Returns all ideas belonging to a given record.

    Args:
        record_id: The ID of the record to retrieve ideas for.

    Returns:
        A list of idea dictionaries ordered by created_at ascending.

    Example:
        >>> ideas = get_ideas("a1b2c3d4-...")
        >>> for idea in ideas:
        ...     print(idea["text"])
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM ideas WHERE record_id = ? ORDER BY created_at ASC",
            (record_id,),
        ).fetchall()

    return [_row_to_dict(row) for row in rows]


def get_ideas_by_subject(subject: str) -> list[dict]:
    """Returns all ideas across all records belonging to a subject.

    Joins the ideas and records tables to filter by subject. Used
    by the dedupe command to find candidate duplicate ideas within
    a subject.

    Args:
        subject: The subject name to retrieve ideas for.

    Returns:
        A list of idea dictionaries each including the parent record's
        title and channel for display context.

    Example:
        >>> ideas = get_ideas_by_subject("Muisca strategy")
        >>> print(len(ideas))
        47
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                ideas.id,
                ideas.record_id,
                ideas.text,
                ideas.created_at,
                records.title AS record_title,
                records.channel
            FROM ideas
            JOIN records ON ideas.record_id = records.id
            WHERE records.subject = ?
            ORDER BY ideas.created_at ASC
            """,
            (subject,),
        ).fetchall()

    return [_row_to_dict(row) for row in rows]


def delete_idea(idea_id: str) -> None:
    """Permanently deletes a single idea by its ID.

    Args:
        idea_id: The unique identifier of the idea to delete.

    Raises:
        ValueError: If no idea with the given ID exists.

    Example:
        >>> delete_idea("b2c3d4e5-...")
    """
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM ideas WHERE id = ?",
            (idea_id,),
        )

    if cursor.rowcount == 0:
        raise ValueError(f"No idea found with id: {idea_id}")