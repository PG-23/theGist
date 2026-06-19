"""Database module for theGist.

This module handles all SQLite database interactions for theGist.
It manages four tables — records, ideas, subject_ideas, and
duplicate_pairs — and initializes the database automatically
on first import.

The database is stored as a single file at the path defined in
config.py. All public functions accept and return plain Python
dictionaries rather than database row objects to keep the interface
clean and independent of the database implementation.

Public interface:
    insert_record(title, channel, url, video_id, subject, transcript, uploaded_at) -> dict
    get_record(record_id) -> dict
    get_record_by_video_id(video_id) -> dict | None
    list_records(subject) -> list[dict]
    get_records_without_ideas(subject) -> list[dict]
    insert_ideas(record_id, texts) -> list[dict]
    insert_idea_label(idea_id, subject, label) -> None
    get_ideas(record_id) -> list[dict]
    get_active_subject_ideas(subject) -> list[dict]
    deactivate_subject_idea(idea_id, subject) -> None
    record_duplicate_pair(kept_id, removed_id, subject, similarity) -> None
    delete_idea(idea_id) -> None
    get_labeled_idea_ids(subject) -> set[str]
    delete_idea_label(idea_id, subject) -> None
    get_labeled_examples(subject) -> list[dict]
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
    uploaded_at TEXT,
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

CREATE TABLE IF NOT EXISTS subject_ideas (
    id          TEXT PRIMARY KEY,
    idea_id     TEXT NOT NULL,
    subject     TEXT NOT NULL,
    is_active   INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (idea_id) REFERENCES ideas (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS idea_labels (
    id          TEXT PRIMARY KEY,
    idea_id     TEXT NOT NULL,
    subject     TEXT NOT NULL,
    label       INTEGER NOT NULL,  -- 1 = relevant, 0 = irrelevant
    labeled_at  TEXT NOT NULL,
    FOREIGN KEY (idea_id) REFERENCES ideas (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS duplicate_pairs (
    id              TEXT PRIMARY KEY,
    kept_idea_id    TEXT NOT NULL,
    removed_idea_id TEXT NOT NULL,
    subject         TEXT NOT NULL,
    similarity      REAL NOT NULL,
    reviewed_at     TEXT NOT NULL
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
    uploaded_at: str | None = None,
) -> dict:
    """Inserts a new video record into the database.

    Args:
        title: The YouTube video title.
        channel: The YouTube channel name.
        url: The full YouTube video URL.
        video_id: The normalized YouTube video ID for deduplication.
        subject: The subject this video belongs to.
        transcript: The full cleaned transcript text.
        uploaded_at: The video upload date in YYYY-MM-DD format,
            or None if unavailable.

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
        ...     subject="Age of Empires II",
        ...     transcript="Today we are playing...",
        ...     uploaded_at="2026-05-29",
        ... )
    """
    record = {
        "id": str(uuid.uuid4()),
        "title": title,
        "channel": channel,
        "url": url,
        "video_id": video_id,
        "subject": subject,
        "uploaded_at": uploaded_at,
        "transcript": transcript,
        "created_at": _now(),
    }

    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO records
                    (id, title, channel, url, video_id,
                     subject, uploaded_at, transcript, created_at)
                VALUES
                    (:id, :title, :channel, :url, :video_id,
                     :subject, :uploaded_at, :transcript, :created_at)
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
    """
    with _connect() as conn:
        if subject:
            rows = conn.execute(
                """
                SELECT * FROM records
                WHERE subject = ?
                ORDER BY created_at DESC
                """,
                (subject,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM records ORDER BY created_at DESC"
            ).fetchall()

    return [_row_to_dict(row) for row in rows]


def get_records_without_ideas(subject: str) -> list[dict]:
    """Returns all records for a subject that have no ideas yet.

    Uses a LEFT JOIN to find records with no matching rows in the
    ideas table.

    Args:
        subject: The subject name to filter records by.

    Returns:
        A list of record dictionaries with no associated ideas,
        ordered by created_at ascending.
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT records.*
            FROM records
            LEFT JOIN ideas ON ideas.record_id = records.id
            WHERE records.subject = ?
            AND ideas.id IS NULL
            ORDER BY records.created_at ASC
            """,
            (subject,),
        ).fetchall()

    return [_row_to_dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Public Interface — Ideas
# ---------------------------------------------------------------------------

def insert_ideas(record_id: str, texts: list[str]) -> list[dict]:
    """Inserts a list of key ideas for a given record.

    Automatically populates subject_ideas in the same transaction
    so the subject pool stays consistent with the ideas table.

    Args:
        record_id: The ID of the record to attach ideas to.
        texts: A list of idea text strings to insert. Empty strings
            are ignored automatically.

    Returns:
        A list of the newly created idea dictionaries.

    Raises:
        ValueError: If record_id does not exist or texts is empty
            after filtering blank strings.
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

    # Build subject_ideas rows using the record's subject
    subject_idea_rows = []

    with _connect() as conn:
        # Look up the subject from the parent record
        row = conn.execute(
            "SELECT subject FROM records WHERE id = ?",
            (record_id,),
        ).fetchone()

        if row is None:
            raise ValueError(f"No record found with id: {record_id}")

        subject = row["subject"]

        # Insert ideas and subject_ideas atomically
        conn.executemany(
            """
            INSERT INTO ideas (id, record_id, text, created_at)
            VALUES (:id, :record_id, :text, :created_at)
            """,
            ideas,
        )

        for idea in ideas:
            subject_idea_rows.append({
                "id": str(uuid.uuid4()),
                "idea_id": idea["id"],
                "subject": subject,
            })

        conn.executemany(
            """
            INSERT INTO subject_ideas (id, idea_id, subject, is_active)
            VALUES (:id, :idea_id, :subject, 1)
            """,
            subject_idea_rows,
        )

    return ideas


def insert_idea_label(
    idea_id: str,
    subject: str,
    label: int,
) -> None:
    """Records a relevance label for an idea.

    Called by filter-ideas when the user marks an idea as relevant
    or irrelevant. Labels accumulate over time to form a training
    dataset for a future supervised classifier.

    Args:
        idea_id: The ID of the idea being labeled.
        subject: The subject the idea belongs to.
        label: 1 for relevant, 0 for irrelevant.

    Raises:
        ValueError: If label is not 0 or 1.
    """
    if label not in (0, 1):
        raise ValueError("Label must be 0 (irrelevant) or 1 (relevant).")

    row = {
        "id": str(uuid.uuid4()),
        "idea_id": idea_id,
        "subject": subject,
        "label": label,
        "labeled_at": _now(),
    }

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO idea_labels
                (id, idea_id, subject, label, labeled_at)
            VALUES
                (:id, :idea_id, :subject, :label, :labeled_at)
            """,
            row,
        )


def get_ideas(record_id: str) -> list[dict]:
    """Returns all ideas belonging to a given record.

    Args:
        record_id: The ID of the record to retrieve ideas for.

    Returns:
        A list of idea dictionaries ordered by created_at ascending.
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM ideas
            WHERE record_id = ?
            ORDER BY created_at ASC
            """,
            (record_id,),
        ).fetchall()

    return [_row_to_dict(row) for row in rows]


def get_active_subject_ideas(subject: str) -> list[dict]:
    """Returns all active ideas for a subject from the subject pool.

    Joins subject_ideas with ideas to return the full idea text
    alongside record context. Only returns ideas where is_active = 1.

    Args:
        subject: The subject name to retrieve active ideas for.

    Returns:
        A list of idea dictionaries each containing idea id, text,
        record title, channel, and uploaded_at for context.

    Example:
        >>> ideas = get_active_subject_ideas("Age of Empires II")
        >>> print(len(ideas))
        47
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                ideas.id,
                ideas.text,
                ideas.record_id,
                records.title AS record_title,
                records.channel,
                records.uploaded_at,
                subject_ideas.id AS subject_idea_id
            FROM subject_ideas
            JOIN ideas ON subject_ideas.idea_id = ideas.id
            JOIN records ON ideas.record_id = records.id
            WHERE subject_ideas.subject = ?
            AND subject_ideas.is_active = 1
            ORDER BY ideas.created_at ASC
            """,
            (subject,),
        ).fetchall()

    return [_row_to_dict(row) for row in rows]


def deactivate_subject_idea(idea_id: str, subject: str) -> None:
    """Marks an idea as inactive in the subject pool.

    Does not delete the original idea from the ideas table.
    The record's original ideas remain intact.

    Args:
        idea_id: The idea ID to deactivate in the subject pool.
        subject: The subject the idea belongs to.

    Raises:
        ValueError: If no matching active subject idea is found.
    """
    with _connect() as conn:
        cursor = conn.execute(
            """
            UPDATE subject_ideas
            SET is_active = 0
            WHERE idea_id = ?
            AND subject = ?
            AND is_active = 1
            """,
            (idea_id, subject),
        )

    if cursor.rowcount == 0:
        raise ValueError(
            f"No active subject idea found for idea_id: {idea_id}"
        )


def record_duplicate_pair(
    kept_idea_id: str,
    removed_idea_id: str,
    subject: str,
    similarity: float,
) -> None:
    """Records a duplicate pair decision in the duplicate_pairs table.

    Called when the user chooses to remove one of two similar ideas
    during a dedupe session. Provides an audit trail for future
    statistics on per-record content uniqueness.

    Args:
        kept_idea_id: The ID of the idea that was kept.
        removed_idea_id: The ID of the idea that was removed.
        subject: The subject the pair belongs to.
        similarity: The Jaccard similarity score between the two ideas.
    """
    row = {
        "id": str(uuid.uuid4()),
        "kept_idea_id": kept_idea_id,
        "removed_idea_id": removed_idea_id,
        "subject": subject,
        "similarity": similarity,
        "reviewed_at": _now(),
    }

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO duplicate_pairs
                (id, kept_idea_id, removed_idea_id, subject,
                 similarity, reviewed_at)
            VALUES
                (:id, :kept_idea_id, :removed_idea_id, :subject,
                 :similarity, :reviewed_at)
            """,
            row,
        )


def delete_idea(idea_id: str) -> None:
    """Permanently deletes a single idea by its ID.

    Args:
        idea_id: The unique identifier of the idea to delete.

    Raises:
        ValueError: If no idea with the given ID exists.
    """
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM ideas WHERE id = ?",
            (idea_id,),
        )

    if cursor.rowcount == 0:
        raise ValueError(f"No idea found with id: {idea_id}")
    

def get_labeled_idea_ids(subject: str) -> set[str]:
    """Returns the set of idea IDs already labeled for a subject.

    Used by filter-ideas to skip ideas that have already been
    reviewed in a previous session. Skipped ideas are not labeled
    so they will reappear in future sessions.

    Args:
        subject: The subject to retrieve labeled idea IDs for.

    Returns:
        A set of idea ID strings that have been labeled for
        the given subject.
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT idea_id FROM idea_labels
            WHERE subject = ?
            """,
            (subject,),
        ).fetchall()

    return {row["idea_id"] for row in rows}


def delete_idea_label(idea_id: str, subject: str) -> None:
    """Removes a label for an idea so it can be relabeled.

    Used by the undo feature in filter-ideas to reverse a labeling
    decision made earlier in the same session.

    Args:
        idea_id: The ID of the idea whose label should be removed.
        subject: The subject the label belongs to.
    """
    with _connect() as conn:
        conn.execute(
            """
            DELETE FROM idea_labels
            WHERE idea_id = ?
            AND subject = ?
            """,
            (idea_id, subject),
        )


def get_labeled_examples(subject: str) -> list[dict]:
    """Returns all labeled ideas for a subject for classifier training.

    Joins idea_labels with ideas to return the full idea text
    alongside its label for use as training data.

    Args:
        subject: The subject to retrieve labeled examples for.

    Returns:
        A list of dictionaries each containing 'idea_id', 'text',
        and 'label' keys. Label is 1 for relevant, 0 for irrelevant.

    Example:
        >>> examples = get_labeled_examples("Age of Empires II")
        >>> print(len(examples))
        646
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                idea_labels.idea_id,
                idea_labels.label,
                ideas.text
            FROM idea_labels
            JOIN ideas ON idea_labels.idea_id = ideas.id
            WHERE idea_labels.subject = ?
            ORDER BY idea_labels.labeled_at ASC
            """,
            (subject,),
        ).fetchall()

    return [_row_to_dict(row) for row in rows]