"""Database module for theGist application.

This module handles all structured data storage for the theGist
knowledge management system. It manages three types of data:

- Records: Video transcripts paired with user curated key ideas,
  stored as individual JSON files in the records directory.
- Tags: A global index mapping user defined tags to the idea IDs
  they are associated with across all records.
- Quizzes: User generated quizzes associated with specific records,
  stored as individual JSON files in the quizzes directory.

All data is persisted as JSON files on disk with no external database
dependencies required.

Typical usage:
    >>> from src.database import create_record, add_ideas, get_record
    >>> record = create_record(title, channel, url, transcript)
    >>> updated = add_ideas(record["id"], ["Idea one", "Idea two"])
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import (
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
    QUIZZES_DIR,
    RECORDS_DIR,
    TAGS_FILE,
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
# Internal Helpers — Records
# ---------------------------------------------------------------------------

def _record_path(record_id: str) -> Path:
    """Returns the file path for a given record ID.

    Args:
        record_id: The unique identifier of the record.

    Returns:
        A Path object pointing to the record JSON file location.
    """
    return RECORDS_DIR / f"{record_id}.json"


def _load_record_file(record_id: str) -> dict:
    """Loads and parses a record JSON file from disk.

    Args:
        record_id: The unique identifier of the record to load.

    Returns:
        The parsed record dictionary.

    Raises:
        FileNotFoundError: If no record with the given ID exists.
    """
    path = _record_path(record_id)
    if not path.exists():
        raise FileNotFoundError(f"Record not found: {record_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_record_file(record: dict) -> None:
    """Saves a record dictionary to disk as a JSON file.

    Args:
        record: The record dictionary to persist. Must contain an 'id' key.
    """
    path = _record_path(record["id"])
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Record saved: {record['id']}")


def _now() -> str:
    """Returns the current datetime as an ISO format string.

    Returns:
        The current datetime as an ISO format string.
    """
    return datetime.now().isoformat()


# ---------------------------------------------------------------------------
# Internal Helpers — Tags
# ---------------------------------------------------------------------------

def _load_tags() -> dict:
    """Loads the global tags index from disk.

    Returns:
        The tags dictionary mapping tag names to lists of idea ID strings.
        Returns an empty dictionary if no tags file exists yet.
    """
    if not TAGS_FILE.exists():
        return {}
    return json.loads(TAGS_FILE.read_text(encoding="utf-8"))


def _save_tags(tags: dict) -> None:
    """Saves the global tags index to disk.

    Args:
        tags: The tags dictionary to persist, mapping tag names to
            lists of idea ID strings.
    """
    TAGS_FILE.write_text(
        json.dumps(tags, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _add_tag_to_index(tag: str, idea_id: str) -> None:
    """Adds an idea ID to a tag entry in the global tags index.

    Creates the tag entry if it does not already exist. Does nothing
    if the idea ID is already associated with the tag.

    Args:
        tag: The tag name to add the idea ID to.
        idea_id: The unique idea ID to associate with the tag.
    """
    tags = _load_tags()
    if tag not in tags:
        tags[tag] = []
    if idea_id not in tags[tag]:
        tags[tag].append(idea_id)
    _save_tags(tags)


def _remove_tag_from_index(tag: str, idea_id: str) -> None:
    """Removes an idea ID from a tag entry in the global tags index.

    Removes the tag entry entirely if it has no remaining idea IDs
    after removal.

    Args:
        tag: The tag name to remove the idea ID from.
        idea_id: The unique idea ID to disassociate from the tag.
    """
    tags = _load_tags()
    if tag in tags and idea_id in tags[tag]:
        tags[tag].remove(idea_id)
        if not tags[tag]:
            del tags[tag]
    _save_tags(tags)


def _remove_idea_from_all_tags(idea_id: str) -> None:
    """Removes an idea ID from every tag it is associated with.

    Used when an idea is permanently deleted to ensure the tags index
    remains consistent.

    Args:
        idea_id: The unique idea ID to remove from all tags.
    """
    tags = _load_tags()
    updated = {
        tag: [i for i in ids if i != idea_id]
        for tag, ids in tags.items()
    }
    cleaned = {tag: ids for tag, ids in updated.items() if ids}
    _save_tags(cleaned)


# ---------------------------------------------------------------------------
# Public Interface — Records
# ---------------------------------------------------------------------------

def create_record(
    title: str,
    channel: str,
    url: str,
    transcript: str,
) -> dict:
    """Creates and saves a new video record with its transcript.

    Generates a unique ID for the record and persists it to disk.
    The record is initialised with an empty key ideas list ready
    for the user to populate.

    Args:
        title: The YouTube video title.
        channel: The YouTube channel name that uploaded the video.
        url: The full YouTube video URL.
        transcript: The full transcript text for the video.

    Returns:
        The newly created record dictionary containing the following keys:
            - id: Unique record identifier string.
            - title: The video title.
            - channel: The channel name.
            - url: The video URL.
            - transcript: The full transcript text.
            - key_ideas: Empty list ready for idea population.
            - created_at: ISO format creation timestamp.
            - updated_at: ISO format last update timestamp.

    Raises:
        ValueError: If title, channel, url, or transcript are empty.

    Example:
        >>> record = create_record("My Video", "My Channel", url, transcript)
        >>> print(record["id"])
        'a1b2c3d4-...'
    """
    if not all([title.strip(), channel.strip(), url.strip(), transcript.strip()]):
        raise ValueError(
            "Title, channel, URL, and transcript are all required "
            "to create a record."
        )

    now = _now()
    record = {
        "id": str(uuid.uuid4()),
        "title": title.strip(),
        "channel": channel.strip(),
        "url": url.strip(),
        "transcript": transcript.strip(),
        "key_ideas": [],
        "created_at": now,
        "updated_at": now,
    }

    _save_record_file(record)
    logger.info(f"Record created: '{title}' by {channel}")
    return record


def get_record(record_id: str) -> dict:
    """Retrieves a record by its unique ID.

    Args:
        record_id: The unique identifier of the record to retrieve.

    Returns:
        The full record dictionary including transcript and key ideas.

    Raises:
        FileNotFoundError: If no record with the given ID exists.

    Example:
        >>> record = get_record("a1b2c3d4-...")
        >>> print(record["title"])
        'My Video Title'
    """
    record = _load_record_file(record_id)
    logger.info(f"Loaded record: '{record['title']}'")
    return record


def list_records() -> list[dict]:
    """Returns a summary list of all saved records.

    Scans the records directory and returns lightweight summaries
    without loading full transcripts, for efficient display in the UI.

    Returns:
        A list of summary dictionaries sorted by creation date descending,
        each containing 'id', 'title', 'channel', 'url', 'idea_count',
        and 'created_at' keys. Returns an empty list if no records exist.

    Example:
        >>> records = list_records()
        >>> for r in records:
        ...     print(r["title"], r["idea_count"])
        'My Video' 12
    """
    summaries = []
    for path in RECORDS_DIR.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            summaries.append({
                "id": record["id"],
                "title": record["title"],
                "channel": record["channel"],
                "url": record["url"],
                "idea_count": len(record.get("key_ideas", [])),
                "created_at": record["created_at"],
            })
        except Exception as e:
            logger.warning(f"Could not load record {path.name}: {e}")

    return sorted(summaries, key=lambda r: r["created_at"], reverse=True)


def delete_record(record_id: str) -> None:
    """Permanently deletes a record and all its associated data.

    Removes the record file from disk and cleans up all tag associations
    for every idea in the record from the global tags index.

    Args:
        record_id: The unique identifier of the record to delete.

    Raises:
        FileNotFoundError: If no record with the given ID exists.

    Example:
        >>> delete_record("a1b2c3d4-...")
    """
    record = _load_record_file(record_id)

    for idea in record.get("key_ideas", []):
        _remove_idea_from_all_tags(idea["id"])

    _record_path(record_id).unlink()
    logger.info(f"Record deleted: '{record['title']}'")


# ---------------------------------------------------------------------------
# Public Interface — Key Ideas
# ---------------------------------------------------------------------------

def add_ideas(record_id: str, ideas: list[str]) -> dict:
    """Adds a list of key idea strings to an existing record.

    Generates a unique ID for each idea and appends them to the
    record's key ideas list. Existing ideas are preserved.

    Args:
        record_id: The unique identifier of the record to add ideas to.
        ideas: A list of key idea strings to add. Empty strings are
            ignored automatically.

    Returns:
        The updated record dictionary with the new ideas appended.

    Raises:
        FileNotFoundError: If no record with the given ID exists.
        ValueError: If the ideas list is empty after filtering blanks.

    Example:
        >>> updated = add_ideas(record_id, ["Celts infantry move faster"])
        >>> print(len(updated["key_ideas"]))
        1
    """
    record = _load_record_file(record_id)
    filtered = [i.strip() for i in ideas if i.strip()]

    if not filtered:
        raise ValueError("No valid ideas provided. Ideas cannot be empty strings.")

    for idea_text in filtered:
        record["key_ideas"].append({
            "id": str(uuid.uuid4()),
            "text": idea_text,
            "tags": [],
        })

    record["updated_at"] = _now()
    _save_record_file(record)
    logger.info(
        f"Added {len(filtered)} ideas to record: '{record['title']}'"
    )
    return record


def update_idea_text(record_id: str, idea_id: str, new_text: str) -> dict:
    """Updates the text of an existing key idea.

    Args:
        record_id: The unique identifier of the record containing the idea.
        idea_id: The unique identifier of the idea to update.
        new_text: The replacement text for the idea.

    Returns:
        The updated record dictionary.

    Raises:
        FileNotFoundError: If the record does not exist.
        ValueError: If the idea ID is not found in the record or
            new_text is empty.

    Example:
        >>> updated = update_idea_text(record_id, idea_id, "Updated text")
        >>> print(updated["key_ideas"][0]["text"])
        'Updated text'
    """
    if not new_text.strip():
        raise ValueError("Idea text cannot be empty.")

    record = _load_record_file(record_id)
    for idea in record["key_ideas"]:
        if idea["id"] == idea_id:
            idea["text"] = new_text.strip()
            record["updated_at"] = _now()
            _save_record_file(record)
            logger.info(f"Updated idea {idea_id} in record '{record['title']}'")
            return record

    raise ValueError(f"Idea ID '{idea_id}' not found in record '{record_id}'.")


def delete_idea(record_id: str, idea_id: str) -> dict:
    """Permanently deletes a key idea from a record.

    Removes the idea from the record and cleans up all its tag
    associations from the global tags index.

    Args:
        record_id: The unique identifier of the record containing the idea.
        idea_id: The unique identifier of the idea to delete.

    Returns:
        The updated record dictionary with the idea removed.

    Raises:
        FileNotFoundError: If the record does not exist.
        ValueError: If the idea ID is not found in the record.

    Example:
        >>> updated = delete_idea(record_id, idea_id)
        >>> print(len(updated["key_ideas"]))
        0
    """
    record = _load_record_file(record_id)
    original_count = len(record["key_ideas"])
    record["key_ideas"] = [
        i for i in record["key_ideas"] if i["id"] != idea_id
    ]

    if len(record["key_ideas"]) == original_count:
        raise ValueError(
            f"Idea ID '{idea_id}' not found in record '{record_id}'."
        )

    _remove_idea_from_all_tags(idea_id)
    record["updated_at"] = _now()
    _save_record_file(record)
    logger.info(f"Deleted idea {idea_id} from record '{record['title']}'")
    return record


# ---------------------------------------------------------------------------
# Public Interface — Tags
# ---------------------------------------------------------------------------

def add_tag_to_idea(record_id: str, idea_id: str, tag: str) -> dict:
    """Adds a tag to a key idea.

    Adds the tag to the idea's tag list and updates the global tags
    index. Does nothing if the idea already has the tag.

    Args:
        record_id: The unique identifier of the record containing the idea.
        idea_id: The unique identifier of the idea to tag.
        tag: The tag string to add. Will be lowercased and stripped.

    Returns:
        The updated record dictionary.

    Raises:
        FileNotFoundError: If the record does not exist.
        ValueError: If the idea ID is not found or tag is empty.

    Example:
        >>> updated = add_tag_to_idea(record_id, idea_id, "early game")
        >>> print(updated["key_ideas"][0]["tags"])
        ['early game']
    """
    tag = tag.strip().lower()
    if not tag:
        raise ValueError("Tag cannot be empty.")

    record = _load_record_file(record_id)
    for idea in record["key_ideas"]:
        if idea["id"] == idea_id:
            if tag not in idea["tags"]:
                idea["tags"].append(tag)
                _add_tag_to_index(tag, idea_id)
            record["updated_at"] = _now()
            _save_record_file(record)
            logger.info(
                f"Added tag '{tag}' to idea {idea_id} "
                f"in record '{record['title']}'"
            )
            return record

    raise ValueError(f"Idea ID '{idea_id}' not found in record '{record_id}'.")


def remove_tag_from_idea(record_id: str, idea_id: str, tag: str) -> dict:
    """Removes a tag from a key idea.

    Removes the tag from the idea's tag list and updates the global
    tags index. Does nothing if the idea does not have the tag.

    Args:
        record_id: The unique identifier of the record containing the idea.
        idea_id: The unique identifier of the idea to untag.
        tag: The tag string to remove.

    Returns:
        The updated record dictionary.

    Raises:
        FileNotFoundError: If the record does not exist.
        ValueError: If the idea ID is not found in the record.

    Example:
        >>> updated = remove_tag_from_idea(record_id, idea_id, "early game")
        >>> print(updated["key_ideas"][0]["tags"])
        []
    """
    tag = tag.strip().lower()
    record = _load_record_file(record_id)

    for idea in record["key_ideas"]:
        if idea["id"] == idea_id:
            if tag in idea["tags"]:
                idea["tags"].remove(tag)
                _remove_tag_from_index(tag, idea_id)
            record["updated_at"] = _now()
            _save_record_file(record)
            logger.info(
                f"Removed tag '{tag}' from idea {idea_id} "
                f"in record '{record['title']}'"
            )
            return record

    raise ValueError(f"Idea ID '{idea_id}' not found in record '{record_id}'.")


def list_all_tags() -> list[str]:
    """Returns a sorted list of all tags currently in use.

    Returns:
        A sorted list of unique tag name strings. Returns an empty
        list if no tags have been created yet.

    Example:
        >>> tags = list_all_tags()
        >>> print(tags)
        ['cavalry counters', 'early game', 'resource management']
    """
    tags = _load_tags()
    return sorted(tags.keys())


def get_ideas_by_tag(tag: str) -> list[dict]:
    """Retrieves all key ideas associated with a given tag.

    Looks up the idea IDs for the tag in the global index, then
    fetches each idea from its parent record along with record
    metadata for display context.

    Args:
        tag: The tag name to retrieve ideas for.

    Returns:
        A list of enriched idea dictionaries each containing the
        idea's 'id', 'text', 'tags', 'record_id', 'record_title',
        and 'channel' fields. Returns an empty list if the tag
        does not exist or has no associated ideas.

    Example:
        >>> ideas = get_ideas_by_tag("early game")
        >>> for idea in ideas:
        ...     print(idea["text"], idea["record_title"])
    """
    tags = _load_tags()
    tag = tag.strip().lower()
    idea_ids = tags.get(tag, [])

    if not idea_ids:
        return []

    enriched = []
    for record_path in RECORDS_DIR.glob("*.json"):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            for idea in record.get("key_ideas", []):
                if idea["id"] in idea_ids:
                    enriched.append({
                        "id": idea["id"],
                        "text": idea["text"],
                        "tags": idea["tags"],
                        "record_id": record["id"],
                        "record_title": record["title"],
                        "channel": record["channel"],
                    })
        except Exception as e:
            logger.warning(f"Could not load record {record_path.name}: {e}")

    return enriched


# ---------------------------------------------------------------------------
# Public Interface — Quizzes
# ---------------------------------------------------------------------------

def save_quiz(
    record_id: str,
    title: str,
    questions: list[dict],
) -> dict:
    """Saves a user generated quiz associated with a record.

    Persists a quiz containing a list of questions to disk. Each
    question is expected to contain the question text, correct answer,
    and a list of choices as generated externally by the user.

    Args:
        record_id: The unique identifier of the record the quiz is
            based on.
        title: A descriptive title for the quiz.
        questions: A list of question dictionaries. Each dictionary
            must contain 'question', 'correct_answer', and 'choices'
            keys.

    Returns:
        The saved quiz dictionary containing 'id', 'record_id',
        'title', 'question_count', 'questions', and 'created_at'.

    Raises:
        ValueError: If the questions list is empty or record_id is
            missing.

    Example:
        >>> quiz = save_quiz(record_id, "Celts Quiz", questions)
        >>> print(quiz["question_count"])
        10
    """
    if not questions:
        raise ValueError("Cannot save a quiz with no questions.")
    if not record_id.strip():
        raise ValueError("A record ID is required to save a quiz.")

    quiz = {
        "id": str(uuid.uuid4()),
        "record_id": record_id,
        "title": title.strip(),
        "question_count": len(questions),
        "questions": questions,
        "created_at": _now(),
    }

    path = QUIZZES_DIR / f"{quiz['id']}.json"
    path.write_text(
        json.dumps(quiz, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Quiz saved: '{title}' ({len(questions)} questions)")
    return quiz


def get_quiz(quiz_id: str) -> dict:
    """Retrieves a saved quiz by its unique ID.

    Args:
        quiz_id: The unique identifier of the quiz to retrieve.

    Returns:
        The full quiz dictionary including all questions.

    Raises:
        FileNotFoundError: If no quiz with the given ID exists.

    Example:
        >>> quiz = get_quiz("q1b2c3d4-...")
        >>> print(quiz["title"])
        'Celts Quiz'
    """
    path = QUIZZES_DIR / f"{quiz_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Quiz not found: {quiz_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_quizzes(record_id: Optional[str] = None) -> list[dict]:
    """Returns a summary list of all saved quizzes.

    Args:
        record_id: Optional record ID to filter quizzes by a specific
            record. If None returns all quizzes across all records.

    Returns:
        A list of quiz summary dictionaries sorted by creation date
        descending, each containing 'id', 'record_id', 'title',
        'question_count', and 'created_at'. Returns an empty list
        if no quizzes exist.

    Example:
        >>> quizzes = list_quizzes(record_id="a1b2c3d4-...")
        >>> print(quizzes[0]["title"])
        'Celts Quiz'
    """
    summaries = []
    for path in QUIZZES_DIR.glob("*.json"):
        try:
            quiz = json.loads(path.read_text(encoding="utf-8"))
            if record_id and quiz.get("record_id") != record_id:
                continue
            summaries.append({
                "id": quiz["id"],
                "record_id": quiz["record_id"],
                "title": quiz["title"],
                "question_count": quiz["question_count"],
                "created_at": quiz["created_at"],
            })
        except Exception as e:
            logger.warning(f"Could not load quiz {path.name}: {e}")

    return sorted(summaries, key=lambda q: q["created_at"], reverse=True)


def delete_quiz(quiz_id: str) -> None:
    """Permanently deletes a saved quiz.

    Args:
        quiz_id: The unique identifier of the quiz to delete.

    Raises:
        FileNotFoundError: If no quiz with the given ID exists.

    Example:
        >>> delete_quiz("q1b2c3d4-...")
    """
    path = QUIZZES_DIR / f"{quiz_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Quiz not found: {quiz_id}")
    path.unlink()
    logger.info(f"Quiz deleted: {quiz_id}")