"""Transcript chunking module for theGist pipeline.

This module handles splitting large transcript texts into smaller, 
overlapping chunks that can be processed individually by the insight
extraction stage. Fixed size chunking with overlap is used to ensure
no insights are lost at chunk boundaries.

Typical usage:
    >>> from src.chunking import chunk_transcript
    >>> chunks = chunk_transcript(Path("data/transcripts/example.txt"))
"""

import json
import logging
from pathlib import Path

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_DIR,
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
# Internal Helpers
# ---------------------------------------------------------------------------

def _load_transcript(transcript_path: Path) -> str:
    """Loads and validates a transcript file from disk.

    Args:
        transcript_path: The Path to the transcript text file to load.

    Returns:
        The full transcript content as a stripped string.

    Raises:
        FileNotFoundError: If no file exists at the given path.
        ValueError: If the file exists but contains no readable text.

    Example:
        >>> text = _load_transcript(Path("data/transcripts/example.txt"))
        >>> print(text[:100])
        'Welcome back everyone, today we are going to...'
    """
    if not transcript_path.exists():
        raise FileNotFoundError(
            f"Transcript file not found: {transcript_path}"
        )

    text = transcript_path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(
            f"Transcript file is empty: {transcript_path}"
        )

    logger.info(f"Loaded transcript: {transcript_path.name} "
                f"({len(text.split())} words)")
    return text


def _split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Splits a transcript string into overlapping word based chunks.

    Divides the transcript into fixed size chunks measured in words,
    with a configurable overlap between consecutive chunks to prevent
    insights from being lost at boundaries.

    Args:
        text: The full transcript string to split into chunks.
        chunk_size: The number of words per chunk as defined in config.
        overlap: The number of words to repeat at the start of each
            subsequent chunk to preserve context across boundaries.

    Returns:
        A list of strings where each string is one chunk of the
        transcript containing at most chunk_size words.

    Raises:
        ValueError: If chunk_size is less than or equal to overlap,
            which would result in infinite or invalid chunking.

    Example:
        >>> chunks = _split_into_chunks("one two three four five", 3, 1)
        >>> print(chunks)
        ['one two three', 'three four five']
    """
    if chunk_size <= overlap:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be greater than "
            f"overlap ({overlap}) to avoid infinite chunking."
        )

    words = text.split()
    total_words = len(words)
    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += step

    logger.info(
        f"Split transcript into {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={overlap})"
    )
    return chunks


def _save_chunks(chunks: list[str], source_name: str) -> Path:
    """Saves a list of transcript chunks to disk as a JSON file.

    Persists the chunks alongside metadata about the source transcript
    and chunking parameters for traceability and debugging purposes.

    Args:
        chunks: A list of transcript chunk strings to save.
        source_name: The stem name of the source transcript file,
            used to derive the output filename.

    Returns:
        A Path object pointing to the saved chunks JSON file.

    Example:
        >>> path = _save_chunks(chunks, "Example_Video_Title")
        >>> print(path)
        data/chunks/Example_Video_Title_chunks.json
    """
    output_path = CHUNKS_DIR / f"{source_name}_chunks.json"

    payload = {
        "source": source_name,
        "chunk_size": CHUNK_SIZE,
        "overlap": CHUNK_OVERLAP,
        "total_chunks": len(chunks),
        "chunks": chunks,
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(f"Chunks saved: {output_path.name}")
    return output_path


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

def chunk_transcript(transcript_path: Path) -> list[str]:
    """Splits a transcript file into overlapping chunks for processing.

    Loads the transcript from disk, divides it into fixed size word
    based chunks with overlap using settings from config.py, saves
    the chunks to the configured chunks directory, and returns the
    list of chunk strings for use by the extraction stage.

    Args:
        transcript_path: The Path to the transcript text file to chunk.
            Typically the output path returned by ingestion.ingest().

    Returns:
        A list of strings where each string is one overlapping chunk
        of the transcript text ready for insight extraction.

    Raises:
        FileNotFoundError: If the transcript file does not exist.
        ValueError: If the transcript is empty or chunking parameters
            are invalid.

    Example:
        >>> from pathlib import Path
        >>> from src.chunking import chunk_transcript
        >>> chunks = chunk_transcript(
        ...     Path("data/transcripts/Example_Video.txt")
        ... )
        >>> print(f"Total chunks: {len(chunks)}")
        Total chunks: 24
    """
    logger.info(f"Starting chunking for: {transcript_path.name}")

    text = _load_transcript(transcript_path)
    chunks = _split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    _save_chunks(chunks, transcript_path.stem)

    logger.info(f"Chunking complete. {len(chunks)} chunks ready for extraction.")
    return chunks