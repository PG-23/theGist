"""Insight extraction module for theGist pipeline.

This module handles extracting key insights from transcript chunks using
a locally running Ollama LLM instance. Each chunk is sent to the model
with a structured prompt and the response is parsed into a list of
discrete insight strings.

Extracted insights from all chunks are aggregated, deduplicated, and
saved to the configured insights directory as a JSON file.

Typical usage:
    >>> from src.extraction import extract_insights
    >>> from pathlib import Path
    >>> insights = extract_insights(chunks, "Example_Video_Title")
"""

import json
import logging
import time
from pathlib import Path

import ollama

from config import (
    EXTRACTION_PROMPT,
    INSIGHTS_DIR,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
    MAX_INSIGHTS_PER_CHUNK,
    OLLAMA_MODEL,
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

def _build_prompt(chunk: str) -> str:
    """Builds the extraction prompt for a given transcript chunk.

    Fills the prompt template defined in config.py with the provided
    chunk text and the configured maximum number of insights per chunk.

    Args:
        chunk: The transcript chunk string to embed in the prompt.

    Returns:
        A fully formatted prompt string ready to send to the Ollama model.

    Example:
        >>> prompt = _build_prompt("cavalry units counter archers effectively")
        >>> print(prompt[:50])
        'You are an expert knowledge extractor. Given the'
    """
    return EXTRACTION_PROMPT.format(
        chunk=chunk,
        max_insights=MAX_INSIGHTS_PER_CHUNK,
    )


def _parse_response(response: str) -> list[str]:
    """Parses the raw LLM response into a list of insight strings.

    Expects the model to return insights prefixed with a dash (-) on
    individual lines as specified in the extraction prompt. Lines that
    do not conform to this format are discarded.

    Args:
        response: The raw response string returned by the Ollama model.

    Returns:
        A list of cleaned insight strings with the leading dash and
        whitespace removed. Returns an empty list if no valid insights
        are found in the response.

    Example:
        >>> raw = "- Cavalry units counter archers\\n- Monks can convert units"
        >>> _parse_response(raw)
        ['Cavalry units counter archers', 'Monks can convert units']
    """
    insights = []
    for line in response.splitlines():
        stripped = line.strip()
        if stripped.startswith("-"):
            insight = stripped.lstrip("- ").strip()
            if insight:
                insights.append(insight)
    return insights


def _query_ollama(prompt: str, retries: int = 3, delay: float = 2.0) -> str:
    """Sends a prompt to the local Ollama model and returns the response.

    Includes retry logic to handle transient connection issues with the
    local Ollama service. Waits a fixed delay between each attempt.

    Args:
        prompt: The fully formatted prompt string to send to the model.
        retries: The number of times to retry on failure before raising.
            Defaults to 3.
        delay: The number of seconds to wait between retry attempts.
            Defaults to 2.0.

    Returns:
        The raw response string from the Ollama model.

    Raises:
        ConnectionError: If the Ollama service cannot be reached after
            all retry attempts are exhausted.
        RuntimeError: If the model returns an unexpected empty response.

    Example:
        >>> response = _query_ollama("Extract insights from: ...")
        >>> print(response[:100])
        '- Cavalry units are effective against ranged units...'
    """
    for attempt in range(1, retries + 1):
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response["message"]["content"].strip()

            if not content:
                raise RuntimeError("Ollama returned an empty response.")

            return content

        except RuntimeError:
            raise

        except Exception as e:
            logger.warning(
                f"Ollama query attempt {attempt}/{retries} failed: {e}"
            )
            if attempt < retries:
                time.sleep(delay)

    raise ConnectionError(
        f"Could not reach Ollama after {retries} attempts. "
        "Ensure Ollama is running at http://localhost:11434."
    )


def _deduplicate_insights(insights: list[str]) -> list[str]:
    """Removes duplicate insights while preserving their original order.

    Performs case insensitive deduplication to catch insights that are
    phrased identically but differ only in capitalisation across chunks.

    Args:
        insights: A list of insight strings that may contain duplicates.

    Returns:
        A deduplicated list of insight strings in their original order
        with the first occurrence of each insight retained.

    Example:
        >>> _deduplicate_insights(["Cavalry counters archers",
        ...                        "cavalry counters archers",
        ...                        "Monks can convert units"])
        ['Cavalry counters archers', 'Monks can convert units']
    """
    seen = set()
    unique = []
    for insight in insights:
        key = insight.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(insight)
    return unique


def _save_insights(insights: list[str], source_name: str) -> Path:
    """Saves extracted insights to disk as a structured JSON file.

    Persists the insights alongside metadata about the source video
    and the model used for extraction for traceability purposes.

    Args:
        insights: The deduplicated list of insight strings to save.
        source_name: The stem name of the source transcript file used
            to derive the output filename.

    Returns:
        A Path object pointing to the saved insights JSON file.

    Example:
        >>> path = _save_insights(insights, "Example_Video_Title")
        >>> print(path)
        data/insights/Example_Video_Title_insights.json
    """
    output_path = INSIGHTS_DIR / f"{source_name}_insights.json"

    payload = {
        "source": source_name,
        "model": OLLAMA_MODEL,
        "total_insights": len(insights),
        "insights": insights,
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(f"Insights saved: {output_path.name}")
    return output_path


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

def extract_insights(chunks: list[str], source_name: str) -> list[str]:
    """Extracts and aggregates insights from a list of transcript chunks.

    Iterates over each chunk, builds an extraction prompt, queries the
    local Ollama model, parses the response into discrete insights, and
    aggregates all insights across chunks. The final list is deduplicated
    and saved to the configured insights directory.

    Args:
        chunks: A list of transcript chunk strings as returned by
            chunking.chunk_transcript().
        source_name: The stem name of the source transcript file used
            for logging and output file naming.

    Returns:
        A deduplicated list of insight strings extracted across all
        chunks of the transcript.

    Raises:
        ValueError: If the chunks list is empty.
        ConnectionError: If the Ollama service cannot be reached.

    Example:
        >>> from src.extraction import extract_insights
        >>> insights = extract_insights(chunks, "Example_Video_Title")
        >>> print(f"Total insights: {len(insights)}")
        Total insights: 31
    """
    if not chunks:
        raise ValueError("Cannot extract insights from an empty chunk list.")

    logger.info(
        f"Starting extraction for: {source_name} "
        f"({len(chunks)} chunks, model={OLLAMA_MODEL})"
    )

    all_insights = []

    for i, chunk in enumerate(chunks, start=1):
        logger.info(f"Processing chunk {i}/{len(chunks)}...")
        prompt = _build_prompt(chunk)

        try:
            response = _query_ollama(prompt)
            parsed = _parse_response(response)
            logger.info(f"Chunk {i}: {len(parsed)} insights extracted.")
            all_insights.extend(parsed)

        except (ConnectionError, RuntimeError) as e:
            logger.error(f"Chunk {i} failed: {e}")
            continue

    unique_insights = _deduplicate_insights(all_insights)
    logger.info(
        f"Extraction complete. {len(unique_insights)} unique insights "
        f"extracted from {len(chunks)} chunks."
    )

    _save_insights(unique_insights, source_name)
    return unique_insights