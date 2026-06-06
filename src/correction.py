"""Transcript correction module for theGist pipeline.

This module provides domain adaptive transcript correction using a locally
running Ollama LLM. It corrects misrecognized domain specific terminology
in auto-generated transcripts, improving the quality and accuracy of
downstream insight extraction.

Domain vocabulary and correction prompts are configured centrally in
config.py. Pre-defined domains include Age of Empires 2 and Computer
Science, with support for user defined custom domains.

Typical usage:
    >>> from src.correction import correct_transcript
    >>> corrected = correct_transcript(raw_transcript, domain="aoe2")
"""

import logging
from pathlib import Path
from typing import Optional

import ollama

from config import (
    ACTIVE_DOMAIN,
    DOMAIN_VOCABULARY,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
    OLLAMA_MODEL,
    TRANSCRIPT_CORRECTION_ENABLED,
    WHISPER_DOMAIN_PROMPT_ENABLED,
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

def _get_domain_config(domain: str) -> dict:
    """Retrieves the configuration dictionary for a given domain key.

    Args:
        domain: The domain key string to look up in DOMAIN_VOCABULARY.
            Must match a key defined in config.py.

    Returns:
        The domain configuration dictionary containing 'name',
        'vocabulary', and 'correction_prompt' keys.

    Raises:
        ValueError: If the domain key is not found in DOMAIN_VOCABULARY.

    Example:
        >>> config = _get_domain_config("aoe2")
        >>> print(config["name"])
        'Age of Empires 2'
    """
    if domain not in DOMAIN_VOCABULARY:
        available = list(DOMAIN_VOCABULARY.keys())
        raise ValueError(
            f"Unknown domain: '{domain}'. "
            f"Available domains: {available}. "
            f"Add a custom domain to DOMAIN_VOCABULARY in config.py."
        )
    return DOMAIN_VOCABULARY[domain]


def _build_correction_prompt(raw_transcript: str, domain_config: dict) -> str:
    """Builds the full correction prompt for a given transcript and domain.

    Combines the domain specific correction instructions from config.py
    with the raw transcript text to form a complete prompt for the LLM.

    Args:
        raw_transcript: The raw transcript string to be corrected.
        domain_config: The domain configuration dictionary containing
            the 'correction_prompt' instruction string.

    Returns:
        A fully formatted correction prompt string ready to send to
        the Ollama model.

    Example:
        >>> prompt = _build_correction_prompt(raw_text, domain_config)
        >>> print(prompt[:60])
        'The following is an auto-generated transcript from an Age...'
    """
    return (
        f"{domain_config['correction_prompt']}\n\n"
        f"Transcript:\n{raw_transcript}"
    )


def _chunk_for_correction(text: str, max_words: int = 800) -> list[str]:
    """Splits a transcript into chunks suitable for the correction pass.

    Uses a larger chunk size than the insight extraction stage since
    the correction LLM needs sufficient context to identify and fix
    terminology errors accurately without losing surrounding meaning.

    Args:
        text: The full transcript string to split into correction chunks.
        max_words: Maximum number of words per correction chunk.
            Defaults to 800 to balance context and model capacity.

    Returns:
        A list of transcript chunk strings ready for individual
        correction passes.

    Example:
        >>> chunks = _chunk_for_correction("word " * 1000)
        >>> print(len(chunks))
        2
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks


def _correct_chunk(chunk: str, domain_config: dict, retries: int = 2) -> str:
    """Applies LLM based correction to a single transcript chunk.

    Sends the chunk to the local Ollama model with domain specific
    correction instructions and returns the corrected text. Falls back
    to the original chunk if correction fails after all retries.

    Args:
        chunk: A transcript chunk string to correct.
        domain_config: The domain configuration dictionary containing
            correction instructions.
        retries: Number of retry attempts on failure. Defaults to 2.

    Returns:
        The corrected transcript chunk string, or the original chunk
        if the correction call fails after all retries.

    Example:
        >>> corrected = _correct_chunk(
        ...     "there's two men I'm going minor arm see with kels",
        ...     domain_config
        ... )
        >>> print(corrected)
        "verse Cuman I'm going men-at-arms here with Celts"
    """
    prompt = _build_correction_prompt(chunk, domain_config)

    for attempt in range(1, retries + 1):
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            corrected = response["message"]["content"].strip()
            if corrected:
                return corrected
        except Exception as e:
            logger.warning(
                f"Correction attempt {attempt}/{retries} failed: {e}"
            )

    logger.warning("Correction failed after all retries. Using original chunk.")
    return chunk


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

def get_whisper_prompt(domain: Optional[str] = None) -> Optional[str]:
    """Returns a Whisper initial prompt string for vocabulary biasing.

    Builds a comma separated string of domain specific vocabulary terms
    to pass as Whisper's initial_prompt parameter, biasing transcription
    toward correct recognition of domain specific words. Returns None
    if Whisper domain prompting is disabled in config.py.

    Args:
        domain: The domain key to retrieve vocabulary for. If None,
            falls back to ACTIVE_DOMAIN from config.py. Returns None
            if no domain is configured or the feature is disabled.

    Returns:
        A comma separated vocabulary string for use as a Whisper prompt,
        or None if the feature is disabled or no domain is configured.

    Example:
        >>> prompt = get_whisper_prompt("aoe2")
        >>> print(prompt[:50])
        'Celts, Britons, Franks, Teutons, Cumans, Lithuani...'
    """
    if not WHISPER_DOMAIN_PROMPT_ENABLED:
        logger.info("Whisper domain prompt disabled in config.")
        return None

    active = domain or ACTIVE_DOMAIN
    if not active:
        return None

    try:
        config = _get_domain_config(active)
        return ", ".join(config["vocabulary"])
    except ValueError:
        logger.warning(f"Domain '{active}' not found. Whisper prompt disabled.")
        return None


def correct_transcript(
    raw_transcript: str,
    domain: Optional[str] = None,
) -> str:
    """Applies domain adaptive LLM correction to a full transcript.

    Splits the transcript into correction sized chunks, applies the
    domain specific LLM correction pass to each chunk, and reassembles
    the corrected transcript. Returns the original transcript unchanged
    if the correction feature is disabled in config.py or no domain
    is active.

    Args:
        raw_transcript: The full raw transcript string to correct.
        domain: The domain key to use for correction. If None, falls
            back to ACTIVE_DOMAIN from config.py. If no domain is
            configured, the original transcript is returned unchanged.

    Returns:
        The domain corrected transcript string with misrecognized
        terminology replaced, or the original transcript if correction
        is disabled or no domain is active.

    Example:
        >>> corrected = correct_transcript(raw_transcript, domain="aoe2")
        >>> print(corrected[:100])
        'verse Cuman I am going men-at-arms here with Celts should be a nice one...'
    """
    if not TRANSCRIPT_CORRECTION_ENABLED:
        logger.info("Transcript correction pass disabled in config.")
        return raw_transcript

    active = domain or ACTIVE_DOMAIN

    if not active:
        logger.info("No active domain configured. Skipping correction pass.")
        return raw_transcript

    try:
        domain_config = _get_domain_config(active)
    except ValueError as e:
        logger.warning(f"Correction skipped: {e}")
        return raw_transcript

    logger.info(
        f"Running correction pass for domain: "
        f"'{domain_config['name']}' ({OLLAMA_MODEL})"
    )

    chunks = _chunk_for_correction(raw_transcript)
    corrected_chunks = []

    for i, chunk in enumerate(chunks, start=1):
        logger.info(f"Correcting chunk {i}/{len(chunks)}...")
        corrected = _correct_chunk(chunk, domain_config)
        corrected_chunks.append(corrected)

    corrected_transcript = " ".join(corrected_chunks)
    logger.info("Correction pass complete.")
    return corrected_transcript


def list_available_domains() -> list[dict]:
    """Returns a list of all available pre-defined domain configurations.

    Provides a summary of each configured domain for display in the
    UI or CLI without exposing the full vocabulary lists.

    Returns:
        A list of dictionaries each containing 'key' and 'name' fields
        for every domain defined in DOMAIN_VOCABULARY.

    Example:
        >>> domains = list_available_domains()
        >>> for d in domains:
        ...     print(d["key"], d["name"])
        aoe2 Age of Empires 2
        computer_science Computer Science
    """
    return [
        {"key": key, "name": val["name"]}
        for key, val in DOMAIN_VOCABULARY.items()
    ]