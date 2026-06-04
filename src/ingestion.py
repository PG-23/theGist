"""Transcript ingestion module for theGist pipeline.

This module handles retrieving transcripts from YouTube videos using yt-dlp.
It first attempts to fetch an existing auto-generated transcript directly,
falling back to downloading the audio and transcribing it locally via Whisper
if no transcript is available.

Typical usage:
    >>> from ingestion import ingest
    >>> transcript_path = ingest("https://www.youtube.com/watch?v=example")
"""

import logging
import re
import subprocess
import tempfile
from pathlib import Path

import whisper
import yt_dlp

from config import (
    TRANSCRIPTS_DIR,
    WHISPER_LANGUAGE,
    WHISPER_MODEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
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

def _sanitize_filename(title: str) -> str:
    """Converts a video title into a safe filename string.

    Removes or replaces characters that are invalid in filenames across
    Windows, macOS, and Linux filesystems.

    Args:
        title: The raw video title string to sanitize.

    Returns:
        A sanitized string safe for use as a filename, with spaces
        replaced by underscores and a maximum length of 80 characters.

    Example:
        >>> _sanitize_filename("AoE2: Best Cavalry Tips! (2024)")
        'AoE2_Best_Cavalry_Tips_2024'
    """
    sanitized = re.sub(r"[^\w\s-]", "", title)
    sanitized = re.sub(r"\s+", "_", sanitized).strip("_")
    return sanitized[:80]


def _fetch_video_metadata(url: str) -> dict:
    """Retrieves metadata for a YouTube video without downloading media.

    Args:
        url: The full YouTube video URL to retrieve metadata for.

    Returns:
        A dictionary containing video metadata fields including
        'title', 'id', and 'subtitles'.

    Raises:
        ValueError: If the URL is invalid or the video is unavailable.

    Example:
        >>> meta = _fetch_video_metadata("https://www.youtube.com/watch?v=example")
        >>> print(meta["title"])
        'Example Video Title'
    """
    ydl_opts = {"quiet": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info
    except yt_dlp.utils.DownloadError as e:
        raise ValueError(f"Could not retrieve metadata for URL: {url}. Reason: {e}")


def _fetch_transcript(url: str, output_path: Path) -> bool:
    """Attempts to download an auto-generated transcript via yt-dlp.

    Tries to retrieve the auto-generated English subtitle file for the
    given video. If successful, the transcript is saved as a plain text
    file at the specified output path.

    Args:
        url: The full YouTube video URL to fetch the transcript from.
        output_path: The Path where the transcript text file will be saved.

    Returns:
        True if a transcript was successfully fetched and saved,
        False if no transcript was available for the video.

    Example:
        >>> success = _fetch_transcript(
        ...     "https://www.youtube.com/watch?v=example",
        ...     Path("data/transcripts/example.txt")
        ... )
        >>> print(success)
        True
    """
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "writeautomaticsub": True,
        "subtitleslangs": [WHISPER_LANGUAGE],
        "subtitlesformat": "vtt",
        "outtmpl": str(output_path.with_suffix("")),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # yt-dlp saves subtitles with a language suffix e.g. filename.en.vtt
        vtt_file = output_path.with_suffix(f".{WHISPER_LANGUAGE}.vtt")
        if not vtt_file.exists():
            logger.info("No auto-generated transcript found for video.")
            return False

        # Strip VTT formatting tags and write clean plain text
        raw = vtt_file.read_text(encoding="utf-8")
        clean = _clean_vtt(raw)
        output_path.write_text(clean, encoding="utf-8")
        vtt_file.unlink()  # Remove the intermediate VTT file

        logger.info(f"Transcript fetched successfully: {output_path.name}")
        return True

    except Exception as e:
        logger.warning(f"Transcript fetch failed: {e}")
        return False


def _clean_vtt(vtt_text: str) -> str:
    """Strips VTT formatting from a subtitle file leaving plain text.

    Removes the VTT header, timestamp lines, HTML tags, and duplicate
    lines that result from overlapping subtitle cues.

    Args:
        vtt_text: The raw VTT formatted subtitle string to clean.

    Returns:
        A cleaned plain text string containing only the spoken content
        with duplicate lines removed.

    Example:
        >>> raw = "WEBVTT\\n\\n00:00:01.000 --> 00:00:03.000\\nHello world"
        >>> _clean_vtt(raw)
        'Hello world'
    """
    lines = vtt_text.splitlines()
    cleaned = []
    seen = set()

    for line in lines:
        # Skip VTT header, blank lines, and timestamp lines
        if not line.strip():
            continue
        if line.startswith("WEBVTT"):
            continue
        if re.match(r"^\d{2}:\d{2}:\d{2}", line):
            continue
        if line.startswith("Kind:"):
            continue
        if line.startswith("Language:"):
            continue

        # Strip HTML tags such as <c> and <i>
        line = re.sub(r"<[^>]+>", "", line).strip()

        # Skip empty lines and duplicates from overlapping cues
        if line and line not in seen:
            seen.add(line)
            cleaned.append(line)

    return " ".join(cleaned)


def _transcribe_audio(url: str, output_path: Path) -> None:
    """Downloads audio and transcribes it locally using Whisper.

    Used as a fallback when no auto-generated transcript is available.
    Downloads the best available audio stream, runs it through the local
    Whisper model, and saves the resulting transcript as plain text.

    Args:
        url: The full YouTube video URL to download audio from.
        output_path: The Path where the transcript text file will be saved.

    Raises:
        RuntimeError: If the audio download or Whisper transcription fails.

    Example:
        >>> _transcribe_audio(
        ...     "https://www.youtube.com/watch?v=example",
        ...     Path("data/transcripts/example.txt")
        ... )
    """
    logger.info("Falling back to Whisper transcription. Downloading audio...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = Path(tmp_dir) / "audio.mp3"

        ydl_opts = {
            "quiet": True,
            "format": "bestaudio/best",
            "outtmpl": str(audio_path.with_suffix("")),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            raise RuntimeError(f"Audio download failed: {e}")

        logger.info(f"Audio downloaded. Running Whisper ({WHISPER_MODEL}) transcription...")

        try:
            model = whisper.load_model(WHISPER_MODEL)
            result = model.transcribe(
                str(audio_path),
                language=WHISPER_LANGUAGE,
                verbose=False,
            )
            transcript = result["text"].strip()
            output_path.write_text(transcript, encoding="utf-8")
            logger.info(f"Whisper transcription complete: {output_path.name}")
        except Exception as e:
            raise RuntimeError(f"Whisper transcription failed: {e}")


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

def ingest(url: str) -> Path:
    """Ingests a YouTube video transcript into the theGist pipeline.

    Retrieves the transcript for the given YouTube video URL by first
    attempting to fetch the auto-generated subtitle file via yt-dlp.
    If no subtitle is available, falls back to downloading the audio
    and transcribing it locally using Whisper.

    The resulting transcript is saved as a plain text file in the
    configured transcripts directory defined in config.py.

    Args:
        url: The full YouTube video URL to ingest. Must be a publicly
            accessible video with either auto-generated captions or
            accessible audio.

    Returns:
        A Path object pointing to the saved transcript text file.

    Raises:
        ValueError: If the provided URL is invalid or the video is
            unavailable.
        RuntimeError: If both transcript fetching and Whisper
            transcription fail.

    Example:
        >>> from ingestion import ingest
        >>> path = ingest("https://www.youtube.com/watch?v=example")
        >>> print(path)
        data/transcripts/Example_Video_Title.txt
    """
    logger.info(f"Starting ingestion for: {url}")

    # Retrieve video metadata to build a meaningful output filename
    metadata = _fetch_video_metadata(url)
    title = metadata.get("title", metadata.get("id", "unknown_video"))
    safe_title = _sanitize_filename(title)
    output_path = TRANSCRIPTS_DIR / f"{safe_title}.txt"

    # Skip ingestion if transcript already exists locally
    if output_path.exists():
        logger.info(f"Transcript already exists, skipping download: {output_path.name}")
        return output_path

    # Attempt fast path: fetch existing auto-generated transcript
    success = _fetch_transcript(url, output_path)

    # Fall back to Whisper transcription if no transcript was available
    if not success:
        _transcribe_audio(url, output_path)

    return output_path