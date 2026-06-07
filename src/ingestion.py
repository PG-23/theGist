"""Transcript ingestion module for theGist application.

This module handles retrieving transcripts from YouTube videos using yt-dlp.
Only videos with auto-generated captions are supported. Videos without
captions are skipped with a clear error message.

Typical usage:
    >>> from src.ingestion import ingest, ingest_playlist
    >>> transcript_path = ingest("https://www.youtube.com/watch?v=example")
"""

import logging
import re
from pathlib import Path

import yt_dlp

from config import (
    CAPTIONS_ONLY,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
    TRANSCRIPTS_DIR,
    WHISPER_LANGUAGE,
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
        'title', 'id', 'uploader', and 'subtitles'.

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

        vtt_file = output_path.with_suffix(f".{WHISPER_LANGUAGE}.vtt")
        if not vtt_file.exists():
            logger.info("No auto-generated transcript found for video.")
            return False

        raw = vtt_file.read_text(encoding="utf-8")
        clean = _clean_vtt(raw)
        output_path.write_text(clean, encoding="utf-8")
        vtt_file.unlink()

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
        if not line.strip():
            continue
        if line.startswith("WEBVTT"):
            continue
        if line.startswith("Kind:"):
            continue
        if line.startswith("Language:"):
            continue
        if re.match(r"^\d{2}:\d{2}:\d{2}", line):
            continue

        line = re.sub(r"<[^>]+>", "", line).strip()

        if line and line not in seen:
            seen.add(line)
            cleaned.append(line)

    return " ".join(cleaned)


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

def ingest(url: str) -> tuple[Path, dict]:
    """Ingests a YouTube video transcript into theGist.

    Retrieves the transcript for the given YouTube video URL using
    auto-generated captions only. Saves the transcript as a plain
    text file and returns both the file path and video metadata
    for use in record creation.

    Args:
        url: The full YouTube video URL to ingest. Must be a publicly
            accessible video with auto-generated captions enabled.

    Returns:
        A tuple containing:
            - Path: The path to the saved transcript text file.
            - dict: Video metadata including title, uploader, and url.

    Raises:
        ValueError: If the URL is invalid, the video is unavailable,
            or no auto-generated captions exist and CAPTIONS_ONLY
            is True.

    Example:
        >>> path, meta = ingest("https://www.youtube.com/watch?v=example")
        >>> print(meta["title"])
        'Example Video Title'
    """
    logger.info(f"Starting ingestion for: {url}")

    metadata = _fetch_video_metadata(url)
    title = metadata.get("title", metadata.get("id", "unknown_video"))
    uploader = metadata.get("uploader", "Unknown Channel")
    safe_title = _sanitize_filename(title)
    output_path = TRANSCRIPTS_DIR / f"{safe_title}.txt"

    if output_path.exists():
        logger.info(f"Transcript already exists, skipping download: {output_path.name}")
        return output_path, {
            "title": title,
            "uploader": uploader,
            "url": url,
        }

    success = _fetch_transcript(url, output_path)

    if not success:
        if CAPTIONS_ONLY:
            logger.warning(
                f"No captions found for: {title}. "
                f"Skipping — set CAPTIONS_ONLY=False in config.py "
                f"to enable Whisper transcription fallback."
            )
            raise ValueError(
                f"No auto-generated captions available for: '{title}'. "
                f"Choose a video with captions enabled for best quality."
            )

    return output_path, {
        "title": title,
        "uploader": uploader,
        "url": url,
    }


def ingest_playlist(playlist_url: str) -> dict:
    """Ingests all caption-enabled videos from a YouTube playlist.

    Fetches all video URLs from the provided playlist, then runs
    ingestion on each video sequentially. Videos without captions
    are counted as failed and skipped automatically.

    Args:
        playlist_url: The full YouTube playlist URL to ingest. Works
            with any publicly accessible playlist regardless of owner.

    Returns:
        A summary dictionary containing the following keys:
            - total: Total number of videos found in the playlist.
            - succeeded: List of tuples (Path, metadata dict) for
              successful ingestions.
            - skipped: List of video titles skipped as already ingested.
            - failed: List of video titles that failed or had no captions.

    Raises:
        ValueError: If the playlist URL is invalid or inaccessible.

    Example:
        >>> summary = ingest_playlist("https://www.youtube.com/playlist?list=...")
        >>> print(f"Succeeded: {len(summary['succeeded'])}")
        Succeeded: 12
    """
    logger.info(f"Fetching playlist metadata: {playlist_url}")

    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "skip_download": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            playlist_info = ydl.extract_info(playlist_url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise ValueError(
            f"Could not access playlist: {playlist_url}. Reason: {e}"
        )

    entries = playlist_info.get("entries", [])
    if not entries:
        raise ValueError(f"No videos found in playlist: {playlist_url}")

    total = len(entries)
    logger.info(f"Playlist contains {total} videos. Starting ingestion...")

    succeeded = []
    skipped = []
    failed = []

    for i, entry in enumerate(entries, start=1):
        video_url = entry.get("url") or entry.get("webpage_url")
        video_title = entry.get("title", f"video_{i}")

        if not video_url:
            logger.warning(f"Video {i}/{total}: No URL found, skipping.")
            failed.append(video_title)
            continue

        safe_title = _sanitize_filename(video_title)
        expected_path = TRANSCRIPTS_DIR / f"{safe_title}.txt"

        if expected_path.exists():
            logger.info(
                f"Video {i}/{total}: Already ingested, skipping — "
                f"{video_title}"
            )
            skipped.append(video_title)
            continue

        logger.info(f"Video {i}/{total}: Ingesting — {video_title}")

        try:
            result = ingest(video_url)
            succeeded.append(result)
            logger.info(f"Video {i}/{total}: Success — {video_title}")
        except Exception as e:
            logger.error(
                f"Video {i}/{total}: Failed — {video_title}. Reason: {e}"
            )
            failed.append(video_title)

    logger.info(
        f"Playlist ingestion complete. "
        f"Total: {total} | "
        f"Succeeded: {len(succeeded)} | "
        f"Skipped: {len(skipped)} | "
        f"Failed: {len(failed)}"
    )

    return {
        "total": total,
        "succeeded": succeeded,
        "skipped": skipped,
        "failed": failed,
    }