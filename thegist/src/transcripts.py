"""Transcript fetching module for theGist.

This module handles retrieving transcripts and metadata from YouTube
videos using yt-dlp. It provides two public functions — one for
fetching a single video transcript and one for expanding a playlist
into individual video URLs.

No file saving or record creation is performed here. The caller is
responsible for deciding what to do with the returned data.
"""

import re
import tempfile
from pathlib import Path

import yt_dlp
from urllib.parse import urlparse, parse_qs


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _fetch_metadata(url: str) -> dict:
    """Fetches metadata for a YouTube video without downloading media.

    Args:
        url: The full YouTube video URL.

    Returns:
        A dictionary containing 'title', 'channel', and 'url' keys.

    Raises:
        ValueError: If the URL is invalid or the video is unavailable.
    """
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "title": info.get("title", info.get("id", "unknown")),
                "channel": info.get("uploader", "Unknown Channel"),
                "url": url,
            }
    except yt_dlp.utils.DownloadError as e:
        raise ValueError(f"Could not retrieve metadata: {e}")


def _fetch_vtt(url: str, output_stem: Path) -> bool:
    """Downloads the auto-generated VTT caption file for a video.

    Args:
        url: The full YouTube video URL.
        output_stem: The file path stem to save the VTT file to.
            yt-dlp will append the language and .vtt extension.

    Returns:
        True if a caption file was successfully downloaded,
        False if no captions were available.
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "vtt",
        "outtmpl": str(output_stem),
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        vtt_file = Path(str(output_stem) + ".en.vtt")
        return vtt_file.exists()

    except Exception:
        return False


def _clean_vtt(vtt_text: str) -> str:
    """Strips VTT formatting leaving only plain spoken text.

    Removes the VTT header, timestamp lines, HTML tags, and duplicate
    lines caused by overlapping caption cues.

    Args:
        vtt_text: The raw VTT formatted string to clean.

    Returns:
        A cleaned plain text string of the spoken content.
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


def _sanitize_filename(title: str) -> str:
    """Converts a video title into a safe filename string.

    Args:
        title: The raw video title to sanitize.

    Returns:
        A sanitized string safe for use as a filename, maximum 80
        characters with spaces replaced by underscores.
    """
    sanitized = re.sub(r"[^\w\s-]", "", title)
    sanitized = re.sub(r"\s+", "_", sanitized).strip("_")
    return sanitized[:80]


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

def extract_video_id(url: str) -> str | None:
    """Extracts the YouTube video ID from any YouTube URL format.

    Handles standard, short, and embed URL formats.

    Args:
        url: A YouTube URL in any common format.

    Returns:
        The video ID string if found, or None if the URL format
        is not recognized.

    Example:
        >>> extract_video_id("https://youtu.be/NDzwpz78qfE?si=abc")
        'NDzwpz78qfE'
        >>> extract_video_id("https://www.youtube.com/watch?v=NDzwpz78qfE")
        'NDzwpz78qfE'
    """
    parsed = urlparse(url)

    # Standard: youtube.com/watch?v=VIDEO_ID
    if "youtube.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        ids = qs.get("v")
        if ids:
            return ids[0]

    # Short: youtu.be/VIDEO_ID
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/").split("?")[0]

    return None


def get_transcript(url: str) -> tuple[str, dict]:
    """Fetches the transcript and metadata for a single YouTube video.

    Attempts to download the auto-generated caption file for the given
    URL. All processing is done in a temporary directory — no files are
    saved to disk by this function.

    Args:
        url: The full YouTube video URL to fetch.

    Returns:
        A tuple of (transcript_text, metadata) where transcript_text
        is the cleaned plain text transcript and metadata is a
        dictionary containing 'title', 'channel', and 'url'.

    Raises:
        ValueError: If the video has no auto-generated captions or
            the URL is invalid.
        RuntimeError: If the caption file cannot be processed.

    Example:
        >>> text, meta = get_transcript("https://youtube.com/watch?v=...")
        >>> print(meta["title"])
        'Example Video Title'
    """
    metadata = _fetch_metadata(url)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_stem = Path(tmp_dir) / "transcript"

        success = _fetch_vtt(url, output_stem)

        if not success:
            raise ValueError(
                f"No auto-generated captions available for: "
                f"'{metadata['title']}'"
            )

        vtt_file = Path(tmp_dir) / "transcript.en.vtt"

        try:
            raw = vtt_file.read_text(encoding="utf-8")
        except Exception as e:
            raise RuntimeError(
                f"Could not read caption file: {e}"
            )

    transcript = _clean_vtt(raw)

    if not transcript.strip():
        raise ValueError(
            f"Transcript was empty after cleaning for: "
            f"'{metadata['title']}'"
        )

    return transcript, metadata


def get_playlist_urls(playlist_url: str) -> list[str]:
    """Expands a YouTube playlist URL into individual video URLs.

    Fetches playlist metadata without downloading any media and
    returns a list of video URLs for the caller to process.

    Args:
        playlist_url: The full YouTube playlist URL to expand.

    Returns:
        A list of individual video URL strings from the playlist.

    Raises:
        ValueError: If the playlist URL is invalid, inaccessible,
            or contains no videos.

    Example:
        >>> urls = get_playlist_urls("https://youtube.com/playlist?...")
        >>> print(len(urls))
        12
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise ValueError(
            f"Could not access playlist: {e}"
        )

    entries = info.get("entries", [])
    if not entries:
        raise ValueError(
            f"No videos found in playlist: {playlist_url}"
        )

    urls = []
    for entry in entries:
        url = entry.get("url") or entry.get("webpage_url")
        if url:
            urls.append(url)

    return urls