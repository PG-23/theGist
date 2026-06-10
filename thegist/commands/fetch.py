"""Fetch command for theGist.

This module implements the fetch command which reads a text file
containing YouTube video or playlist URLs and fetches transcripts
for each one. Only videos with auto-generated captions are supported.
Successfully fetched transcripts are saved as records in the database.

Usage:
    thegist fetch <links_file> --subject <subject>
"""

from pathlib import Path

from thegist.src.database import get_record_by_video_id, insert_record
from thegist.src.transcripts import get_playlist_urls, get_transcript, extract_video_id


def register(subparsers) -> None:
    """Registers the fetch command with the top level argument parser.

    Args:
        subparsers: The subparsers group from the top level parser.
    """
    parser = subparsers.add_parser(
        "fetch",
        help="Fetch transcripts from a file of YouTube URLs.",
        description=(
            "Reads a text file containing YouTube video or playlist URLs "
            "(one per line) and fetches transcripts for each. Only videos "
            "with auto-generated captions are supported. Successfully "
            "fetched transcripts are saved to the database under the "
            "given subject."
        ),
    )
    parser.add_argument(
        "links_file",
        type=Path,
        help="Path to a text file containing YouTube URLs, one per line.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="Subject name to associate all fetched transcripts with.",
    )
    parser.set_defaults(func=run)


def _is_playlist(url: str) -> bool:
    """Returns True if the URL points to a YouTube playlist.

    Args:
        url: The YouTube URL to check.

    Returns:
        True if the URL contains a playlist identifier, False otherwise.
    """
    return "playlist?list=" in url


def _read_links(links_file: Path) -> list[str]:
    """Reads and validates a links file returning a clean list of URLs.

    Skips blank lines and lines beginning with # which are treated
    as comments. Strips surrounding whitespace from each URL.

    Args:
        links_file: Path to the text file containing URLs.

    Returns:
        A list of non-empty URL strings.

    Raises:
        FileNotFoundError: If the links file does not exist.
        ValueError: If the file contains no valid URLs.
    """
    if not links_file.exists():
        raise FileNotFoundError(
            f"Links file not found: {links_file}"
        )

    urls = []
    for line in links_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)

    if not urls:
        raise ValueError(
            f"No valid URLs found in: {links_file}"
        )

    return urls


def _fetch_single(url, subject, succeeded, skipped, failed):
    video_id = extract_video_id(url)

    if video_id:
        existing = get_record_by_video_id(video_id)
        if existing:
            print(f"  ⏭ {existing['title']} (already fetched)")
            skipped.append(existing["title"])
            return

    try:
        transcript, metadata = get_transcript(url)
        insert_record(
            title=metadata["title"],
            channel=metadata["channel"],
            url=url,
            video_id=video_id or metadata["title"],
            subject=subject,
            transcript=transcript,
        )
        print(f"  ✓ {metadata['title']}")
        succeeded.append(metadata["title"])

    except ValueError as e:
        print(f"  ✗ {url}")
        print(f"    Reason: {e}")
        failed.append(url)
    except Exception as e:
        print(f"  ✗ {url}")
        print(f"    Reason: {e}")
        failed.append(url)


def run(args) -> None:
    """Executes the fetch command.

    Reads the links file, detects URL types, fetches transcripts for
    videos and expands playlists to their constituent videos, then
    prints a summary of results.

    Args:
        args: Parsed argument namespace containing links_file and subject.
    """
    try:
        urls = _read_links(args.links_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    print(f"\ntheGist — Fetching transcripts")
    print(f"File    : {args.links_file.name}")
    print(f"Subject : {args.subject}")
    print(f"URLs    : {len(urls)}\n")
    print("-" * 60)

    succeeded = []
    skipped = []
    failed = []

    for url in urls:
        if _is_playlist(url):
            print(f"Playlist: {url}")
            try:
                video_urls = get_playlist_urls(url)
                print(f"  Found {len(video_urls)} video(s)")
                for video_url in video_urls:
                    _fetch_single(
                        video_url,
                        args.subject,
                        succeeded,
                        skipped,
                        failed,
                    )
            except ValueError as e:
                print(f"  ✗ Playlist error: {e}")
                failed.append(url)
        else:
            _fetch_single(url, args.subject, succeeded, skipped, failed)

    print("-" * 60)
    print(f"\nSummary")
    print(f"  Succeeded : {len(succeeded)}")
    print(f"  Skipped   : {len(skipped)}")
    print(f"  Failed    : {len(failed)}")

    if skipped:
        print(f"\nAlready fetched:")
        for title in skipped:
            print(f"  - {title}")

    if failed:
        print(f"\nFailed (no captions or error):")
        for item in failed:
            print(f"  - {item}")

    print()