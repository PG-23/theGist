"""Command line entry point for theGist pipeline.

This module provides a CLI interface for running the full theGist
pipeline — from transcript ingestion through to insight extraction,
storage, and knowledge testing. Each stage can be run individually
or the full pipeline can be executed in a single command.

Usage:
    # Run the full pipeline on a YouTube URL
    python main.py ingest <url>

    # Run extraction and storage on an already ingested transcript
    python main.py extract <transcript_stem>

    # Query the knowledge base semantically
    python main.py query "<query string>" [--source <source_name>]

    # Start an interactive quiz session
    python main.py quiz <source_name>

    # Run the full pipeline and start a quiz in one command
    python main.py run <url>
"""

import argparse
import logging
import sys
from pathlib import Path

from config import LOG_DATE_FORMAT, LOG_FORMAT, LOG_LEVEL
from src.chunking import chunk_transcript
from src.extraction import extract_insights
from src.ingestion import ingest, ingest_playlist
from src.learning import run_quiz_session
from src.storage import query_insights, store_insights
from src.topics import (
    create_topic,
    delete_topic,
    list_topics,
    refresh_topic,
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
# Pipeline Stage Commands
# ---------------------------------------------------------------------------

def cmd_topic(args: argparse.Namespace) -> None:
    """Manages study topics from the command line.

    Supports creating, listing, refreshing, and deleting study topics
    via subactions passed as arguments.

    Args:
        args: Parsed argument namespace containing:
            - action: One of 'create', 'list', 'refresh', or 'delete'.
            - name: Topic name (required for create, refresh, delete).
            - query: Search query (required for create).
            - source: Optional source filter (used with create).
    """
    action = args.action

    if action == "list":
        topics = list_topics()
        if not topics:
            print("\nNo study topics found. Create one with:")
            print(
                "  python main.py topic create "
                "\"<name>\" \"<query>\"\n"
            )
            return
        print(f"\n{'='*60}")
        print(f"  Study Topics ({len(topics)})")
        print(f"{'='*60}")
        for t in topics:
            print(f"\n  {t['name']}")
            print(f"  Query    : {t['query']}")
            print(f"  Insights : {t['insight_count']}")
            print(f"  Refreshed: {t['refreshed_at'][:10]}")
        print()

    elif action == "create":
        if not args.name or not args.query:
            print("\nError: --name and --query are required for create.\n")
            sys.exit(1)
        try:
            topic = create_topic(
                name=args.name,
                query=args.query,
                source_name=getattr(args, "source", None),
            )
            print(f"\nTopic created: '{topic['name']}'")
            print(f"Insights     : {topic['insight_count']}")
            print(f"Query        : {topic['query']}\n")
        except ValueError as e:
            print(f"\nError: {e}\n")
            sys.exit(1)

    elif action == "refresh":
        if not args.name:
            print("\nError: --name is required for refresh.\n")
            sys.exit(1)
        try:
            topic = refresh_topic(args.name)
            print(f"\nTopic refreshed: '{topic['name']}'")
            print(f"Insights       : {topic['insight_count']}\n")
        except FileNotFoundError as e:
            print(f"\nError: {e}\n")
            sys.exit(1)

    elif action == "delete":
        if not args.name:
            print("\nError: --name is required for delete.\n")
            sys.exit(1)
        try:
            delete_topic(args.name)
            print(f"\nTopic deleted: '{args.name}'\n")
        except FileNotFoundError as e:
            print(f"\nError: {e}\n")
            sys.exit(1)


def cmd_playlist(args: argparse.Namespace) -> None:
    """Ingests all videos from a YouTube playlist and runs the full pipeline.

    Fetches all video URLs from the provided playlist URL and runs the
    complete theGist pipeline — ingestion, chunking, extraction, and
    storage — on each video sequentially. Already processed videos are
    skipped automatically. Prints a reminder for large playlists and a
    summary of results on completion.

    Args:
        args: Parsed argument namespace containing:
            - url: The YouTube playlist URL to process.
    """
    logger.info("Running playlist ingestion...")

    try:
        # Fetch playlist metadata first to warn on large playlists
        import yt_dlp
        ydl_opts = {
            "quiet": True,
            "extract_flat": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            playlist_info = ydl.extract_info(args.url, download=False)

        total = len(playlist_info.get("entries", []))

        if total >= 10:
            print(f"\nNote: This playlist contains {total} videos.")
            print(
                "The full pipeline will run on each video sequentially, "
                "including LLM insight extraction. This may take a "
                "significant amount of time for large playlists.\n"
            )

        summary = ingest_playlist(args.url)

        print(f"\nPlaylist pipeline complete.")
        print(f"  Total videos   : {summary['total']}")
        print(f"  Succeeded      : {len(summary['succeeded'])}")
        print(f"  Skipped        : {len(summary['skipped'])}")
        print(f"  Failed         : {len(summary['failed'])}")

        if summary["failed"]:
            print(f"\nFailed videos:")
            for url in summary["failed"]:
                print(f"  - {url}")

        print()

    except ValueError as e:
        print(f"\nError: {e}\n")
        sys.exit(1)


def cmd_ingest(args: argparse.Namespace) -> None:
    """Executes the ingestion stage for a given YouTube URL.

    Fetches the transcript for the provided URL and saves it to the
    configured transcripts directory. Prints the path of the saved
    transcript on success.

    Args:
        args: Parsed argument namespace containing:
            - url: The YouTube video URL to ingest.
    """
    logger.info("Running ingestion stage...")
    transcript_path = ingest(args.url)
    print(f"\nTranscript saved: {transcript_path}\n")


def cmd_extract(args: argparse.Namespace) -> None:
    """Executes extraction and storage stages for an ingested transcript.

    Loads a previously ingested transcript by its stem name, chunks it,
    extracts insights using the local LLM, and stores them in ChromaDB.

    Args:
        args: Parsed argument namespace containing:
            - source: The stem name of the transcript file to process.
    """
    from config import TRANSCRIPTS_DIR

    transcript_path = TRANSCRIPTS_DIR / f"{args.source}.txt"

    if not transcript_path.exists():
        print(f"\nError: Transcript not found: {transcript_path}")
        print("Run 'python main.py ingest <url>' first.\n")
        sys.exit(1)

    logger.info("Running chunking stage...")
    chunks = chunk_transcript(transcript_path)

    logger.info("Running extraction stage...")
    insights = extract_insights(chunks, transcript_path.stem)

    logger.info("Running storage stage...")
    total = store_insights(insights, transcript_path.stem)

    print(f"\nExtraction complete.")
    print(f"  Insights extracted : {len(insights)}")
    print(f"  Total in collection: {total}\n")


def cmd_query(args: argparse.Namespace) -> None:
    """Queries the ChromaDB knowledge base with a natural language query.

    Performs a semantic search against stored insights and prints the
    top matching results with their similarity distance scores.

    Args:
        args: Parsed argument namespace containing:
            - query: The natural language query string to search for.
            - source: Optional source name to filter results by video.
    """
    source = getattr(args, "source", None)
    results = query_insights(args.query, source_name=source)

    if not results:
        print("\nNo results found. Try a different query or ingest more content.\n")
        return

    print(f"\nTop {len(results)} results for: '{args.query}'\n")
    print("-" * 60)
    for i, r in enumerate(results, start=1):
        print(f"{i}. [{r['distance']}] {r['insight']}")
        print(f"   Source: {r['source']}\n")


def cmd_quiz(args: argparse.Namespace) -> None:
    """Starts an interactive quiz session for a given source video.

    Retrieves stored insights for the specified source and runs a
    terminal based multiple choice quiz session, printing a score
    summary on completion.

    Args:
        args: Parsed argument namespace containing:
            - source: The stem name of the source transcript to quiz on.
    """
    try:
        summary = run_quiz_session(args.source)
        print(f"Final score: {summary['correct']}/{summary['total']} "
              f"({summary['score_percent']}%)")
    except ValueError as e:
        print(f"\nError: {e}\n")
        sys.exit(1)


def cmd_run(args: argparse.Namespace) -> None:
    """Runs the full theGist pipeline end to end for a YouTube URL.

    Executes ingestion, chunking, extraction, and storage in sequence
    for the given URL, then immediately starts an interactive quiz
    session on the extracted insights.

    Args:
        args: Parsed argument namespace containing:
            - url: The YouTube video URL to process end to end.
    """
    logger.info("Running full pipeline...")

    # Ingestion
    transcript_path = ingest(args.url)
    print(f"\nTranscript saved: {transcript_path.name}")

    # Chunking
    chunks = chunk_transcript(transcript_path)
    print(f"Chunks created  : {len(chunks)}")

    # Extraction
    insights = extract_insights(chunks, transcript_path.stem)
    print(f"Insights found  : {len(insights)}")

    # Storage
    total = store_insights(insights, transcript_path.stem)
    print(f"Total in store  : {total}")

    # Quiz
    print("\nStarting quiz session...\n")
    try:
        summary = run_quiz_session(transcript_path.stem)
        print(f"Final score: {summary['correct']}/{summary['total']} "
              f"({summary['score_percent']}%)")
    except ValueError as e:
        print(f"\nError starting quiz: {e}\n")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the argument parser for the theGist CLI.

    Defines the top level parser and all subcommands with their
    respective arguments and help strings.

    Returns:
        A fully configured ArgumentParser instance ready to parse
        command line arguments.
    """
    parser = argparse.ArgumentParser(
        prog="theGist",
        description=(
            "theGist — extract expert insights from video transcripts "
            "and reinforce your knowledge with quizzes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py ingest https://www.youtube.com/watch?v=example
  python main.py extract Britons_vs_Teutons_1v1_Arabia_vs_Gali_AoE2
  python main.py query "how do I counter cavalry"
  python main.py query "resource management tips" --source Britons_vs_Teutons_1v1
  python main.py quiz Britons_vs_Teutons_1v1_Arabia_vs_Gali_AoE2
  python main.py run https://www.youtube.com/watch?v=example
        """,
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        metavar="<command>",
    )
    subparsers.required = True

    # topic
    topic_parser = subparsers.add_parser(
        "topic",
        help="Create and manage study topics.",
    )
    topic_parser.add_argument(
        "action",
        choices=["create", "list", "refresh", "delete"],
        help="Action to perform on topics.",
    )
    topic_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Topic name (required for create, refresh, delete).",
    )
    topic_parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Search query for topic creation.",
    )
    topic_parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional source filter for topic creation.",
    )
    topic_parser.set_defaults(func=cmd_topic)

    # playlist
    playlist_parser = subparsers.add_parser(
        "playlist",
        help="Ingest all videos from a YouTube playlist.",
    )
    playlist_parser.add_argument(
        "url",
        type=str,
        help="The YouTube playlist URL to ingest.",
    )
    playlist_parser.set_defaults(func=cmd_playlist)

    # ingest
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Fetch and save a transcript from a YouTube URL.",
    )
    ingest_parser.add_argument(
        "url",
        type=str,
        help="The YouTube video URL to ingest.",
    )
    ingest_parser.set_defaults(func=cmd_ingest)

    # extract
    extract_parser = subparsers.add_parser(
        "extract",
        help="Chunk, extract insights, and store for an ingested transcript.",
    )
    extract_parser.add_argument(
        "source",
        type=str,
        help="Stem name of the transcript file (without .txt extension).",
    )
    extract_parser.set_defaults(func=cmd_extract)

    # query
    query_parser = subparsers.add_parser(
        "query",
        help="Semantically search stored insights.",
    )
    query_parser.add_argument(
        "query",
        type=str,
        help="Natural language query string.",
    )
    query_parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional: filter results to a specific source video.",
    )
    query_parser.set_defaults(func=cmd_query)

    # quiz
    quiz_parser = subparsers.add_parser(
        "quiz",
        help="Start an interactive quiz session for a source video.",
    )
    quiz_parser.add_argument(
        "source",
        type=str,
        help="Stem name of the source transcript to quiz on.",
    )
    quiz_parser.set_defaults(func=cmd_quiz)

    # run
    run_parser = subparsers.add_parser(
        "run",
        help="Run the full pipeline end to end and start a quiz.",
    )
    run_parser.add_argument(
        "url",
        type=str,
        help="The YouTube video URL to process end to end.",
    )
    run_parser.set_defaults(func=cmd_run)

    return parser


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for the theGist CLI.

    Parses command line arguments and dispatches to the appropriate
    command handler function based on the subcommand provided.
    """
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()