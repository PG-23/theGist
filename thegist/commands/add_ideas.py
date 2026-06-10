"""Add ideas command for theGist.

This module implements the add-ideas command which guides the user
through a continuous session of adding key ideas to records that
have none yet. Records are processed one at a time in order, with
the user prompted to continue after each one.

Usage:
    thegist add-ideas --subject <subject>
"""

import re
from pathlib import Path

from thegist.config import DATA_DIR
from thegist.src.database import get_records_without_ideas, insert_ideas

EXPORTS_DIR = DATA_DIR / "exports"


def register(subparsers) -> None:
    """Registers the add-ideas command with the top level argument parser.

    Args:
        subparsers: The subparsers group from the top level parser.
    """
    parser = subparsers.add_parser(
        "add-ideas",
        help="Add key ideas to records interactively.",
        description=(
            "Guides you through a session of adding key ideas to records "
            "that have none yet. Records are processed one at a time. "
            "Press Enter to confirm prompts or type n to skip or exit."
        ),
    )
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="Subject name to find records needing ideas.",
    )
    parser.set_defaults(func=run)


def _sanitize(title: str) -> str:
    """Converts a video title into a safe filename stem.

    Args:
        title: The raw video title to sanitize.

    Returns:
        A sanitized string safe for use as a filename, maximum
        60 characters with spaces replaced by underscores.
    """
    sanitized = re.sub(r"[^\w\s-]", "", title)
    sanitized = re.sub(r"\s+", "_", sanitized).strip("_")
    return sanitized[:60]


def _ask(prompt: str) -> bool:
    """Prompts the user with a yes/no question.

    Enter and y/yes are treated as yes. Only n/no is treated as no.

    Args:
        prompt: The question string to display to the user.

    Returns:
        True if the user confirmed, False if they declined.
    """
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("", "y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  Please enter y or n.")


def _export_files(record: dict) -> tuple[Path, Path]:
    """Exports the transcript and an ideas template to data/exports/.

    Creates the exports directory if it does not exist. Writes the
    full transcript text and an empty ideas template file ready for
    the user to fill in.

    Args:
        record: The record dictionary to export files for.

    Returns:
        A tuple of (transcript_path, ideas_path) pointing to the
        exported files.
    """
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = _sanitize(record["title"])

    transcript_path = EXPORTS_DIR / f"{stem}_transcript.txt"
    ideas_path = EXPORTS_DIR / f"{stem}_ideas.txt"

    transcript_path.write_text(record["transcript"], encoding="utf-8")

    template = (
        "IDEAS\n"
        "# Add your key ideas below, one per line, prefixed with *\n"
        "# Delete these comment lines before saving\n"
        "\n"
        "* \n"
    )
    ideas_path.write_text(template, encoding="utf-8")

    return transcript_path, ideas_path


def _parse_ideas(ideas_path: Path) -> list[str]:
    """Reads and parses an ideas file into a clean list of strings.

    Skips blank lines, comment lines starting with #, the IDEAS
    header, and bullet prefixes.

    Args:
        ideas_path: Path to the ideas text file to parse.

    Returns:
        A list of clean idea strings with all formatting removed.
    """
    ideas = []
    for line in ideas_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper() == "IDEAS":
            continue
        if line.startswith("#"):
            continue
        if line.startswith("*") or line.startswith("-"):
            line = line[1:].strip()
        if line:
            ideas.append(line)
    return ideas


def _process_record(record: dict) -> bool:
    """Runs the full ideas collection flow for a single record."""

    print(f"\n{'-' * 60}")
    print(f"Record  : {record['title']}")
    print(f"Channel : {record['channel']}")

    transcript_path, ideas_path = _export_files(record)

    print(f"\nFiles saved to: {EXPORTS_DIR}\n")
    print(f"  Transcript : {transcript_path.name}")
    print(f"  Ideas      : {ideas_path.name}\n")
    print("Steps:")
    print("  1. Open the transcript with your LLM and generate ideas")
    print("  2. Paste ideas into the ideas file and save it")
    print("  3. Press Enter to continue\n")

    input("Press Enter when your ideas file is ready...")

    while True:
        ideas = _parse_ideas(ideas_path)

        if not ideas:
            print("\nNo ideas found in the ideas file.")
            print("Make sure your ideas are prefixed with * and saved.")
            if not _ask("Try again? [Y/n]: "):
                _cleanup_exports(transcript_path, ideas_path)
                print("\nSkipping this record.\n")
                return False
            input("Press Enter when ready...")
            continue

        break

    print(f"\nReady to save {len(ideas)} idea(s) for:")
    print(f"  {record['title']}\n")
    print("Preview:")
    for idea in ideas[:3]:
        print(f"  - {idea}")
    if len(ideas) > 3:
        print(f"  ... and {len(ideas) - 3} more")
    print()

    if _ask("Save these ideas? [Y/n]: "):
        insert_ideas(record["id"], ideas)
        _cleanup_exports(transcript_path, ideas_path)
        print(f"\n✓ {len(ideas)} idea(s) saved.\n")
        return True
    else:
        _cleanup_exports(transcript_path, ideas_path)
        print("\nIdeas not saved.\n")
        return False


def _cleanup_exports(transcript_path: Path, ideas_path: Path) -> None:
    """Removes exported transcript and ideas files after processing.

    Called after a record has been processed regardless of whether
    ideas were saved or skipped, keeping the exports folder clean
    for the next record.

    Args:
        transcript_path: Path to the transcript file to remove.
        ideas_path: Path to the ideas file to remove.
    """
    for path in (transcript_path, ideas_path):
        try:
            if path.exists():
                path.unlink()
        except Exception as e:
            print(f"  Warning: Could not remove {path.name}: {e}")


def run(args) -> None:
    """Executes the add-ideas command.

    Finds all records without ideas for the given subject and guides
    the user through a continuous session processing one record at a
    time. After each record the user is shown how many remain and
    asked whether to continue.

    Args:
        args: Parsed argument namespace containing subject.
    """
    subject = args.subject
    records = get_records_without_ideas(subject)

    if not records:
        print(f"\nNo records without ideas found for: {subject}")
        print("All records for this subject already have ideas.\n")
        return

    total = len(records)
    print(f"\nFound {total} record(s) without ideas for: {subject}")

    # Preview up to three pending records
    for record in records[:3]:
        print(f"  - {record['title']}")
    if total > 3:
        print(f"  ... and {total - 3} more")

    print()

    processed = 0
    for record in records:
        remaining = total - processed
        if remaining == 0:
            break

        if not _ask(f"Add ideas to the next record? [Y/n]: "):
            print(f"\nSession ended. {remaining} record(s) still without ideas.")
            print("Run add-ideas again to continue.\n")
            return

        _process_record(record)
        processed += 1

        remaining_after = total - processed
        if remaining_after > 0:
            print(f"{remaining_after} record(s) still without ideas.")
        else:
            print("All records now have ideas.\n")
            return

    print(f"\nSession complete. All records for '{subject}' have ideas.\n")