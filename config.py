"""Central configuration for theGist application.

This module defines all configurable constants used across the project.
Modifying values here propagates changes throughout the entire application
without requiring edits to individual source files.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------

# Root directory of the project, resolved relative to this file
ROOT_DIR = Path(__file__).parent.resolve()

# Top level data directory
DATA_DIR = ROOT_DIR / "data"

# Transcripts directory — raw transcript text files fetched from YouTube
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

# Records directory — one JSON file per saved video record
RECORDS_DIR = DATA_DIR / "records"

# Quizzes directory — one JSON file per saved quiz
QUIZZES_DIR = DATA_DIR / "quizzes"

# Tags index file — maps tags to the idea IDs they are associated with
TAGS_FILE = DATA_DIR / "tags.json"

# Ensure all required directories exist on import
for _dir in (TRANSCRIPTS_DIR, RECORDS_DIR, QUIZZES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Ingestion Settings
# ---------------------------------------------------------------------------

# Controls whether videos without auto-generated captions are skipped.
# When True, videos without captions raise a ValueError rather than
# falling back to local Whisper transcription.
CAPTIONS_ONLY: bool = True

# Language code for caption fetching
WHISPER_LANGUAGE: str = "en"

# ---------------------------------------------------------------------------
# Logging Settings
# ---------------------------------------------------------------------------

# Logging level: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_LEVEL: str = "INFO"

# Log format string used across all modules
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# Log date format
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"