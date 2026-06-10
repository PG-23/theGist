"""Central configuration for theGist.

This module defines all paths and settings used across the project.
Modifying values here propagates changes throughout the application
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

# SQLite database file — all records and ideas are stored here
DATABASE_PATH = DATA_DIR / "thegist.db"

# ---------------------------------------------------------------------------
# Logging Settings
# ---------------------------------------------------------------------------

# Logging level: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_LEVEL: str = "INFO"

# Log format string used across all modules
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# Log date format
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"