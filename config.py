"""Central configuration for theGist pipeline.

This module defines all configurable constants used across the project.
Modifying values here propagates changes throughout the entire pipeline
without requiring edits to individual source files.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------

# Root directory of the project, resolved relative to this file
ROOT_DIR = Path(__file__).parent.resolve()

# Data directories for each stage of the pipeline
DATA_DIR = ROOT_DIR / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CHUNKS_DIR = DATA_DIR / "chunks"
INSIGHTS_DIR = DATA_DIR / "insights"

# Ensure all data directories exist on import
for _dir in (TRANSCRIPTS_DIR, CHUNKS_DIR, INSIGHTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Transcription Settings (Whisper)
# ---------------------------------------------------------------------------

# Whisper model size: "tiny", "base", "small", "medium", "large"
# Larger models are more accurate but slower and require more memory.
# "base" is recommended as a starting point for most hardware.
WHISPER_MODEL = "base"

# Language of the audio content. None defaults to Whisper auto-detection.
WHISPER_LANGUAGE = "en"

# ---------------------------------------------------------------------------
# Chunking Settings
# ---------------------------------------------------------------------------

# Number of words per chunk when splitting transcripts.
# Smaller chunks are more precise; larger chunks preserve more context.
CHUNK_SIZE = 500

# Number of words overlapping between consecutive chunks.
# Overlap prevents insights from being lost at chunk boundaries.
CHUNK_OVERLAP = 50

# ---------------------------------------------------------------------------
# Insight Extraction Settings (Ollama)
# ---------------------------------------------------------------------------

# Local Ollama model to use for insight extraction.
# Recommended options: "llama3", "mistral", "phi3"
OLLAMA_MODEL = "llama3"

# Base URL for the local Ollama API server.
OLLAMA_BASE_URL = "http://localhost:11434"

# Maximum number of insights to extract per chunk.
# Prevents over-extraction from dense transcript sections.
MAX_INSIGHTS_PER_CHUNK = 5

# Prompt template for insight extraction.
# {chunk} is replaced at runtime with the actual transcript text.
EXTRACTION_PROMPT = """You are an expert knowledge extractor. Given the following transcript excerpt, 
extract up to {max_insights} key insights, tips, or pieces of expert knowledge. 

Focus only on meaningful, actionable insights. Ignore filler conversation, 
greetings, and off-topic commentary.

Return each insight as a single clear sentence on its own line, prefixed with a dash (-).

Transcript:
{chunk}

Insights:"""

# ---------------------------------------------------------------------------
# Embedding Settings (Sentence Transformers)
# ---------------------------------------------------------------------------

# Local embedding model used to convert insights into vector representations.
# "all-MiniLM-L6-v2" is lightweight, fast, and effective for semantic search.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Vector Storage Settings (ChromaDB)
# ---------------------------------------------------------------------------

# Directory where ChromaDB persists its vector database on disk.
CHROMA_DB_DIR = DATA_DIR / "chroma"

# Name of the ChromaDB collection that stores extracted insights.
CHROMA_COLLECTION_NAME = "thegist_insights"

# Number of similar results to return for a semantic search query.
CHROMA_N_RESULTS = 5

# ---------------------------------------------------------------------------
# Learning Layer Settings
# ---------------------------------------------------------------------------

# Number of insights to include in a single trivia or quiz session.
QUIZ_QUESTION_COUNT = 10

# Number of multiple choice options presented per trivia question.
QUIZ_CHOICES_COUNT = 4

# ---------------------------------------------------------------------------
# Logging Settings
# ---------------------------------------------------------------------------

# Logging level: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_LEVEL = "INFO"

# Log format string used across all modules
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# Log date format
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"