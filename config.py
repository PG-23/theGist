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
# Domain Adaptation Settings
# ---------------------------------------------------------------------------

# The active domain for transcript correction and vocabulary biasing.
# Set to a key from DOMAIN_VOCABULARY to enable domain specific adaptation,
# or None to disable domain adaptation entirely.
# Example: ACTIVE_DOMAIN = "aoe2"
ACTIVE_DOMAIN: str | None = "aoe2"

# Controls whether videos without auto-generated captions are skipped.
# When True, videos without captions are skipped rather than falling back
# to local Whisper transcription. Recommended for best insight quality.
# Set to False to re-enable Whisper transcription for caption-less videos.
CAPTIONS_ONLY: bool = True

# Pre-defined domain vocabulary libraries.
# Each domain maps to a dictionary containing:
#   - "name": Human readable display name for the domain
#   - "vocabulary": List of domain specific terms for Whisper prompt biasing
#   - "correction_prompt": Instructions for the LLM correction pass
DOMAIN_VOCABULARY: dict = {
    "aoe2": {
        "name": "Age of Empires 2",
        "vocabulary": [
            # Civilizations
            "Celts", "Britons", "Franks", "Teutons", "Cumans", "Lithuanians",
            "Koreans", "Mongols", "Aztecs", "Vikings", "Saracens", "Persians",
            "Chinese", "Japanese", "Byzantines", "Goths", "Huns", "Mayans",
            "Spanish", "Turks", "Berbers", "Malians", "Portuguese", "Italians",
            "Slavs", "Bulgarians", "Tatars", "Khmer", "Malay", "Burmese",
            "Vietnamese", "Incas", "Indians", "Romans",
            # Units
            "men-at-arms", "halberdier", "spearman", "pikeman", "militia",
            "arbalest", "crossbowman", "skirmisher", "cavalry archer",
            "heavy cavalry archer", "knight", "cavalier", "paladin",
            "hussar", "light cavalry", "scout cavalry", "camel rider",
            "battle elephant", "war elephant", "trebuchet", "mangonel",
            "onager", "siege onager", "scorpion", "bombardment cannon",
            "monk", "missionary", "trade cart", "villager", "galley",
            "war galley", "demolition ship", "fire ship", "longboat",
            "cannon galleon", "petard",
            # Strategies and terminology
            "fast castle", "feudal age", "castle age", "imperial age",
            "dark age", "build order", "boom", "all-in", "flush",
            "scout rush", "men-at-arms rush", "archer rush", "tower rush",
            "drush", "forward base", "forward castle", "relics",
            "town center", "market", "blacksmith", "stable", "barracks",
            "archery range", "siege workshop", "university", "monastery",
            "castle", "wonder", "loom", "double-bit axe", "horse collar",
            "wheelbarrow", "hand cart", "bloodlines", "husbandry",
            "thumb ring", "ballistics", "chemistry", "murder holes",
            "heated shot", "bracer", "plate barding armor",
            # Maps and game modes
            "Arabia", "Black Forest", "Islands", "Arena", "Hideout",
            "Nomad", "Gold Rush", "Team Islands", "Migration",
            "random map", "death match", "regicide", "treaty",
            # Common commentary phrases
            "micro", "macro", "eco", "economy", "pop cap", "population",
            "food", "wood", "gold", "stone", "resources", "villagers",
            "idle time", "scouting", "walling", "palisade", "stone wall",
            "garrison", "patrol", "stance", "aggressive", "defensive",
        ],
        "correction_prompt": (
            "The following is an auto-generated transcript from an "
            "Age of Empires 2 video commentary. Correct any misrecognized "
            "game-specific terms including civilization names such as Celts, "
            "Cumans, Britons, Teutons and Lithuanians, unit names such as "
            "halberdier, arbalest, mangonel and trebuchet, and strategy "
            "terms such as fast castle, feudal age, build order and boom. "
            "Do not change the meaning, add information, or alter anything "
            "that is already correct. Return only the corrected transcript "
            "with no explanation or preamble."
        ),
    },
    "computer_science": {
        "name": "Computer Science",
        "vocabulary": [
            # Languages and frameworks
            "Python", "JavaScript", "TypeScript", "Rust", "Go", "Kotlin",
            "React", "Vue", "Angular", "Node.js", "Django", "FastAPI",
            "PyTorch", "TensorFlow", "Kubernetes", "Docker", "GraphQL",
            # Concepts
            "algorithm", "data structure", "binary tree", "hash map",
            "recursion", "dynamic programming", "big O notation",
            "asynchronous", "concurrency", "parallelism", "mutex",
            "deadlock", "race condition", "garbage collection",
            "heap", "stack", "queue", "linked list",
            # Infrastructure
            "API", "REST", "microservices", "monolith", "serverless",
            "CI/CD", "DevOps", "pipeline", "containerization",
            "load balancer", "cache", "CDN", "latency", "throughput",
            # ML and AI
            "neural network", "transformer", "embedding", "fine-tuning",
            "inference", "tokenizer", "gradient descent", "backpropagation",
            "overfitting", "regularization", "hyperparameter",
        ],
        "correction_prompt": (
            "The following is an auto-generated transcript from a computer "
            "science video or lecture. Correct any misrecognized technical "
            "terms including programming languages, framework names, "
            "algorithm names, and computer science concepts. "
            "Do not change the meaning, add information, or alter anything "
            "that is already correct. Return only the corrected transcript "
            "with no explanation or preamble."
        ),
    },
}

# ---------------------------------------------------------------------------
# Logging Settings
# ---------------------------------------------------------------------------

# Logging level: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_LEVEL = "INFO"

# Log format string used across all modules
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# Log date format
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"