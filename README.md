# 💡 theGist

**theGist** is a local, privacy-first knowledge extraction and learning platform. It ingests video transcripts from any YouTube video, uses a locally running large language model to extract key insights and expert knowledge, stores them in a semantic vector database, and helps users actively reinforce that knowledge through an AI-generated multiple choice quiz.

Built entirely on a free, local tech stack — no cloud APIs, no usage fees, no data leaving your machine.

---

## Features

- **Transcript Ingestion** — fetches auto-generated captions via yt-dlp, with automatic fallback to local Whisper transcription when captions are unavailable
- **Insight Extraction** — chunks transcripts intelligently and uses a locally running LLM via Ollama to identify and extract key insights and expert knowledge
- **Semantic Search** — stores insights as vector embeddings in ChromaDB, enabling natural language queries that retrieve results by meaning rather than keyword matching
- **Knowledge Quiz** — generates multiple choice questions from stored insights using the local LLM, with immediate feedback and a scored session summary
- **Dual Interface** — accessible via a clean Streamlit web UI or a fully featured command line interface

---

## Tech Stack

| Component | Tool |
|---|---|
| Transcript ingestion | yt-dlp |
| Audio transcription | OpenAI Whisper (local) |
| Insight extraction | Ollama + Llama 3 (local) |
| Vector embeddings | Sentence Transformers |
| Vector database | ChromaDB (local) |
| Web interface | Streamlit |
| Language | Python 3.11 |

---

## Project Structure

```
theGist/
│
├── data/                        # Runtime data (git ignored)
│   ├── transcripts/             # Raw transcript text files
│   ├── chunks/                  # Chunked transcript JSON files
│   ├── insights/                # Extracted insights JSON files
│   └── chroma/                  # ChromaDB vector store
│
├── src/                         # Core pipeline modules
│   ├── __init__.py
│   ├── ingestion.py             # Transcript fetching and transcription
│   ├── chunking.py              # Transcript splitting with overlap
│   ├── extraction.py            # LLM-based insight extraction
│   ├── storage.py               # Vector storage and semantic search
│   └── learning.py              # Quiz generation and evaluation
│
├── tests/                       # Jupyter notebook tests per module
│   ├── test_ingestion.ipynb
│   ├── test_chunking.ipynb
│   ├── test_extraction.ipynb
│   ├── test_storage.ipynb
│   └── test_learning.ipynb
│
├── app.py                       # Streamlit web UI entry point
├── main.py                      # CLI entry point
├── config.py                    # Centralised pipeline configuration
├── requirements.txt             # Project dependencies
└── README.md
```

---

## Prerequisites

Before installing theGist, ensure the following are available on your system:

1. **Python 3.11** via [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. **Ollama** — download and install from [ollama.com](https://ollama.com), then pull the Llama 3 model:
   ```bash
   ollama pull llama3
   ```
3. **ffmpeg** — required for Whisper audio transcription fallback:
   ```bash
   conda install -c conda-forge ffmpeg
   ```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/PG-23/theGist.git
cd theGist

# 2. Create and activate a conda environment
conda create -n thegist python=3.11
conda activate thegist

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ensure Ollama is running
ollama serve
```

---

## Usage

### Streamlit Web UI

Launch the web interface in your browser:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` and use the three page interface:

- **Ingest** — paste a YouTube URL to run the full pipeline
- **Explore** — search your knowledge base with natural language queries
- **Quiz** — test your knowledge with an AI-generated multiple choice quiz

### Command Line Interface

Run individual pipeline stages or the full pipeline from the terminal:

```bash
# Ingest a transcript from a YouTube URL
python main.py ingest https://www.youtube.com/watch?v=example

# Extract insights and store them for an ingested transcript
python main.py extract <transcript_stem_name>

# Query the knowledge base semantically
python main.py query "how do I counter cavalry units"

# Filter a query to a specific source video
python main.py query "resource management tips" --source <transcript_stem_name>

# Start an interactive quiz session
python main.py quiz <transcript_stem_name>

# Run the full pipeline end to end and start a quiz
python main.py run https://www.youtube.com/watch?v=example
```

---

## Configuration

All pipeline settings are centralised in `config.py`. Key options include:

| Setting | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base` | Whisper model size — larger models are more accurate but slower |
| `OLLAMA_MODEL` | `llama3` | Local LLM used for extraction and quiz generation |
| `CHUNK_SIZE` | `500` | Words per transcript chunk |
| `CHUNK_OVERLAP` | `50` | Overlapping words between chunks to preserve context |
| `MAX_INSIGHTS_PER_CHUNK` | `5` | Maximum insights extracted per chunk |
| `CHROMA_N_RESULTS` | `5` | Number of semantic search results returned |
| `QUIZ_QUESTION_COUNT` | `10` | Default number of quiz questions per session |

---

## How It Works

```
YouTube URL
    │
    ▼
Transcript Ingestion (yt-dlp / Whisper)
    │
    ▼
Chunking (fixed-size with overlap)
    │
    ▼
Insight Extraction (Ollama + Llama 3)
    │
    ▼
Vector Storage (ChromaDB + Sentence Transformers)
    │
    ├──▶ Semantic Search (Explore page)
    │
    └──▶ Quiz Generation (Ollama + Llama 3)
```

---

## Privacy

theGist runs entirely on your local machine. No video content, transcripts, or extracted insights are sent to any external server or API. All LLM inference is performed locally via Ollama.

---

## Future Improvements

- Support for non-YouTube transcript sources including local video files and podcast feeds
- Multi-video knowledge base aggregation across a playlist or channel
- Spaced repetition scheduling for quiz questions based on past performance
- Improved insight quality through prompt refinement and multi-pass extraction
- Streamlit UI deployment guide for sharing a hosted instance

---

## License

This project is licensed under the MIT License.

---

## Author

Patrick Guinn — [LinkedIn](https://www.linkedin.com/in/patrick-guinn/) | [GitHub](https://github.com/PG-23)