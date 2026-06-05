"""Streamlit web interface for theGist application.

This module provides a multi-page web UI built with Streamlit, allowing
users to ingest video transcripts, explore extracted insights via semantic
search, and test their knowledge through an interactive quiz interface.

Usage:
    streamlit run app.py
"""

import logging
import sys
from pathlib import Path

import streamlit as st

from config import (
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
    QUIZ_QUESTION_COUNT,
    TRANSCRIPTS_DIR,
)
from src.chunking import chunk_transcript
from src.extraction import extract_insights
from src.ingestion import ingest
from src.learning import evaluate_answer, generate_quiz
from src.storage import get_all_insights, query_insights, store_insights

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
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="theGist",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Initialises all required Streamlit session state variables.

    Sets default values for session state keys used across all pages
    if they have not already been initialised. Called once at app
    startup to ensure a consistent initial state.
    """
    defaults = {
        "current_page": "Ingest",
        "quiz_questions": [],
        "quiz_index": 0,
        "quiz_score": 0,
        "quiz_results": [],
        "quiz_active": False,
        "quiz_source": None,
        "last_ingested_source": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------

def _render_sidebar() -> str:
    """Renders the sidebar navigation and returns the selected page name.

    Displays the application title, description, and navigation buttons
    in the sidebar. Highlights the currently active page button.

    Returns:
        The name of the currently selected page as a string.
    """
    with st.sidebar:
        st.title("💡 theGist")
        st.caption("Extract expert insights. Learn smarter.")
        st.divider()

        pages = ["Ingest", "Explore", "Quiz"]
        icons = ["📥", "🔍", "🧠"]

        for page, icon in zip(pages, icons):
            is_active = st.session_state.current_page == page
            if st.button(
                f"{icon} {page}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.current_page = page
                st.rerun()

        st.divider()
        st.caption("theGist — v0.1.0")

    return st.session_state.current_page


# ---------------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------------

def _get_available_sources() -> list[str]:
    """Returns a list of transcript stem names available in the store.

    Scans the configured transcripts directory for saved transcript
    files and returns their stem names for use in source selection
    dropdowns across the UI.

    Returns:
        A sorted list of transcript stem name strings. Returns an
        empty list if no transcripts have been ingested yet.
    """
    return sorted([f.stem for f in TRANSCRIPTS_DIR.glob("*.txt")])


# ---------------------------------------------------------------------------
# Page: Ingest
# ---------------------------------------------------------------------------

def _render_ingest_page() -> None:
    """Renders the transcript ingestion page.

    Provides a form for submitting a YouTube URL through the full
    theGist pipeline — ingestion, chunking, extraction, and storage.
    Displays live progress updates and a summary of extracted insights
    on completion.
    """
    st.header("📥 Ingest a Video")
    st.write(
        "Paste a YouTube URL below to extract expert insights from the "
        "video transcript and add them to your knowledge base."
    )

    with st.form("ingest_form"):
        url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        submitted = st.form_submit_button(
            "Extract Insights",
            use_container_width=True,
            type="primary",
        )

    if submitted and url.strip():
        _run_pipeline(url.strip())
    elif submitted:
        st.warning("Please enter a valid YouTube URL.")


def _run_pipeline(url: str) -> None:
    """Executes the full theGist pipeline for a given URL with UI feedback.

    Runs ingestion, chunking, extraction, and storage sequentially,
    updating a Streamlit progress bar and status messages at each stage.
    Displays a summary of results and a preview of extracted insights
    on successful completion.

    Args:
        url: The YouTube video URL to process through the pipeline.
    """
    progress = st.progress(0, text="Starting pipeline...")

    try:
        # Stage 1 — Ingestion
        progress.progress(10, text="Fetching transcript...")
        transcript_path = ingest(url)
        source_name = transcript_path.stem

        # Stage 2 — Chunking
        progress.progress(35, text="Splitting transcript into chunks...")
        chunks = chunk_transcript(transcript_path)

        # Stage 3 — Extraction
        progress.progress(55, text="Extracting insights with local LLM...")
        insights = extract_insights(chunks, source_name)

        # Stage 4 — Storage
        progress.progress(85, text="Storing insights in knowledge base...")
        total = store_insights(insights, source_name)

        progress.progress(100, text="Complete!")
        st.session_state.last_ingested_source = source_name

        # Success summary
        st.success(f"Pipeline complete for: **{source_name.replace('_', ' ')}**")

        col1, col2, col3 = st.columns(3)
        col1.metric("Chunks Processed", len(chunks))
        col2.metric("Insights Extracted", len(insights))
        col3.metric("Total in Knowledge Base", total)

        # Insight preview
        with st.expander("Preview extracted insights", expanded=True):
            for i, insight in enumerate(insights[:10], start=1):
                st.write(f"{i}. {insight}")
            if len(insights) > 10:
                st.caption(f"...and {len(insights) - 10} more insights stored.")

    except ValueError as e:
        progress.empty()
        st.error(f"Invalid URL or video unavailable: {e}")
    except Exception as e:
        progress.empty()
        st.error(f"Pipeline failed: {e}")
        logger.error(f"Pipeline error for {url}: {e}")


# ---------------------------------------------------------------------------
# Page: Explore
# ---------------------------------------------------------------------------

def _render_explore_page() -> None:
    """Renders the knowledge base exploration and semantic search page.

    Allows users to query stored insights using natural language and
    optionally filter results by source video. Also displays all
    available insights for a selected source in an expandable panel.
    """
    st.header("🔍 Explore Insights")
    st.write(
        "Search your knowledge base using natural language. "
        "Results are ranked by semantic similarity."
    )

    sources = _get_available_sources()

    if not sources:
        st.info(
            "No insights found. Go to the **Ingest** page to add a video first."
        )
        return

    # Search form
    with st.form("search_form"):
        query = st.text_input(
            "Search query",
            placeholder="e.g. how do I counter cavalry units",
        )
        source_filter = st.selectbox(
            "Filter by source (optional)",
            options=["All sources"] + sources,
        )
        submitted = st.form_submit_button(
            "Search",
            use_container_width=True,
            type="primary",
        )

    if submitted and query.strip():
        source_name = None if source_filter == "All sources" else source_filter
        results = query_insights(query.strip(), source_name=source_name)

        if results:
            st.subheader(f"Top {len(results)} results")
            for i, r in enumerate(results, start=1):
                with st.container(border=True):
                    st.write(f"**{i}.** {r['insight']}")
                    col1, col2 = st.columns([3, 1])
                    col1.caption(f"Source: {r['source'].replace('_', ' ')}")
                    col2.caption(f"Similarity: {round((1 - r['distance']) * 100, 1)}%")
        else:
            st.info("No results found. Try a different search query.")

    elif submitted:
        st.warning("Please enter a search query.")

    # Browse all insights for a source
    st.divider()
    st.subheader("Browse All Insights")
    browse_source = st.selectbox(
        "Select a source to browse",
        options=sources,
        key="browse_source",
    )

    if browse_source:
        all_insights = get_all_insights(browse_source)
        st.caption(f"{len(all_insights)} insights from: {browse_source.replace('_', ' ')}")
        with st.expander("View all insights", expanded=False):
            for i, insight in enumerate(all_insights, start=1):
                st.write(f"{i}. {insight}")


# ---------------------------------------------------------------------------
# Page: Quiz
# ---------------------------------------------------------------------------

def _render_quiz_page() -> None:
    """Renders the interactive knowledge quiz page.

    Provides source selection and quiz configuration controls, manages
    quiz session state across Streamlit reruns, presents multiple choice
    questions one at a time, and displays a score summary on completion.
    """
    st.header("🧠 Knowledge Quiz")
    st.write(
        "Test your understanding of extracted insights with a "
        "multiple choice quiz generated by the local LLM."
    )

    sources = _get_available_sources()

    if not sources:
        st.info(
            "No insights found. Go to the **Ingest** page to add a video first."
        )
        return

    # Quiz setup — only shown when no active quiz session
    if not st.session_state.quiz_active:
        with st.form("quiz_setup_form"):
            source = st.selectbox("Select a source to quiz on", options=sources)
            question_count = st.slider(
                "Number of questions",
                min_value=3,
                max_value=min(QUIZ_QUESTION_COUNT, 15),
                value=5,
            )
            start = st.form_submit_button(
                "Start Quiz",
                use_container_width=True,
                type="primary",
            )

        if start:
            with st.spinner("Generating quiz questions..."):
                try:
                    questions = generate_quiz(source, question_count)
                    if questions:
                        st.session_state.quiz_questions = questions
                        st.session_state.quiz_index = 0
                        st.session_state.quiz_score = 0
                        st.session_state.quiz_results = []
                        st.session_state.quiz_active = True
                        st.session_state.quiz_source = source
                        st.rerun()
                    else:
                        st.error("No questions could be generated. Try ingesting more content.")
                except ValueError as e:
                    st.error(f"Quiz error: {e}")

    # Active quiz session
    elif st.session_state.quiz_active:
        questions = st.session_state.quiz_questions
        idx = st.session_state.quiz_index
        total = len(questions)

        # Quiz complete
        if idx >= total:
            _render_quiz_summary()
            return

        # Progress indicator
        st.progress(idx / total, text=f"Question {idx + 1} of {total}")

        question = questions[idx]

        with st.container(border=True):
            st.subheader(f"Q{idx + 1}. {question['question']}")
            st.caption(
                f"Source: {st.session_state.quiz_source.replace('_', ' ')}"
            )

            selected = st.radio(
                "Choose your answer:",
                options=question["choices"],
                key=f"question_{idx}",
                index=None,
            )

            col1, col2 = st.columns([1, 4])
            submit_answer = col1.button(
                "Submit",
                type="primary",
                use_container_width=True,
            )

            if submit_answer and selected:
                result = evaluate_answer(question, selected)
                st.session_state.quiz_results.append(result)

                if result["correct"]:
                    st.session_state.quiz_score += 1
                    st.success("Correct!")
                else:
                    st.error(
                        f"Incorrect. The correct answer was: "
                        f"**{result['correct_answer']}**"
                    )
                    st.info(f"Insight: *{result['source_insight']}*")

                st.session_state.quiz_index += 1

                next_label = "Next Question" if idx + 1 < total else "See Results"
                if st.button(next_label, use_container_width=True):
                    st.rerun()

            elif submit_answer and not selected:
                st.warning("Please select an answer before submitting.")


def _render_quiz_summary() -> None:
    """Renders the quiz completion summary screen.

    Displays the final score, a percentage, a performance message
    based on the score band, and an expandable breakdown of all
    questions showing correct and incorrect answers.
    """
    score = st.session_state.quiz_score
    total = len(st.session_state.quiz_questions)
    pct = round((score / total) * 100, 1) if total > 0 else 0

    st.balloons()
    st.subheader("Quiz Complete!")

    col1, col2 = st.columns(2)
    col1.metric("Score", f"{score}/{total}")
    col2.metric("Percentage", f"{pct}%")

    if pct >= 80:
        st.success("Excellent work! You have a strong grasp of this content.")
    elif pct >= 60:
        st.info("Good effort! Review the incorrect answers to reinforce your knowledge.")
    else:
        st.warning("Keep practicing! Try ingesting more videos on this topic to build depth.")

    # Results breakdown
    with st.expander("Review your answers", expanded=False):
        for i, result in enumerate(st.session_state.quiz_results, start=1):
            icon = "✅" if result["correct"] else "❌"
            st.write(f"{icon} **Q{i}:** {st.session_state.quiz_questions[i - 1]['question']}")
            if not result["correct"]:
                st.caption(f"Your answer: {result['user_answer']}")
                st.caption(f"Correct answer: {result['correct_answer']}")
                st.caption(f"Insight: {result['source_insight']}")
            st.divider()

    if st.button("Start New Quiz", use_container_width=True, type="primary"):
        st.session_state.quiz_active = False
        st.session_state.quiz_questions = []
        st.session_state.quiz_index = 0
        st.session_state.quiz_score = 0
        st.session_state.quiz_results = []
        st.rerun()


# ---------------------------------------------------------------------------
# App Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for the theGist Streamlit application.

    Renders the sidebar navigation and dispatches to the appropriate
    page rendering function based on the currently selected page.
    """
    page = _render_sidebar()

    if page == "Ingest":
        _render_ingest_page()
    elif page == "Explore":
        _render_explore_page()
    elif page == "Quiz":
        _render_quiz_page()


if __name__ == "__main__":
    main()