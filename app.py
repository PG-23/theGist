"""Streamlit web interface for theGist application.

This module provides the main web UI for theGist knowledge management
system. It allows users to fetch and download video transcripts, save
records pairing transcripts with curated key ideas, organize ideas
using tags, browse topics, and store and take quizzes.

Usage:
    streamlit run app.py
"""

import logging
import random
from pathlib import Path

import streamlit as st

from config import (
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
    TRANSCRIPTS_DIR,
)
from src.database import (
    add_ideas,
    add_tag_to_idea,
    create_record,
    delete_idea,
    delete_quiz,
    delete_record,
    get_ideas_by_tag,
    get_quiz,
    get_record,
    list_all_tags,
    list_quizzes,
    list_records,
    remove_tag_from_idea,
    save_quiz,
    update_idea_text,
)
from src.ingestion import ingest, ingest_playlist

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
    if they have not already been initialised.
    """
    defaults = {
        "current_page": "Transcripts",
        "pending_transcript": None,
        "transcript_metadata": {},
        "quiz_active": False,
        "quiz_data": None,
        "quiz_index": 0,
        "quiz_score": 0,
        "quiz_results": [],
        "confirm_delete_record": None,
        "confirm_delete_idea": None,
        "confirm_delete_quiz": None,
        "editing_idea": None,
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

    Returns:
        The name of the currently selected page as a string.
    """
    with st.sidebar:
        st.title("💡 theGist")
        st.caption("Curate expert insights. Learn smarter.")
        st.divider()

        pages = ["Transcripts", "Library", "Topics", "Quiz"]
        icons = ["📄", "📚", "🏷️", "🧠"]

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
        st.caption("theGist — v0.2.0")

    return st.session_state.current_page


# ---------------------------------------------------------------------------
# Shared Utilities
# ---------------------------------------------------------------------------

def _parse_ideas_input(raw: str) -> list[str]:
    """Parses pasted key ideas text into a clean list of idea strings.

    Handles the standard LLM output format containing an optional IDEAS
    header and bullet points prefixed with asterisks or dashes. Strips
    all formatting characters and returns only the idea text.

    Args:
        raw: The raw pasted text containing key ideas, optionally with
            an IDEAS header and bullet point prefixes.

    Returns:
        A list of clean idea strings with all formatting removed.
        Returns an empty list if no valid ideas are found.

    Example:
        >>> raw = "IDEAS\\n\\n* Celts infantry move faster\\n* Fast castle is strong"
        >>> _parse_ideas_input(raw)
        ['Celts infantry move faster', 'Fast castle is strong']
    """
    ideas = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper() == "IDEAS":
            continue
        if line.startswith("*") or line.startswith("-"):
            line = line[1:].strip()
        if line:
            ideas.append(line)
    return ideas


def _get_record_selections(record_id: str) -> dict:
    """Returns the selection state dictionary for a given record.

    Initialises the selection dictionary in session state if it does
    not already exist.

    Args:
        record_id: The unique identifier of the record.

    Returns:
        A dictionary mapping idea_id strings to boolean selection state.
    """
    key = f"selections_{record_id}"
    if key not in st.session_state:
        st.session_state[key] = {}
    return st.session_state[key]


def _get_review_status(record: dict) -> tuple[str, str]:
    """Computes the review status of a record based on idea tagging.

    A record is considered Reviewed when every key idea has at least
    one tag assigned. If any idea has no tags the record is Unreviewed.
    Records with no ideas are considered Unreviewed by default.

    Args:
        record: The full record dictionary including key_ideas.

    Returns:
        A tuple of (status_label, status_icon) where status_label is
        either 'Reviewed' or 'Unreviewed' and status_icon is the
        corresponding emoji indicator.

    Example:
        >>> label, icon = _get_review_status(record)
        >>> print(icon, label)
        '✅ Reviewed'
    """
    ideas = record.get("key_ideas", [])
    if not ideas:
        return "Unreviewed", "🔶"
    all_tagged = all(len(idea.get("tags", [])) > 0 for idea in ideas)
    return ("Reviewed", "✅") if all_tagged else ("Unreviewed", "🔶")


# ---------------------------------------------------------------------------
# Page: Transcripts
# ---------------------------------------------------------------------------

def _render_transcripts_page() -> None:
    """Renders the transcript collection page.

    Allows users to fetch transcripts from a single YouTube video or
    an entire playlist. Fetched transcripts can be downloaded as .txt
    files and saved as records with pasted key ideas.
    """
    st.header("📄 Transcripts")
    st.write(
        "Fetch transcripts from YouTube videos with auto-generated captions. "
        "Download the transcript to generate key ideas externally, then "
        "save the transcript and your ideas as a record."
    )

    tab1, tab2 = st.tabs(["Single Video", "Playlist"])

    with tab1:
        with st.form("single_video_form"):
            url = st.text_input(
                "YouTube Video URL",
                placeholder="https://www.youtube.com/watch?v=...",
            )
            submitted = st.form_submit_button(
                "Fetch Transcript",
                use_container_width=True,
                type="primary",
            )

        if submitted and url.strip():
            with st.spinner("Fetching transcript..."):
                try:
                    transcript_path, metadata = ingest(url.strip())
                    st.session_state.pending_transcript = {
                        "path": str(transcript_path),
                        "title": metadata["title"],
                        "channel": metadata["uploader"],
                        "url": metadata["url"],
                    }
                    st.session_state.transcript_metadata[
                        transcript_path.stem
                    ] = {
                        "title": metadata["title"],
                        "channel": metadata["uploader"],
                        "url": metadata["url"],
                    }
                    st.success(f"Transcript fetched: **{metadata['title']}**")
                except ValueError as e:
                    st.error(f"Could not fetch transcript: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    logger.error(f"Ingestion error: {e}")
        elif submitted:
            st.warning("Please enter a YouTube URL.")

        if st.session_state.pending_transcript:
            _render_save_transcript_panel()

    with tab2:
        st.write(
            "Fetch transcripts for all caption-enabled videos in a playlist. "
            "Each transcript will be available to save individually in the "
            "Library page after fetching."
        )
        with st.form("playlist_form"):
            playlist_url = st.text_input(
                "YouTube Playlist URL",
                placeholder="https://www.youtube.com/playlist?list=...",
            )
            st.caption(
                "Playlist must be Public or Unlisted. "
                "Videos without auto-generated captions will be skipped. "
                "Large playlists may take several minutes to fetch."
            )
            submitted_playlist = st.form_submit_button(
                "Fetch Playlist Transcripts",
                use_container_width=True,
                type="primary",
            )

        if submitted_playlist and playlist_url.strip():
            _run_playlist_fetch(playlist_url.strip())
        elif submitted_playlist:
            st.warning("Please enter a playlist URL.")


def _render_save_transcript_panel() -> None:
    """Renders the download and save panel for a freshly fetched transcript.

    Displays the transcript content, provides a download button for the
    .txt file, and offers a form to paste key ideas and save the record.
    """
    pending = st.session_state.pending_transcript
    transcript_path = Path(pending["path"])

    if not transcript_path.exists():
        st.session_state.pending_transcript = None
        return

    transcript_text = transcript_path.read_text(encoding="utf-8")

    st.divider()
    st.subheader(f"📄 {pending['title']}")
    st.caption(f"Channel: {pending['channel']}")

    st.download_button(
        label="⬇️ Download Transcript (.txt)",
        data=transcript_text,
        file_name=f"{transcript_path.stem}.txt",
        mime="text/plain",
        use_container_width=True,
    )

    with st.expander("Preview transcript", expanded=False):
        st.text(transcript_text[:2000] + (
            "\n\n... [truncated for preview]"
            if len(transcript_text) > 2000 else ""
        ))

    st.divider()
    st.subheader("Save as Record")
    st.write(
        "Use the transcript above with your preferred LLM to generate "
        "key ideas, then paste them below — one idea per line."
    )

    with st.form("save_record_form"):
        ideas_input = st.text_area(
            "Paste key ideas here (one per line)",
            height=200,
            placeholder=(
                "IDEAS\n\n"
                "* Celts infantry move faster than other civilizations\n"
                "* Fast castle is a strong opening on Arabia"
            ),
        )
        save_submitted = st.form_submit_button(
            "Save Record",
            use_container_width=True,
            type="primary",
        )

    if save_submitted:
        ideas = _parse_ideas_input(ideas_input)
        if not ideas:
            st.warning("Please paste at least one key idea before saving.")
        else:
            try:
                record = create_record(
                    title=pending["title"],
                    channel=pending["channel"],
                    url=pending["url"],
                    transcript=transcript_text,
                )
                add_ideas(record["id"], ideas)
                st.success(
                    f"Record saved with {len(ideas)} key ideas. "
                    f"View and tag your ideas in the **Library** page."
                )
                st.session_state.pending_transcript = None
                st.rerun()
            except Exception as e:
                st.error(f"Could not save record: {e}")
                logger.error(f"Record save error: {e}")


def _run_playlist_fetch(playlist_url: str) -> None:
    """Fetches transcripts for all caption-enabled videos in a playlist.

    Args:
        playlist_url: The YouTube playlist URL to process.
    """
    import yt_dlp

    st.info("Fetching playlist metadata...")
    try:
        ydl_opts = {"quiet": True, "extract_flat": True, "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)

        entries = info.get("entries", [])
        if not entries:
            st.error("No videos found in playlist.")
            return

        total = len(entries)
        if total >= 10:
            st.warning(
                f"This playlist contains {total} videos. "
                f"Only caption-enabled videos will be fetched. "
                f"This may take a few minutes."
            )

        st.write(f"Found **{total} videos**. Fetching transcripts...")
        succeeded, failed = [], []
        progress = st.progress(0, text="Starting...")

        for i, entry in enumerate(entries, start=1):
            video_url = entry.get("url") or entry.get("webpage_url")
            video_title = entry.get("title", f"video_{i}")
            progress.progress(
                int((i / total) * 100),
                text=f"Fetching ({i}/{total}): {video_title[:50]}..."
            )

            if not video_url:
                failed.append(video_title)
                continue

            try:
                transcript_path, metadata = ingest(video_url)
                st.session_state.transcript_metadata[
                    transcript_path.stem
                ] = {
                    "title": metadata["title"],
                    "channel": metadata["uploader"],
                    "url": metadata["url"],
                }
                succeeded.append(metadata["title"])
            except ValueError:
                failed.append(video_title)
            except Exception as e:
                logger.error(f"Playlist fetch error for {video_title}: {e}")
                failed.append(video_title)

        progress.progress(100, text="Complete!")
        st.success("Playlist fetch complete!")

        col1, col2 = st.columns(2)
        col1.metric("Fetched", len(succeeded))
        col2.metric("Failed / No Captions", len(failed))

        if succeeded:
            st.info(
                "Transcripts saved to disk. Visit the **Library** page "
                "to save records with key ideas for each video."
            )
        if failed:
            with st.expander(f"Videos without captions ({len(failed)})"):
                for title in failed:
                    st.caption(f"⏭ {title}")

    except Exception as e:
        st.error(f"Playlist error: {e}")
        logger.error(f"Playlist fetch error: {e}")


# ---------------------------------------------------------------------------
# Page: Library
# ---------------------------------------------------------------------------

def _render_library_page() -> None:
    """Renders the library page for browsing and managing saved records.

    Displays all saved records with their key ideas. Each idea can be
    tagged, edited, or deleted. Records can also be deleted entirely.
    Unsaved transcripts available on disk are shown for easy saving.
    """
    st.header("📚 Library")
    st.write(
        "Browse your saved records, manage key ideas, and assign tags "
        "to organize your knowledge."
    )

    tab1, tab2 = st.tabs(["Saved Records", "Unsaved Transcripts"])

    with tab1:
        records = list_records()
        if not records:
            st.info(
                "No records saved yet. Fetch a transcript on the "
                "**Transcripts** page and save it with key ideas."
            )
        else:
            st.caption(f"{len(records)} record(s) saved")
            for summary in records:
                _render_record_card(summary)

    with tab2:
        st.write(
            "These transcripts have been fetched but not yet saved as "
            "records. Select one to add key ideas and save it."
        )
        _render_unsaved_transcripts()


def _render_record_card(summary: dict) -> None:
    """Renders a single record card with its ideas and management controls.

    Args:
        summary: A record summary dictionary from list_records().
    """
    record = get_record(summary["id"])
    status_label, status_icon = _get_review_status(record)
    record_id = summary["id"]

    with st.container(border=True):
        col1, col2 = st.columns([4, 1])

        with col1:
            st.subheader(summary["title"])
            st.caption(f"Channel: {summary['channel']}")
            st.caption(
                f"{status_icon} {status_label} · "
                f"{summary['idea_count']} idea(s) · "
                f"Saved: {summary['created_at'][:10]}"
            )

        with col2:
            if st.button(
                "🗑 Delete Record",
                key=f"del_rec_{record_id}",
                use_container_width=True,
            ):
                st.session_state.confirm_delete_record = record_id

        if st.session_state.confirm_delete_record == record_id:
            st.warning(
                "⚠️ This will permanently delete this record and all "
                "its key ideas. This action cannot be undone."
            )
            col_a, col_b = st.columns(2)
            if col_a.button(
                "Confirm Delete",
                key=f"confirm_del_{record_id}",
                type="primary",
                use_container_width=True,
            ):
                try:
                    delete_record(record_id)
                    st.session_state.confirm_delete_record = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")
            if col_b.button(
                "Cancel",
                key=f"cancel_del_{record_id}",
                use_container_width=True,
            ):
                st.session_state.confirm_delete_record = None
                st.rerun()

        if summary["idea_count"] > 0:
            with st.expander(
                f"View and manage {summary['idea_count']} idea(s)",
                expanded=False,
            ):
                all_tags = list_all_tags()
                idea_ids = [i["id"] for i in record["key_ideas"]]
                selections = _get_record_selections(record_id)

                col_all, col_none, _ = st.columns([1, 1, 4])
                if col_all.button(
                    "Select All",
                    key=f"sel_all_{record_id}",
                    use_container_width=True,
                ):
                    for iid in idea_ids:
                        selections[iid] = True
                    st.rerun()

                if col_none.button(
                    "Deselect All",
                    key=f"desel_all_{record_id}",
                    use_container_width=True,
                ):
                    for iid in idea_ids:
                        selections[iid] = False
                    st.rerun()

                for idea in record["key_ideas"]:
                    _render_idea_row(record_id, idea, all_tags)

                selected_ids = [
                    iid for iid in idea_ids
                    if selections.get(iid, False)
                ]

                if selected_ids:
                    st.divider()
                    st.caption(f"{len(selected_ids)} idea(s) selected")

                    col_tag, col_del = st.columns(2)

                    with col_tag:
                        with st.form(key=f"bulk_tag_form_{record_id}"):
                            bulk_tag = st.text_input(
                                "Tag to apply",
                                placeholder="e.g. early game",
                                key=f"bulk_tag_input_{record_id}",
                            )
                            tag_btn = st.form_submit_button(
                                "🏷️ Add Tag to Selected",
                                use_container_width=True,
                                type="primary",
                            )
                        if tag_btn and bulk_tag.strip():
                            errors = []
                            for iid in selected_ids:
                                try:
                                    add_tag_to_idea(
                                        record_id, iid, bulk_tag.strip()
                                    )
                                except Exception as e:
                                    errors.append(str(e))
                            if errors:
                                st.error(f"Some tags failed: {errors}")
                            else:
                                st.success(
                                    f"Tag '{bulk_tag.strip()}' added to "
                                    f"{len(selected_ids)} idea(s)."
                                )
                                selections.clear()
                                st.rerun()
                        elif tag_btn:
                            st.warning("Please enter a tag name.")

                    with col_del:
                        if st.button(
                            "🗑 Delete Selected",
                            key=f"bulk_del_{record_id}",
                            use_container_width=True,
                        ):
                            st.session_state.confirm_delete_idea = (
                                f"bulk_{record_id}"
                            )

                    if (
                        st.session_state.confirm_delete_idea
                        == f"bulk_{record_id}"
                    ):
                        st.warning(
                            f"⚠️ Permanently delete {len(selected_ids)} "
                            f"idea(s)? This cannot be undone."
                        )
                        ca, cb = st.columns(2)
                        if ca.button(
                            "Confirm Delete",
                            key=f"confirm_bulk_del_{record_id}",
                            type="primary",
                            use_container_width=True,
                        ):
                            for iid in selected_ids:
                                try:
                                    delete_idea(record_id, iid)
                                except Exception as e:
                                    logger.error(
                                        f"Could not delete idea {iid}: {e}"
                                    )
                            st.session_state.confirm_delete_idea = None
                            st.session_state[f"selections_{record_id}"] = {}
                            st.rerun()
                        if cb.button(
                            "Cancel",
                            key=f"cancel_bulk_del_{record_id}",
                            use_container_width=True,
                        ):
                            st.session_state.confirm_delete_idea = None
                            st.rerun()

        with st.expander("View transcript", expanded=False):
            st.text(record["transcript"][:3000] + (
                "\n\n... [truncated]"
                if len(record["transcript"]) > 3000 else ""
            ))
            st.download_button(
                label="⬇️ Download Transcript",
                data=record["transcript"],
                file_name=f"{record['title'][:60].replace(' ', '_')}.txt",
                mime="text/plain",
                key=f"dl_{record_id}",
            )


def _render_idea_row(record_id: str, idea: dict, all_tags: list[str]) -> None:
    """Renders a single key idea row with a toggle select button and edit control.

    Selection state is managed via a per-record selections dictionary
    in session state. Uses toggle buttons instead of checkboxes to
    avoid Streamlit widget state caching issues.

    Args:
        record_id: The ID of the parent record.
        idea: The idea dictionary containing id, text, and tags.
        all_tags: List of all existing tags for reference.
    """
    idea_id = idea["id"]
    is_editing = st.session_state.editing_idea == idea_id
    selections = _get_record_selections(record_id)
    is_selected = selections.get(idea_id, False)

    with st.container(border=True):
        col_toggle, col_text, col_edit = st.columns([0.5, 8, 1])

        toggle_label = "🔵" if is_selected else "⚪"
        if col_toggle.button(
            toggle_label,
            key=f"toggle_{record_id}_{idea_id}",
            use_container_width=True,
            help="Select this idea",
        ):
            selections[idea_id] = not is_selected
            st.rerun()

        if is_editing:
            with col_text:
                new_text = st.text_input(
                    "Edit idea",
                    value=idea["text"],
                    key=f"edit_input_{idea_id}",
                )
                c1, c2 = st.columns(2)
                if c1.button(
                    "Save",
                    key=f"save_edit_{idea_id}",
                    type="primary",
                    use_container_width=True,
                ):
                    try:
                        update_idea_text(record_id, idea_id, new_text)
                        st.session_state.editing_idea = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not update idea: {e}")
                if c2.button(
                    "Cancel",
                    key=f"cancel_edit_{idea_id}",
                    use_container_width=True,
                ):
                    st.session_state.editing_idea = None
                    st.rerun()
        else:
            with col_text:
                st.write(idea["text"])
                if idea["tags"]:
                    st.caption(
                        "Tags: " + " · ".join(
                            f"🏷️ {t}" for t in idea["tags"]
                        )
                    )

        if not is_editing:
            if col_edit.button(
                "✏️",
                key=f"edit_{idea_id}",
                use_container_width=True,
                help="Edit this idea",
            ):
                st.session_state.editing_idea = idea_id
                st.rerun()


def _render_unsaved_transcripts() -> None:
    """Renders a list of transcript files on disk without saved records.

    Identifies transcript files that do not yet have a corresponding
    record and allows the user to save them with key ideas or delete
    them permanently.
    """
    saved_titles = {r["title"] for r in list_records()}
    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*.txt"))

    unsaved = []
    for f in transcript_files:
        stem = f.stem
        cached = st.session_state.transcript_metadata.get(stem, {})
        title = cached.get("title", stem.replace("_", " "))
        if title not in saved_titles:
            unsaved.append((f, stem, cached))

    if not unsaved:
        st.info("All fetched transcripts have been saved as records.")
        return

    st.caption(f"{len(unsaved)} unsaved transcript(s)")

    for transcript_file, stem, cached in unsaved:
        title = cached.get("title", stem.replace("_", " "))
        cached_channel = cached.get("channel", "")
        cached_url = cached.get("url", "")

        with st.container(border=True):
            col_title, col_del = st.columns([5, 1])

            with col_title:
                st.write(f"**{title}**")
                if cached_channel:
                    st.caption(f"Channel: {cached_channel}")

            if col_del.button(
                "🗑",
                key=f"del_unsaved_{stem}",
                use_container_width=True,
                help="Delete this transcript",
            ):
                st.session_state.confirm_delete_record = f"unsaved_{stem}"

            if st.session_state.confirm_delete_record == f"unsaved_{stem}":
                st.warning(
                    "⚠️ Permanently delete this transcript file? "
                    "This cannot be undone."
                )
                ca, cb = st.columns(2)
                if ca.button(
                    "Confirm Delete",
                    key=f"confirm_del_unsaved_{stem}",
                    type="primary",
                    use_container_width=True,
                ):
                    try:
                        transcript_file.unlink()
                        st.session_state.transcript_metadata.pop(stem, None)
                        st.session_state.confirm_delete_record = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not delete transcript: {e}")
                if cb.button(
                    "Cancel",
                    key=f"cancel_del_unsaved_{stem}",
                    use_container_width=True,
                ):
                    st.session_state.confirm_delete_record = None
                    st.rerun()

            transcript_text = transcript_file.read_text(encoding="utf-8")

            st.download_button(
                label="⬇️ Download Transcript",
                data=transcript_text,
                file_name=transcript_file.name,
                mime="text/plain",
                key=f"dl_unsaved_{stem}",
            )

            with st.expander("Save as record", expanded=False):
                with st.form(key=f"save_unsaved_{stem}"):
                    channel = st.text_input(
                        "Channel name",
                        value=cached_channel,
                        placeholder="e.g. Hera",
                        key=f"channel_{stem}",
                    )
                    video_url = st.text_input(
                        "Video URL",
                        value=cached_url,
                        placeholder="https://www.youtube.com/watch?v=...",
                        key=f"url_{stem}",
                    )
                    ideas_input = st.text_area(
                        "Paste key ideas (one per line)",
                        height=150,
                        key=f"ideas_{stem}",
                    )
                    save_btn = st.form_submit_button(
                        "Save Record",
                        use_container_width=True,
                        type="primary",
                    )

                if save_btn:
                    ideas = _parse_ideas_input(ideas_input)
                    if not channel.strip():
                        st.warning("Please enter the channel name.")
                    elif not video_url.strip():
                        st.warning("Please enter the video URL.")
                    elif not ideas:
                        st.warning("Please paste at least one key idea.")
                    else:
                        try:
                            record = create_record(
                                title=title,
                                channel=channel.strip(),
                                url=video_url.strip(),
                                transcript=transcript_text,
                            )
                            add_ideas(record["id"], ideas)
                            st.success(f"Saved with {len(ideas)} ideas.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not save: {e}")


# ---------------------------------------------------------------------------
# Page: Topics
# ---------------------------------------------------------------------------

def _render_topics_page() -> None:
    """Renders the topics page showing all ideas grouped by tag.

    Displays each unique tag as a topic section containing all ideas
    across all records that share that tag.
    """
    st.header("🏷️ Topics")
    st.write(
        "Each tag you create becomes a topic here. Browse all ideas "
        "across your videos that share the same tag."
    )

    tags = list_all_tags()

    if not tags:
        st.info(
            "No topics yet. Go to the **Library** page and add tags "
            "to your key ideas to create topics."
        )
        return

    st.caption(f"{len(tags)} topic(s) across your knowledge base")

    for tag in tags:
        ideas = get_ideas_by_tag(tag)
        with st.expander(
            f"🏷️ {tag.title()} — {len(ideas)} idea(s)",
            expanded=False,
        ):
            topic_text = "IDEAS\n\n" + "\n".join(
                f"* {idea['text']}" for idea in ideas
            )
            st.download_button(
                label=f"⬇️ Download {tag.title()} ideas (.txt)",
                data=topic_text,
                file_name=f"{tag.replace(' ', '_')}_ideas.txt",
                mime="text/plain",
                key=f"dl_topic_{tag}",
            )

            for idea in ideas:
                with st.container(border=True):
                    st.write(idea["text"])
                    col1, col2 = st.columns([3, 1])
                    col1.caption(
                        f"From: **{idea['record_title']}** · {idea['channel']}"
                    )
                    other_tags = [t for t in idea["tags"] if t != tag]
                    if other_tags:
                        col2.caption(f"Also: {', '.join(other_tags)}")


# ---------------------------------------------------------------------------
# Page: Quiz
# ---------------------------------------------------------------------------

def _render_quiz_page() -> None:
    """Renders the quiz management and playback page.

    Allows users to save externally generated quizzes associated with
    a record, browse saved quizzes, and take interactive quiz sessions.
    """
    st.header("🧠 Quiz")
    st.write(
        "Save quizzes generated externally from your key ideas and "
        "take them interactively here."
    )

    if st.session_state.quiz_active:
        _render_active_quiz()
        return

    tab1, tab2 = st.tabs(["Saved Quizzes", "Add Quiz"])

    with tab1:
        quizzes = list_quizzes()

        if not quizzes:
            st.info("No quizzes saved yet. Add a quiz in the **Add Quiz** tab.")
        else:
            st.caption(f"{len(quizzes)} quiz(zes) saved")
            for quiz_summary in quizzes:
                with st.container(border=True):
                    col1, col2, col3 = st.columns([4, 1, 1])
                    col1.subheader(quiz_summary["title"])
                    col1.caption(
                        f"{quiz_summary['question_count']} questions · "
                        f"Saved: {quiz_summary['created_at'][:10]}"
                    )

                    if col2.button(
                        "▶ Start",
                        key=f"start_{quiz_summary['id']}",
                        type="primary",
                        use_container_width=True,
                    ):
                        try:
                            quiz = get_quiz(quiz_summary["id"])
                            st.session_state.quiz_active = True
                            st.session_state.quiz_data = quiz
                            st.session_state.quiz_index = 0
                            st.session_state.quiz_score = 0
                            st.session_state.quiz_results = []
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not load quiz: {e}")

                    if col3.button(
                        "🗑",
                        key=f"del_quiz_{quiz_summary['id']}",
                        use_container_width=True,
                        help="Delete this quiz",
                    ):
                        st.session_state.confirm_delete_quiz = quiz_summary["id"]

                    if st.session_state.confirm_delete_quiz == quiz_summary["id"]:
                        st.warning(
                            "⚠️ Permanently delete this quiz? "
                            "This cannot be undone."
                        )
                        ca, cb = st.columns(2)
                        if ca.button(
                            "Confirm Delete",
                            key=f"confirm_del_quiz_{quiz_summary['id']}",
                            type="primary",
                            use_container_width=True,
                        ):
                            try:
                                delete_quiz(quiz_summary["id"])
                                st.session_state.confirm_delete_quiz = None
                                st.rerun()
                            except Exception as e:
                                st.error(f"Delete failed: {e}")
                        if cb.button(
                            "Cancel",
                            key=f"cancel_del_quiz_{quiz_summary['id']}",
                            use_container_width=True,
                        ):
                            st.session_state.confirm_delete_quiz = None
                            st.rerun()

    with tab2:
        st.write(
            "Paste a quiz generated from your key ideas. Each question "
            "must be entered in the format below."
        )
        st.info(
            "Expected format — one question block per line group:\n\n"
            "**QUESTION:** What unit counters cavalry?\n\n"
            "**CORRECT:** Spearman\n\n"
            "**WRONG1:** Knight\n\n"
            "**WRONG2:** Archer\n\n"
            "**WRONG3:** Monk\n\n"
            "Separate each question block with a blank line."
        )

        records = list_records()
        if not records:
            st.warning(
                "No records found. Save a record in the Library "
                "before adding a quiz."
            )
            return

        with st.form("add_quiz_form"):
            quiz_title = st.text_input(
                "Quiz title",
                placeholder="e.g. Celts Strategy Quiz",
            )
            record_options = {r["title"]: r["id"] for r in records}
            selected_record = st.selectbox(
                "Associate with record",
                options=list(record_options.keys()),
            )
            quiz_text = st.text_area(
                "Paste quiz content",
                height=400,
                placeholder=(
                    "QUESTION: What unit counters cavalry?\n"
                    "CORRECT: Spearman\n"
                    "WRONG1: Knight\n"
                    "WRONG2: Archer\n"
                    "WRONG3: Monk\n\n"
                    "QUESTION: What age comes after Feudal?\n"
                    "CORRECT: Castle Age\n"
                    "WRONG1: Dark Age\n"
                    "WRONG2: Imperial Age\n"
                    "WRONG3: Bronze Age"
                ),
            )
            save_quiz_btn = st.form_submit_button(
                "Save Quiz",
                use_container_width=True,
                type="primary",
            )

        if save_quiz_btn:
            if not quiz_title.strip():
                st.warning("Please enter a quiz title.")
            elif not quiz_text.strip():
                st.warning("Please paste quiz content.")
            else:
                questions = _parse_quiz_text(quiz_text)
                if not questions:
                    st.error(
                        "Could not parse any questions. Check the "
                        "format and try again."
                    )
                else:
                    try:
                        record_id = record_options[selected_record]
                        quiz = save_quiz(
                            record_id=record_id,
                            title=quiz_title.strip(),
                            questions=questions,
                        )
                        st.success(
                            f"Quiz saved with {quiz['question_count']} questions."
                        )
                    except Exception as e:
                        st.error(f"Could not save quiz: {e}")


def _parse_quiz_text(text: str) -> list[dict]:
    """Parses pasted quiz text into a list of question dictionaries.

    Expects questions separated by blank lines with QUESTION, CORRECT,
    WRONG1, WRONG2, and WRONG3 prefixed lines.

    Args:
        text: The raw pasted quiz text to parse.

    Returns:
        A list of question dictionaries each containing 'question',
        'correct_answer', and 'choices' keys. Returns an empty list
        if no valid questions are found.
    """
    questions = []
    blocks = text.strip().split("\n\n")

    for block in blocks:
        fields = {}
        for line in block.splitlines():
            line = line.strip()
            for key in ("QUESTION", "CORRECT", "WRONG1", "WRONG2", "WRONG3"):
                if line.upper().startswith(f"{key}:"):
                    fields[key] = line[len(key) + 1:].strip()

        required = ("QUESTION", "CORRECT", "WRONG1", "WRONG2", "WRONG3")
        if all(k in fields for k in required):
            choices = [
                fields["CORRECT"],
                fields["WRONG1"],
                fields["WRONG2"],
                fields["WRONG3"],
            ]
            random.shuffle(choices)
            questions.append({
                "question": fields["QUESTION"],
                "correct_answer": fields["CORRECT"],
                "choices": choices,
            })

    return questions


def _render_active_quiz() -> None:
    """Renders the active quiz session interface.

    Presents questions one at a time with multiple choice answers,
    provides immediate feedback, and shows a summary on completion.
    """
    quiz = st.session_state.quiz_data
    idx = st.session_state.quiz_index
    questions = quiz["questions"]
    total = len(questions)

    if idx >= total:
        _render_quiz_summary(quiz)
        return

    st.progress(idx / total, text=f"Question {idx + 1} of {total}")
    st.subheader(quiz["title"])

    question = questions[idx]

    with st.container(border=True):
        st.subheader(f"Q{idx + 1}. {question['question']}")

        selected = st.radio(
            "Choose your answer:",
            options=question["choices"],
            key=f"quiz_q_{idx}",
            index=None,
        )

        col1, col2 = st.columns([1, 4])
        submit = col1.button(
            "Submit",
            type="primary",
            use_container_width=True,
        )

        if submit and selected:
            is_correct = (
                selected.strip().lower()
                == question["correct_answer"].strip().lower()
            )
            st.session_state.quiz_results.append({
                "question": question["question"],
                "correct": is_correct,
                "user_answer": selected,
                "correct_answer": question["correct_answer"],
            })

            if is_correct:
                st.session_state.quiz_score += 1
                st.success("Correct!")
            else:
                st.error(
                    f"Incorrect. The correct answer was: "
                    f"**{question['correct_answer']}**"
                )

            st.session_state.quiz_index += 1
            next_label = "Next Question" if idx + 1 < total else "See Results"
            if st.button(next_label, use_container_width=True):
                st.rerun()

        elif submit and not selected:
            st.warning("Please select an answer before submitting.")


def _render_quiz_summary(quiz: dict) -> None:
    """Renders the quiz completion summary.

    Args:
        quiz: The full quiz dictionary that was just completed.
    """
    score = st.session_state.quiz_score
    total = len(quiz["questions"])
    pct = round((score / total) * 100, 1) if total > 0 else 0

    st.balloons()
    st.subheader("Quiz Complete!")

    col1, col2 = st.columns(2)
    col1.metric("Score", f"{score}/{total}")
    col2.metric("Percentage", f"{pct}%")

    if pct >= 80:
        st.success("Excellent work!")
    elif pct >= 60:
        st.info("Good effort! Review the incorrect answers below.")
    else:
        st.warning("Keep studying and try again!")

    with st.expander("Review your answers", expanded=False):
        for i, result in enumerate(st.session_state.quiz_results, start=1):
            icon = "✅" if result["correct"] else "❌"
            st.write(f"{icon} **Q{i}:** {result['question']}")
            if not result["correct"]:
                st.caption(f"Your answer: {result['user_answer']}")
                st.caption(f"Correct answer: {result['correct_answer']}")
            st.divider()

    if st.button(
        "Back to Quizzes",
        use_container_width=True,
        type="primary",
    ):
        st.session_state.quiz_active = False
        st.session_state.quiz_data = None
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

    if page == "Transcripts":
        _render_transcripts_page()
    elif page == "Library":
        _render_library_page()
    elif page == "Topics":
        _render_topics_page()
    elif page == "Quiz":
        _render_quiz_page()


if __name__ == "__main__":
    main()