"""Filter ideas command for theGist.

This module implements the filter-ideas command which uses sentence
transformer embeddings to identify ideas that may not be relevant
to the stated subject. The user reviews each flagged idea
interactively and their decisions are saved as labeled examples
for future classifier training.

Previously labeled ideas are skipped automatically unless --relabel
is specified. Original record ideas are never modified. Irrelevant
ideas are marked inactive in the subject_ideas pool.

Usage:
    thegist filter-ideas --subject <subject> [--threshold <float>]
        [--min-threshold <float>] [--relabel]
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from thegist.src.database import (
    deactivate_subject_idea,
    delete_idea_label,
    get_active_subject_ideas,
    get_labeled_idea_ids,
    insert_idea_label,
)

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.25
MIN_IDEAS = 10


def register(subparsers) -> None:
    """Registers the filter-ideas command with the top level parser.

    Args:
        subparsers: The subparsers group from the top level parser.
    """
    parser = subparsers.add_parser(
        "filter-ideas",
        help="Find and remove ideas not relevant to the subject.",
        description=(
            "Uses sentence embeddings to find ideas that may not be "
            "relevant to the subject. You review each flagged idea "
            "interactively. Decisions are saved as labeled examples "
            "for future classifier training. Previously labeled ideas "
            "are skipped unless --relabel is specified."
        ),
    )
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="Subject name to filter ideas for.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            f"Upper similarity bound. Ideas below this value are flagged. "
            f"Default is {DEFAULT_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.0,
        help=(
            "Lower similarity bound. Only ideas at or above this value "
            "and below --threshold are flagged. Defaults to 0.0."
        ),
    )
    parser.add_argument(
        "--relabel",
        action="store_true",
        default=False,
        help=(
            "Include previously labeled ideas in the review session. "
            "Use this to revisit and correct earlier decisions."
        ),
    )
    parser.set_defaults(func=run)


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _load_model() -> SentenceTransformer:
    """Loads the sentence transformer embedding model.

    Returns:
        The loaded SentenceTransformer model instance.
    """
    print("Loading embedding model...")
    return SentenceTransformer(MODEL_NAME)


def _build_reference_embedding(
    model: SentenceTransformer,
    subject: str,
) -> np.ndarray:
    """Builds a reference embedding for the subject.

    Encodes several descriptive phrases about the subject and
    averages them into a centroid to form a robust reference
    representation.

    Args:
        model: The loaded SentenceTransformer model.
        subject: The subject name to build a reference for.

    Returns:
        A normalized numpy array representing the subject embedding.
    """
    reference_phrases = [
        subject,
        f"key insight about {subject}",
        f"strategy tip for {subject}",
        f"important concept in {subject}",
        f"expert knowledge about {subject}",
    ]

    embeddings = model.encode(reference_phrases, normalize_embeddings=True)
    centroid = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid


def _compute_similarities(
    model: SentenceTransformer,
    ideas: list[dict],
    reference: np.ndarray,
) -> list[tuple[dict, float]]:
    """Computes cosine similarity between each idea and the reference.

    Args:
        model: The loaded SentenceTransformer model.
        ideas: List of active subject idea dictionaries.
        reference: The normalized reference embedding array.

    Returns:
        A list of (idea, similarity) tuples sorted ascending by
        similarity so the least relevant ideas appear first.
    """
    texts = [idea["text"] for idea in ideas]
    embeddings = model.encode(texts, normalize_embeddings=True)
    similarities = embeddings @ reference
    pairs = list(zip(ideas, similarities.tolist()))
    pairs.sort(key=lambda x: x[1])
    return pairs


def _flag_candidates(
    pairs: list[tuple[dict, float]],
    threshold: float,
    min_threshold: float = 0.0,
    labeled_ids: set[str] = None,
    relabel: bool = False,
) -> list[tuple[dict, float]]:
    """Returns ideas within the similarity range that need labeling.

    Excludes previously labeled ideas unless relabel is True.

    Args:
        pairs: List of (idea, similarity) tuples.
        threshold: Upper bound — ideas below this are flagged.
        min_threshold: Lower bound — ideas at or above this included.
        labeled_ids: Set of idea IDs already labeled this subject.
        relabel: If True include previously labeled ideas.

    Returns:
        A filtered list of (idea, similarity) tuples.
    """
    labeled_ids = labeled_ids or set()
    results = []
    for idea, sim in pairs:
        if sim < min_threshold or sim >= threshold:
            continue
        if not relabel and idea["id"] in labeled_ids:
            continue
        results.append((idea, sim))
    return results


def _display_candidate(
    idea: dict,
    similarity: float,
    index: int,
    total: int,
    is_first: bool,
) -> None:
    """Prints a flagged idea for the user to review.

    Args:
        idea: The idea dictionary to display.
        similarity: The cosine similarity score to the subject.
        index: The current candidate number in the session.
        total: Total number of flagged candidates.
        is_first: Whether this is the first candidate in the session.
    """
    print(f"\n{'-' * 60}")
    print(f"Candidate {index} of {total}  (relevance: {similarity:.2f})\n")
    print(f"  Idea   : {idea['text']}")
    print(f"  From   : {idea['record_title']} | {idea['channel']}")
    if idea.get("uploaded_at"):
        print(f"  Date   : {idea['uploaded_at']}")
    print()


def _ask_action(is_first: bool) -> str:
    """Prompts the user to classify a flagged idea.

    Undo is disabled for the first candidate since there is nothing
    to undo yet.

    Args:
        is_first: Whether this is the first candidate in the session.

    Returns:
        One of 'keep', 'remove', 'skip', or 'undo'.
    """
    if is_first:
        prompt = "Action? [K]eep / [R]emove / [S]kip (default: Keep): "
        valid_undo = False
    else:
        prompt = (
            "Action? [K]eep / [R]emove / [S]kip / [U]ndo previous "
            "(default: Keep): "
        )
        valid_undo = True

    while True:
        raw = input(prompt).strip().lower()
        if raw in ("", "k", "keep"):
            return "keep"
        if raw in ("r", "remove"):
            return "remove"
        if raw in ("s", "skip"):
            return "skip"
        if valid_undo and raw in ("u", "undo"):
            return "undo"
        if not valid_undo and raw in ("u", "undo"):
            print("  Nothing to undo yet.")
        else:
            options = "K, R, S, or U" if valid_undo else "K, R, or S"
            print(f"  Please enter {options}.")


def _apply_decision(
    idea: dict,
    subject: str,
    decision: str,
    stats: dict,
) -> None:
    """Applies the user's relevance decision to an idea.

    Args:
        idea: The idea dictionary being reviewed.
        subject: The subject the idea belongs to.
        decision: One of 'keep' or 'remove'.
        stats: The session statistics dictionary to update.
    """
    if decision == "keep":
        insert_idea_label(idea["id"], subject, label=1)
        print("  → Marked as relevant.")
        stats["kept"] += 1

    elif decision == "remove":
        deactivate_subject_idea(idea["id"], subject)
        insert_idea_label(idea["id"], subject, label=0)
        print("  → Marked as irrelevant and removed from subject pool.")
        stats["removed"] += 1


def _undo_decision(
    prev_idea: dict,
    prev_decision: str,
    subject: str,
    stats: dict,
) -> None:
    """Reverses the previous labeling decision.

    Restores the idea to active in subject_ideas if it was removed
    and deletes its label from idea_labels.

    Args:
        prev_idea: The idea dictionary from the previous decision.
        prev_decision: The decision that was made — 'keep' or 'remove'.
        subject: The subject the idea belongs to.
        stats: The session statistics dictionary to update.
    """
    delete_idea_label(prev_idea["id"], subject)

    if prev_decision == "remove":
        # Restore the idea to active in the subject pool
        from thegist.src.database import _connect
        with _connect() as conn:
            conn.execute(
                """
                UPDATE subject_ideas
                SET is_active = 1
                WHERE idea_id = ?
                AND subject = ?
                """,
                (prev_idea["id"], subject),
            )
        stats["removed"] -= 1
        print(
            f"  → Undone. '{prev_idea['text'][:50]}' "
            f"restored to active pool."
        )

    elif prev_decision == "keep":
        stats["kept"] -= 1
        print(
            f"  → Undone. Label removed for "
            f"'{prev_idea['text'][:50]}'."
        )


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

def run(args) -> None:
    """Executes the filter-ideas command.

    Loads active subject ideas, computes embedding similarity,
    flags candidates in the specified range that have not been
    labeled yet, and guides the user through an interactive
    review session with undo support.

    Args:
        args: Parsed argument namespace containing subject,
            threshold, min_threshold, and relabel.
    """
    subject = args.subject
    threshold = args.threshold
    min_threshold = args.min_threshold
    relabel = args.relabel

    if not 0.0 < threshold < 1.0:
        print("\nError: --threshold must be between 0.0 and 1.0.\n")
        return

    if min_threshold >= threshold:
        print(
            "\nError: --min-threshold must be less than --threshold.\n"
        )
        return

    ideas = get_active_subject_ideas(subject)

    if not ideas:
        print(f"\nNo active ideas found for subject: {subject}\n")
        return

    if len(ideas) < MIN_IDEAS:
        print(
            f"\nNot enough ideas to filter. "
            f"Found {len(ideas)} but minimum is {MIN_IDEAS}.\n"
        )
        return

    labeled_ids = set() if relabel else get_labeled_idea_ids(subject)

    print(f"\ntheGist — Filter Ideas")
    print(f"Subject   : {subject}")
    print(f"Ideas     : {len(ideas)}")
    if min_threshold > 0.0:
        print(f"Range     : {min_threshold} — {threshold}")
    else:
        print(f"Threshold : {threshold}")
    if relabel:
        print(f"Mode      : Relabel (includes previously labeled ideas)")
    if labeled_ids:
        print(f"Already labeled: {len(labeled_ids)} idea(s) will be skipped")
    print()

    model = _load_model()
    reference = _build_reference_embedding(model, subject)

    print("Computing similarities...")
    pairs = _compute_similarities(model, ideas, reference)
    candidates = _flag_candidates(
        pairs, threshold, min_threshold, labeled_ids, relabel
    )

    if not candidates:
        print(
            f"\nNo unlabeled ideas found in the specified range. "
            f"All candidates have already been reviewed.\n"
            f"Use --relabel to revisit previous decisions or "
            f"adjust --threshold to find new candidates.\n"
        )
        return

    print(f"Found {len(candidates)} unlabeled candidate(s) to review.\n")

    stats = {
        "kept": 0,
        "removed": 0,
        "skipped": 0,
    }

    # Track the previous decision for undo support
    prev_idea = None
    prev_decision = None

    i = 0
    while i < len(candidates):
        idea, similarity = candidates[i]
        is_first = prev_idea is None

        _display_candidate(idea, similarity, i + 1, len(candidates), is_first)
        action = _ask_action(is_first)

        if action == "undo":
            _undo_decision(prev_idea, prev_decision, subject, stats)
            # Step back to re-display the previous candidate
            i -= 1
            prev_idea = None
            prev_decision = None
            continue

        if action in ("keep", "remove"):
            _apply_decision(idea, subject, action, stats)
            prev_idea = idea
            prev_decision = action
        else:
            # skip
            print("  → Skipped.")
            stats["skipped"] += 1
            prev_idea = None
            prev_decision = None

        i += 1

    print(f"\n{'-' * 60}")
    print(f"\nFilter session complete.\n")
    print(f"  Reviewed : {len(candidates)} idea(s)")
    print(f"  Kept     : {stats['kept']}")
    print(f"  Removed  : {stats['removed']}")
    print(f"  Skipped  : {stats['skipped']}")
    total_labels = stats["kept"] + stats["removed"]
    print(f"  Labels saved for training : {total_labels}\n")