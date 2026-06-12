"""Filter ideas command for theGist.

This module implements the filter-ideas command which uses sentence
transformer embeddings to identify ideas that may not be relevant
to the stated subject. The user reviews each flagged idea
interactively and their decisions are saved as labeled examples
for future classifier training.

Original record ideas are never modified. Irrelevant ideas are
marked inactive in the subject_ideas pool.

Usage:
    thegist filter-ideas --subject <subject> [--threshold <float>]
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from thegist.src.database import (
    deactivate_subject_idea,
    get_active_subject_ideas,
    insert_idea_label,
)

# Embedding model — lightweight and effective for semantic similarity
MODEL_NAME = "all-MiniLM-L6-v2"

# Default similarity threshold below which ideas are flagged
# Ideas with cosine similarity to the subject reference below this
# value are considered potentially irrelevant
DEFAULT_THRESHOLD = 0.25

# Minimum number of active ideas required to run filter-ideas
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
            "for future classifier training."
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
            f"Similarity threshold between 0.0 and 1.0. Ideas with "
            f"cosine similarity to the subject below this value are "
            f"flagged. Lower values flag fewer ideas. "
            f"Default is {DEFAULT_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.0,
        help=(
            "Lower bound of the similarity range to review. "
            "Only ideas with similarity at or above this value and below "
            "--threshold are flagged. Defaults to 0.0 to show all ideas "
            "below --threshold. Use this to review only new candidates "
            "when increasing the threshold from a previous session."
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

    The reference embedding represents what relevant ideas for this
    subject should semantically look like. It is the centroid of
    several descriptive phrases about the subject to give a more
    robust representation than a single phrase.

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

    # Normalize the centroid
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
        A list of (idea, similarity) tuples sorted by similarity
        ascending so the least relevant ideas appear first.
    """
    texts = [idea["text"] for idea in ideas]
    embeddings = model.encode(texts, normalize_embeddings=True)

    # Cosine similarity is dot product of normalized vectors
    similarities = embeddings @ reference

    pairs = list(zip(ideas, similarities.tolist()))
    pairs.sort(key=lambda x: x[1])
    return pairs


def _flag_candidates(
    pairs: list[tuple[dict, float]],
    threshold: float,
    min_threshold: float = 0.0,
) -> list[tuple[dict, float]]:
    """Returns ideas with similarity within the specified range.

    Args:
        pairs: List of (idea, similarity) tuples.
        threshold: The upper bound — ideas below this are flagged.
        min_threshold: The lower bound — ideas at or above this
            are included. Defaults to 0.0 to include all ideas
            below threshold.

    Returns:
        A filtered list of (idea, similarity) tuples where similarity
        is at or above min_threshold and below threshold.
    """
    return [
        (idea, sim)
        for idea, sim in pairs
        if min_threshold <= sim < threshold
    ]


def _display_candidate(
    idea: dict,
    similarity: float,
    index: int,
    total: int,
) -> None:
    """Prints a flagged idea for the user to review.

    Args:
        idea: The idea dictionary to display.
        similarity: The cosine similarity score to the subject.
        index: The current candidate number in the session.
        total: Total number of flagged candidates.
    """
    print(f"\n{'-' * 60}")
    print(f"Candidate {index} of {total}  (relevance: {similarity:.2f})\n")
    print(f"  Idea   : {idea['text']}")
    print(f"  From   : {idea['record_title']} | {idea['channel']}")
    if idea.get("uploaded_at"):
        print(f"  Date   : {idea['uploaded_at']}")
    print()


def _ask_relevance() -> str:
    """Prompts the user to classify a flagged idea.

    Returns:
        One of 'keep', 'remove', or 'skip'.
    """
    valid = {"k", "keep", "r", "remove", "s", "skip", ""}
    while True:
        raw = input(
            "Action? [K]eep / [R]emove / [S]kip (default: Keep): "
        ).strip().lower()
        if raw in valid:
            if raw in ("", "k", "keep"):
                return "keep"
            if raw in ("r", "remove"):
                return "remove"
            if raw in ("s", "skip"):
                return "skip"
        print("  Please enter K, R, or S.")


def _apply_decision(
    idea: dict,
    subject: str,
    decision: str,
    stats: dict,
) -> None:
    """Applies the user's relevance decision to an idea.

    Saves the label to the database regardless of decision for
    future classifier training. Deactivates the idea in the subject
    pool if marked irrelevant.

    Args:
        idea: The idea dictionary being reviewed.
        subject: The subject the idea belongs to.
        decision: One of 'keep', 'remove', or 'skip'.
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

    elif decision == "skip":
        print("  → Skipped. No label recorded.")
        stats["skipped"] += 1


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

def run(args) -> None:
    """Executes the filter-ideas command.

    Loads active subject ideas, computes embedding similarity to the
    subject reference, flags candidates below the threshold, and
    guides the user through an interactive review session.

    Args:
        args: Parsed argument namespace containing subject and threshold.
    """
    subject = args.subject
    threshold = args.threshold

    if not 0.0 < threshold < 1.0:
        print("\nError: --threshold must be between 0.0 and 1.0 exclusive.\n")
        return

    ideas = get_active_subject_ideas(subject)

    if not ideas:
        print(f"\nNo active ideas found for subject: {subject}")
        print("Run add-ideas first to populate the subject pool.\n")
        return

    if len(ideas) < MIN_IDEAS:
        print(
            f"\nNot enough ideas to filter. "
            f"Found {len(ideas)} but minimum is {MIN_IDEAS}.\n"
        )
        return

    print(f"\ntheGist — Filter Ideas")
    print(f"Subject   : {subject}")
    print(f"Ideas     : {len(ideas)}")
    if args.min_threshold > 0.0:
        print(f"Range     : {args.min_threshold} — {threshold}")
    else:
        print(f"Threshold : {threshold}")
    print()

    model = _load_model()
    reference = _build_reference_embedding(model, subject)

    print("Computing similarities...")
    pairs = _compute_similarities(model, ideas, reference)
    candidates = _flag_candidates(pairs, threshold, args.min_threshold)

    if not candidates:
        print(
            f"\nNo ideas flagged below threshold {threshold}. "
            f"All ideas appear relevant to: {subject}\n"
            f"Try raising the threshold to flag more candidates.\n"
        )
        return

    print(f"Found {len(candidates)} candidate(s) to review.\n")

    stats = {
        "kept": 0,
        "removed": 0,
        "skipped": 0,
    }

    for i, (idea, similarity) in enumerate(candidates, start=1):
        _display_candidate(idea, similarity, i, len(candidates))
        decision = _ask_relevance()
        _apply_decision(idea, subject, decision, stats)

    print(f"\n{'-' * 60}")
    print(f"\nFilter session complete.\n")
    print(f"  Reviewed : {len(candidates)} idea(s)")
    print(f"  Kept     : {stats['kept']}")
    print(f"  Removed  : {stats['removed']}")
    print(f"  Skipped  : {stats['skipped']}")
    total_labels = stats["kept"] + stats["removed"]
    print(f"  Labels saved for training : {total_labels}\n")