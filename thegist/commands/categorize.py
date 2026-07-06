"""Categorize command for theGist.

This module implements the categorize command which guides the user
through an interactive session assigning active subject ideas to
named categories. Ideas are suggested a category based on embedding
similarity between the idea text and each category description.
The user accepts the suggestion or overrides it.

By default processes uncategorized ideas and ideas currently in
Other. Use --all to re-evaluate all ideas in the subject.

Usage:
    thegist categorize --subject <subject> [--all]
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from thegist.src.database import (
    assign_idea_category,
    get_active_subject_ideas,
    get_all_categorized_ideas,
    get_categories,
    get_category_by_name,
    get_ideas_by_category,
    get_uncategorized_ideas,
    get_other_ideas,
)

MODEL_NAME = "all-MiniLM-L6-v2"


def register(subparsers) -> None:
    """Registers the categorize command with the top level parser.

    Args:
        subparsers: The subparsers group from the top level parser.
    """
    parser = subparsers.add_parser(
        "categorize",
        help="Assign ideas to categories interactively.",
        description=(
            "Guides you through assigning ideas to named categories. "
            "A category is suggested based on semantic similarity "
            "between the idea and each category description. "
            "By default processes uncategorized ideas and ideas "
            "currently in Other. Use --all to re-evaluate every idea."
        ),
    )
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="Subject name to categorize ideas for.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help=(
            "Maximum number of ideas to process in this session. "
            "Progress is saved automatically so the next run "
            "continues where you left off."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        dest="all_ideas",
        help=(
            "Re-evaluate all ideas in the subject regardless of "
            "their current category. Useful after adding a new category."
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


def _build_category_embeddings(
    model: SentenceTransformer,
    categories: list[dict],
) -> list[tuple[dict, np.ndarray]]:
    """Builds normalized embeddings for each category description.

    Args:
        model: The loaded SentenceTransformer model.
        categories: List of category dictionaries.

    Returns:
        A list of (category, embedding) tuples.
    """
    descriptions = [cat["description"] for cat in categories]
    embeddings = model.encode(descriptions, normalize_embeddings=True)
    return list(zip(categories, embeddings))


def _suggest_category(
    idea_text: str,
    model: SentenceTransformer,
    category_embeddings: list[tuple[dict, np.ndarray]],
) -> tuple[dict, float]:
    """Suggests the most semantically similar category for an idea.

    Computes cosine similarity between the idea embedding and each
    category description embedding and returns the best match.

    Args:
        idea_text: The idea text to suggest a category for.
        model: The loaded SentenceTransformer model.
        category_embeddings: List of (category, embedding) tuples.

    Returns:
        A tuple of (suggested_category, similarity_score).
    """
    idea_embedding = model.encode(idea_text, normalize_embeddings=True)

    best_category = None
    best_score = -1.0

    for category, cat_embedding in category_embeddings:
        score = float(np.dot(idea_embedding, cat_embedding))
        if score > best_score:
            best_score = score
            best_category = category

    return best_category, best_score


def _display_idea(
    idea: dict,
    index: int,
    total: int,
    suggested: dict,
    similarity: float,
    categories: list[dict],
) -> None:
    """Displays an idea with its suggested category and options.

    Args:
        idea: The idea dictionary to display.
        index: Current position in the session.
        total: Total number of ideas to process.
        suggested: The suggested category dictionary.
        similarity: The similarity score for the suggestion.
        categories: All available categories for numbering.
    """
    print(f"\n{'-' * 60}")
    print(f"Idea {index} of {total}\n")
    print(f"  Text   : {idea['text']}")
    print(f"  From   : {idea['record_title']} | {idea['channel']}")
    if idea.get("uploaded_at"):
        print(f"  Date   : {idea['uploaded_at']}")
    if idea.get("current_category"):
        print(f"  Current: {idea['current_category']}")
    print()
    print(f"  Suggested: {suggested['name']} (similarity: {similarity:.2f})\n")
    print("  Categories:")
    for i, cat in enumerate(categories, start=1):
        marker = "→" if cat["id"] == suggested["id"] else " "
        print(f"    {marker} {i}. {cat['name']}")
    print()


def _ask_assignment(
    categories: list[dict],
    suggested: dict,
    is_first: bool,
) -> str:
    """Prompts the user to accept or override the suggested category.

    Args:
        categories: All available categories.
        suggested: The suggested category dictionary.
        is_first: Whether this is the first idea in the session.

    Returns:
        One of 'accept', a category index string, 'skip', or 'undo',
        or 'quit'.
    """
    n = len(categories)

    if is_first:
        prompt = f"Assign? [Enter=accept / 1-{n} / S=skip / Q=quit]: "
    else:
        prompt = f"Assign? [Enter=accept / 1-{n} / S=skip / U=undo / Q=quit]: "

    while True:
        raw = input(prompt).strip().lower()

        if raw == "":
            return "accept"

        if raw in ("s", "skip"):
            return "skip"

        if raw in ("q", "quit"):
            return "quit"
        
        if not is_first and raw in ("u", "undo"):
            return "undo"

        if raw.isdigit() and 1 <= int(raw) <= n:
            return raw

        options = f"Enter, 1-{n}, S, Q" + ("" if is_first else ", or U")
        print(f"  Please enter {options}.")


def _apply_assignment(
    idea: dict,
    choice: str,
    suggested: dict,
    categories: list[dict],
    stats: dict,
) -> dict:
    """Applies the user's category assignment to an idea.

    Args:
        idea: The idea dictionary being assigned.
        choice: The user's choice — 'accept' or a number string.
        suggested: The suggested category dictionary.
        categories: All available categories.
        stats: Session statistics dictionary to update.

    Returns:
        The category that was assigned.
    """
    if choice == "accept":
        assigned = suggested
    else:
        assigned = categories[int(choice) - 1]

    assign_idea_category(idea["id"], assigned["id"])

    if choice == "accept":
        print(f"  → Assigned to: {assigned['name']} (accepted suggestion)")
    else:
        print(f"  → Assigned to: {assigned['name']}")

    stats["assigned"] += 1
    stats.setdefault("category_counts", {})
    stats["category_counts"][assigned["name"]] = (
        stats["category_counts"].get(assigned["name"], 0) + 1
    )

    return assigned


def _undo_assignment(
    prev_idea: dict,
    prev_category: dict,
    subject: str,
    other_category: dict,
    stats: dict,
) -> None:
    """Reverses the previous category assignment.

    Reassigns the previous idea back to Other if it was newly
    categorized, or removes its category if it was uncategorized.

    Args:
        prev_idea: The previously assigned idea dictionary.
        prev_category: The category that was assigned.
        subject: The subject name.
        other_category: The Other category dictionary.
        stats: Session statistics dictionary to update.
    """
    # Reassign back to Other as the neutral undo state
    assign_idea_category(prev_idea["id"], other_category["id"])
    stats["assigned"] -= 1
    stats["category_counts"][prev_category["name"]] = (
        stats["category_counts"].get(prev_category["name"], 1) - 1
    )
    print(
        f"  → Undone. '{prev_idea['text'][:50]}' "
        f"returned to Other."
    )


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

def run(args) -> None:
    """Executes the categorize command.

    Loads ideas to process based on the --all flag, builds category
    embeddings for suggestion, and guides the user through an
    interactive assignment session with undo support.

    Args:
        args: Parsed argument namespace containing subject and all_ideas.
    """
    subject = args.subject
    all_ideas = args.all_ideas

    # Load categories
    categories = get_categories(subject)
    if not categories:
        print(
            f"\nNo categories found for: {subject}\n"
            f"Create categories first with:\n"
            f"  thegist add-category --subject \"{subject}\" "
            f"--name <name> --description <description>\n"
        )
        return

    # Ensure Other category exists
    other_category = get_category_by_name(subject, "Other")
    if not other_category:
        print(
            f"\nNo 'Other' category found for: {subject}\n"
            f"Create it with:\n"
            f"  thegist add-category --subject \"{subject}\" "
            f"--name \"Other\" "
            f"--description \"ideas that do not fit any named category\"\n"
        )
        return

    # Load ideas to process
    if all_ideas:
        ideas = get_all_categorized_ideas(subject)
        mode_label = "all ideas (including already categorized)"
    else:
        uncategorized = get_uncategorized_ideas(subject)
        other_ideas = get_other_ideas(subject, other_category["id"])
        ideas = uncategorized + other_ideas
        mode_label = "uncategorized and Other ideas"

    if not ideas:
        print(
            f"\nNo ideas to categorize for: {subject}\n"
            f"All ideas are already categorized. "
            f"Use --all to re-evaluate everything.\n"
        )
        return

    # Apply max limit if specified
    if args.max is not None and args.max > 0:
        ideas = ideas[:args.max]
        print(f"Session limited to {args.max} idea(s) by --max.\n")

    print(f"\ntheGist — Categorize")
    print(f"Subject    : {subject}")
    print(f"Mode       : {mode_label}")
    print(f"Ideas      : {len(ideas)}")
    if args.max is not None:
        print(f"Max        : {args.max} per session")
    print(f"Categories : {len(categories)}\n")

    model = _load_model()
    category_embeddings = _build_category_embeddings(model, categories)

    print(f"Ready to categorize {len(ideas)} idea(s).\n")

    stats = {
        "assigned": 0,
        "skipped": 0,
        "category_counts": {},
    }

    prev_idea = None
    prev_category = None

    i = 0
    while i < len(ideas):
        idea = ideas[i]
        is_first = prev_idea is None

        suggested, similarity = _suggest_category(
            idea["text"], model, category_embeddings
        )

        _display_idea(idea, i + 1, len(ideas), suggested, similarity, categories)
        choice = _ask_assignment(categories, suggested, is_first)

        if choice == "quit":
            remaining = len(ideas) - i
            print(
                f"\nSession ended early. "
                f"{stats['assigned']} idea(s) assigned this session. "
                f"{remaining} idea(s) remaining.\n"
                f"Run categorize again to continue.\n"
            )
            # Print partial summary before exiting
            if stats["category_counts"]:
                print("  Breakdown so far:")
                for cat_name, count in sorted(
                    stats["category_counts"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    print(f"    {cat_name} : {count}")
            print()
            return

        if choice == "undo":
            _undo_assignment(
                prev_idea, prev_category, subject, other_category, stats
            )
            i -= 1
            prev_idea = None
            prev_category = None
            continue

        if choice == "skip":
            print("  → Skipped.")
            stats["skipped"] += 1
            prev_idea = None
            prev_category = None
        else:
            assigned = _apply_assignment(
                idea, choice, suggested, categories, stats
            )
            prev_idea = idea
            prev_category = assigned

        i += 1

    print(f"\n{'-' * 60}")
    print(f"\nCategorize session complete.\n")
    print(f"  Assigned : {stats['assigned']} idea(s)")
    print(f"  Skipped  : {stats['skipped']} idea(s)\n")

    if stats["category_counts"]:
        print("  Breakdown:")
        for cat_name, count in sorted(
            stats["category_counts"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"    {cat_name} : {count}")
    print()