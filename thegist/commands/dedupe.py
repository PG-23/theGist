"""Dedupe command for theGist.

This module implements the dedupe command which uses MinHash and
Locality Sensitive Hashing (LSH) to find semantically similar ideas
within a subject pool and guides the user through an interactive
session to resolve duplicates.

Original record ideas are never modified. Duplicates are resolved
by marking ideas inactive in the subject_ideas pool and recording
the decision in the duplicate_pairs table for future statistics.

Usage:
    thegist dedupe --subject <subject> [--threshold <float>]
"""

from datasketch import MinHash, MinHashLSH

from thegist.src.database import (
    deactivate_subject_idea,
    get_active_subject_ideas,
    record_duplicate_pair,
)

# Minimum number of active subject ideas required to run dedupe
MIN_IDEAS = 20

# Number of hash permutations for MinHash — higher is more accurate
# but slower. 128 is a good balance for this use case.
NUM_PERMUTATIONS = 128


def register(subparsers) -> None:
    """Registers the dedupe command with the top level argument parser.

    Args:
        subparsers: The subparsers group from the top level parser.
    """
    parser = subparsers.add_parser(
        "dedupe",
        help="Find and resolve duplicate ideas in a subject pool.",
        description=(
            "Uses MinHash and LSH to find similar ideas within a subject "
            "and guides you through resolving them interactively. "
            "Original record ideas are never modified."
        ),
    )
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="Subject name to deduplicate ideas for.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help=(
            "Similarity threshold between 0.0 and 1.0. "
            "Ideas with Jaccard similarity above this value are flagged "
            "as candidates. Lower values catch more pairs but may include "
            "false positives. Default is 0.4."
        ),
    )
    parser.set_defaults(func=run)


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Converts an idea string into a list of word tokens.

    Lowercases and splits on whitespace. MinHash operates on sets
    of tokens so tokenization quality directly affects similarity
    accuracy.

    Args:
        text: The idea text string to tokenize.

    Returns:
        A list of lowercase word strings.

    Example:
        >>> _tokenize("Cavalry archers have bonus range")
        ['cavalry', 'archers', 'have', 'bonus', 'range']
    """
    return text.lower().split()


def _build_minhash(tokens: list[str]) -> MinHash:
    """Builds a MinHash signature for a list of tokens.

    Args:
        tokens: A list of word token strings to hash.

    Returns:
        A MinHash object representing the token set signature.
    """
    mh = MinHash(num_perm=NUM_PERMUTATIONS)
    for token in tokens:
        mh.update(token.encode("utf-8"))
    return mh


def _find_candidates(
    ideas: list[dict],
    threshold: float,
) -> list[tuple[dict, dict, float]]:
    """Finds candidate duplicate pairs using MinHash and LSH.

    Builds a MinHash signature for each idea and inserts them into
    an LSH index. Queries each idea against the index to find
    similar pairs above the threshold. Each pair is returned once
    with its estimated Jaccard similarity score.

    Args:
        ideas: A list of active subject idea dictionaries.
        threshold: The minimum Jaccard similarity to flag a pair.

    Returns:
        A list of tuples where each tuple contains
        (idea_a, idea_b, similarity) for each candidate pair.
        Sorted by similarity descending so the most similar pairs
        are reviewed first.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERMUTATIONS)
    signatures = {}

    # Build signatures and insert into LSH index
    for idea in ideas:
        tokens = _tokenize(idea["text"])
        if not tokens:
            continue
        mh = _build_minhash(tokens)
        signatures[idea["id"]] = (mh, idea)
        lsh.insert(idea["id"], mh)

    # Query each idea to find its neighbors
    seen = set()
    candidates = []

    for idea_id, (mh, idea) in signatures.items():
        neighbors = lsh.query(mh)
        for neighbor_id in neighbors:
            if neighbor_id == idea_id:
                continue

            pair_key = tuple(sorted([idea_id, neighbor_id]))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            neighbor_mh, neighbor_idea = signatures[neighbor_id]
            similarity = mh.jaccard(neighbor_mh)

            candidates.append((idea, neighbor_idea, similarity))

    # Sort by similarity descending — review most similar pairs first
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def _ask_choice() -> str:
    """Prompts the user to choose how to resolve a duplicate pair.

    Accepts A, B, Both, or Skip. Enter defaults to Both since
    keeping is always safer than accidentally deleting.

    Returns:
        One of 'a', 'b', 'both', or 'skip'.
    """
    valid = {"a", "b", "both", "skip", ""}
    while True:
        raw = input("Keep? [A / B / Both / Skip] (default: Both): ").strip().lower()
        if raw in valid:
            return raw if raw else "both"
        print("  Please enter A, B, Both, or Skip.")


def _display_pair(
    idea_a: dict,
    idea_b: dict,
    similarity: float,
    index: int,
    total: int,
) -> None:
    """Prints a candidate duplicate pair for the user to review.

    Args:
        idea_a: The first idea dictionary in the pair.
        idea_b: The second idea dictionary in the pair.
        similarity: The Jaccard similarity score between the pair.
        index: The current pair number in the session.
        total: The total number of candidate pairs in the session.
    """
    print(f"\n{'-' * 60}")
    print(f"Pair {index} of {total}  (similarity: {similarity:.2f})\n")
    print(f"  A: {idea_a['text']}")
    print(f"     From: {idea_a['record_title']} | {idea_a['channel']}")
    print()
    print(f"  B: {idea_b['text']}")
    print(f"     From: {idea_b['record_title']} | {idea_b['channel']}")
    print()


def _resolve_pair(
    idea_a: dict,
    idea_b: dict,
    similarity: float,
    subject: str,
    choice: str,
    stats: dict,
) -> None:
    """Applies the user's decision to a duplicate pair.

    Updates the subject_ideas pool and records the decision in
    duplicate_pairs where applicable.

    Args:
        idea_a: The first idea dictionary.
        idea_b: The second idea dictionary.
        similarity: The Jaccard similarity score.
        subject: The subject the pair belongs to.
        choice: One of 'a', 'b', 'both', or 'skip'.
        stats: The session statistics dictionary to update.
    """
    if choice == "a":
        deactivate_subject_idea(idea_b["id"], subject)
        record_duplicate_pair(
            kept_idea_id=idea_a["id"],
            removed_idea_id=idea_b["id"],
            subject=subject,
            similarity=similarity,
        )
        print(f"  → Kept A, removed B.")
        stats["deleted"] += 1

    elif choice == "b":
        deactivate_subject_idea(idea_a["id"], subject)
        record_duplicate_pair(
            kept_idea_id=idea_b["id"],
            removed_idea_id=idea_a["id"],
            subject=subject,
            similarity=similarity,
        )
        print(f"  → Kept B, removed A.")
        stats["deleted"] += 1

    elif choice == "both":
        print(f"  → Kept both.")
        stats["kept_both"] += 1

    elif choice == "skip":
        print(f"  → Skipped.")
        stats["skipped"] += 1


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

def run(args) -> None:
    """Executes the dedupe command.

    Loads active subject ideas, finds candidate duplicate pairs
    using MinHash and LSH, and guides the user through an
    interactive resolution session.

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
            f"\nNot enough ideas to deduplicate. "
            f"Found {len(ideas)} but minimum is {MIN_IDEAS}.\n"
        )
        return

    print(f"\ntheGist — Dedupe")
    print(f"Subject   : {subject}")
    print(f"Ideas     : {len(ideas)}")
    print(f"Threshold : {threshold}\n")
    print("Finding candidate pairs...")

    candidates = _find_candidates(ideas, threshold)

    if not candidates:
        print(
            f"\nNo duplicate candidates found at threshold {threshold}. "
            f"Try lowering the threshold.\n"
        )
        return

    print(f"Found {len(candidates)} candidate pair(s).\n")

    stats = {
        "deleted": 0,
        "kept_both": 0,
        "skipped": 0,
    }

    # Track which ideas have already been deactivated this session
    # to avoid showing pairs where one idea was already removed
    deactivated = set()

    for i, (idea_a, idea_b, similarity) in enumerate(candidates, start=1):

        # Skip pairs where either idea was already removed this session
        if idea_a["id"] in deactivated or idea_b["id"] in deactivated:
            continue

        _display_pair(idea_a, idea_b, similarity, i, len(candidates))
        choice = _ask_choice()
        _resolve_pair(idea_a, idea_b, similarity, subject, choice, stats)

        if choice == "a":
            deactivated.add(idea_b["id"])
        elif choice == "b":
            deactivated.add(idea_a["id"])

    print(f"\n{'-' * 60}")
    print(f"\nDedupe session complete.\n")
    print(f"  Reviewed  : {len(candidates)} pair(s)")
    print(f"  Deleted   : {stats['deleted']} idea(s)")
    print(f"  Kept both : {stats['kept_both']} pair(s)")
    print(f"  Skipped   : {stats['skipped']} pair(s)")
    remaining = len(ideas) - stats["deleted"]
    print(f"  Active    : {remaining} idea(s) remaining in pool\n")