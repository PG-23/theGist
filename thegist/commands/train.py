"""Train command for theGist.

This module implements the train command which builds a supervised
classifier from labeled examples collected during filter-ideas
sessions. Currently supports training a relevance classifier that
predicts whether an idea is relevant or irrelevant to a subject.

Usage:
    thegist train --subject <subject> --type filter
"""

from thegist.src.classifier import get_model_path, train
from thegist.src.database import get_labeled_examples


def register(subparsers) -> None:
    """Registers the train command with the top level argument parser.

    Args:
        subparsers: The subparsers group from the top level parser.
    """
    parser = subparsers.add_parser(
        "train",
        help="Train a supervised classifier from labeled examples.",
        description=(
            "Builds a classifier from labeled examples collected "
            "during filter-ideas sessions. The trained model is saved "
            "to disk and used automatically by filter-ideas."
        ),
    )
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="Subject name to train the classifier for.",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["filter"],
        default="filter",
        help=(
            "Type of classifier to train. Currently supports 'filter' "
            "for relevance classification. Default is 'filter'."
        ),
    )
    parser.set_defaults(func=run)


def run(args) -> None:
    """Executes the train command.

    Loads labeled examples from the database, trains the specified
    classifier type, evaluates on a held out test set, and saves
    the model to disk.

    Args:
        args: Parsed argument namespace containing subject and type.
    """
    subject = args.subject
    classifier_type = args.type

    print(f"\ntheGist — Train Classifier")
    print(f"Subject : {subject}")
    print(f"Type    : {classifier_type}\n")

    # Load labeled examples
    examples = get_labeled_examples(subject)

    if not examples:
        print(
            f"No labeled examples found for: {subject}\n"
            f"Run filter-ideas first to collect labeled data.\n"
        )
        return

    relevant = sum(1 for e in examples if e["label"] == 1)
    irrelevant = sum(1 for e in examples if e["label"] == 0)

    print(f"Labeled examples found:")
    print(f"  Relevant   : {relevant}")
    print(f"  Irrelevant : {irrelevant}")
    print(f"  Total      : {len(examples)}\n")

    if irrelevant < 20:
        print(
            f"Warning: Only {irrelevant} irrelevant examples found. "
            f"At least 20 are recommended for meaningful training. "
            f"Consider running more filter-ideas sessions.\n"
        )

    try:
        metrics = train(examples, subject)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    model_path = get_model_path(subject)

    print(f"\nTraining complete.\n")
    print(f"  Train set  : {metrics['train_size']} examples")
    print(f"  Test set   : {metrics['test_size']} examples\n")
    print(f"  Accuracy   : {metrics['accuracy']}%")
    print(
        f"  Precision  : {metrics['precision']}%  "
        f"(of ideas flagged irrelevant, how many truly were)"
    )
    print(
        f"  Recall     : {metrics['recall']}%  "
        f"(of all irrelevant ideas, how many were caught)"
    )
    print(f"\nModel saved to: {model_path}\n")