"""Add category command for theGist.

This module implements the add-category command which allows users
to define named categories for organizing ideas within a subject.
Categories are created explicitly and each has a name and description.
The description is used by the categorize command to suggest which
category an idea belongs to based on semantic similarity.

Usage:
    thegist add-category --subject <subject> --name <name>
                         --description <description>
"""

from thegist.src.database import (
    get_categories,
    get_category_by_name,
    insert_category,
)


def register(subparsers) -> None:
    """Registers the add-category command with the top level parser.

    Args:
        subparsers: The subparsers group from the top level parser.
    """
    parser = subparsers.add_parser(
        "add-category",
        help="Create a new category for organizing ideas.",
        description=(
            "Creates a named category for a subject. Categories are "
            "used by the categorize command to organize ideas into "
            "meaningful groups. The description is used to suggest "
            "which category an idea belongs to based on semantic "
            "similarity so a clear and detailed description produces "
            "better suggestions."
        ),
    )
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="Subject name to create the category for.",
    )
    parser.add_argument(
        "--name",
        default=None,
        type=str,
        help="Display name for the category.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help=(
            "Description of what ideas belong in this category. "
            "A detailed description improves suggestion accuracy "
            "during categorize sessions."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="List all existing categories for the subject and exit.",
    )
    parser.set_defaults(func=run)


def run(args) -> None:
    """Executes the add-category command.

    If --list is specified prints all existing categories for the
    subject and exits. Otherwise creates a new category with the
    provided name and description.

    Args:
        args: Parsed argument namespace containing subject, name,
            description, and list.
    """
    subject = args.subject

    # List mode — show existing categories and exit
    if args.list:
        categories = get_categories(subject)
        if not categories:
            print(f"\nNo categories found for: {subject}")
            print("Use add-category to create your first category.\n")
            return

        print(f"\nCategories for: {subject}\n")
        print("-" * 60)
        for cat in categories:
            print(f"  {cat['name']}")
            print(f"  {cat['description']}")
            print(f"  Created: {cat['created_at'][:10]}")
            print()
        return

    name = args.name.strip()
    description = args.description.strip()

    if not name:
        print("\nError: --name cannot be empty.\n")
        return

    if not description:
        print("\nError: --description cannot be empty.\n")
        return

    # Check if category already exists
    existing = get_category_by_name(subject, name)
    if existing:
        print(
            f"\nCategory '{name}' already exists for: {subject}\n"
            f"Description: {existing['description']}\n"
        )
        return

    try:
        category = insert_category(subject, name, description)
        print(f"\nCategory created successfully.\n")
        print(f"  Subject     : {subject}")
        print(f"  Name        : {category['name']}")
        print(f"  Description : {category['description']}\n")

        # Show all current categories after creation
        categories = get_categories(subject)
        print(f"All categories for {subject} ({len(categories)} total):")
        for cat in categories:
            print(f"  - {cat['name']}")
        print()

    except ValueError as e:
        print(f"\nError: {e}\n")