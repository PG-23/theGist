"""Command line entry point for theGist.

This module defines the top level CLI parser and registers all
available subcommands. Each subcommand is implemented in its own
module under the commands/ directory.

Usage:
    thegist <command> [options]
"""

import argparse

from thegist.commands import fetch
from thegist.commands import add_ideas
from thegist.commands import dedupe
from thegist.commands import filter_ideas


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="thegist",
        description=(
            "theGist — fetch and manage video transcripts "
            "for knowledge extraction and learning."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="<command>",
        required=True,
    )

    fetch.register(subparsers)
    add_ideas.register(subparsers)
    dedupe.register(subparsers)
    filter_ideas.register(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()