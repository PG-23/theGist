"""Command line entry point for theGist.

This module defines the top level CLI parser and registers all
available subcommands. Each subcommand is implemented in its own
module under the commands/ directory.

Usage:
    thegist <command> [options]
"""

import argparse

from thegist.commands import fetch


def main() -> None:
    """Parses command line arguments and dispatches to the correct command.

    Creates the top level argument parser, registers all subcommands,
    and calls the handler function attached to the parsed subcommand.
    """
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()