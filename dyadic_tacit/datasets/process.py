"""Unified command line entry point for the three dataset adapters."""

from __future__ import annotations

import argparse

from . import process_coblock, process_hanabi, process_overcooked


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-aligned DTU dataset records")
    parser.add_argument("dataset", choices=("overcooked", "hanabi", "coblock"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-manifest")
    parser.add_argument("--usage-map")
    args = parser.parse_args()
    processors = {"overcooked": process_overcooked, "hanabi": process_hanabi, "coblock": process_coblock}
    print(processors[args.dataset](args.input, args.output, args.split_manifest, args.usage_map))


if __name__ == "__main__":
    main()
