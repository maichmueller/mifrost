"""Validate the generated nanobind extension stub."""

from __future__ import annotations

import argparse
from pathlib import Path


def validate_stub(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Generated stub is missing: {path}")
    text = path.read_text()
    if not text.strip():
        raise SystemExit(f"Generated stub is empty: {path}")
    compile(text, str(path), "exec")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("src/mifrost/_core.pyi"),
        help="Path to the generated nanobind stub.",
    )
    args = parser.parse_args()
    validate_stub(args.path)


if __name__ == "__main__":
    main()
