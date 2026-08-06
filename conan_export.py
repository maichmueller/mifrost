#!/usr/bin/env python3
"""Export local Conan dependency recipes declared in conandata.yml."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml


@dataclass(frozen=True)
class LocalRecipeExport:
    name: str
    version: str
    recipe_dir: Path


@dataclass(frozen=True)
class RecipeReference:
    """Minimal ``name/version`` view of a Conan recipe reference.

    CMake invokes this script with the interpreter that builds the project,
    which is not required to have the ``conan`` Python package importable --
    Conan itself is located separately as an executable via ``--conan_cmd``.
    Parsing the handful of reference strings in ``conandata.yml`` locally keeps
    the script runnable under a bare interpreter.
    """

    name: str
    version: str

    @classmethod
    def loads(cls, text: str) -> RecipeReference:
        reference = text.strip()
        if not reference:
            raise ValueError("empty Conan recipe reference")
        # Drop the optional revision (``#rrev``) and user/channel (``@u/c``)
        # parts; only name and version select a local recipe directory.
        reference = reference.split("#", 1)[0]
        reference = reference.split("@", 1)[0]
        name, separator, version = reference.partition("/")
        if not separator or not name or not version:
            raise ValueError(f"malformed Conan recipe reference: {text!r}")
        return cls(name=name, version=version)


def _load_conandata(conandata_path: Path) -> dict[str, object]:
    if not conandata_path.is_file():
        raise FileNotFoundError(f"conandata.yml not found at {conandata_path}")

    data = yaml.safe_load(conandata_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("conandata.yml must contain a mapping")
    return data


def _read_requirements(conandata_path: Path) -> list[str]:
    data = _load_conandata(conandata_path)
    requirements = data.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("conandata.yml must define a non-empty 'requirements' list")
    if not all(isinstance(requirement, str) for requirement in requirements):
        raise ValueError("conandata.yml requirements entries must be strings")
    return requirements


def local_recipe_exports(
    conandata_path: Path,
    dependencies_dir: Path,
) -> list[LocalRecipeExport]:
    exports: list[LocalRecipeExport] = []
    for requirement in _read_requirements(conandata_path):
        reference = RecipeReference.loads(requirement)
        recipe_dir = dependencies_dir / reference.name
        if not (recipe_dir / "conanfile.py").is_file():
            continue
        exports.append(
            LocalRecipeExport(
                name=reference.name,
                version=str(reference.version),
                recipe_dir=recipe_dir,
            )
        )
    return exports


def export_local_recipes(
    exports: Sequence[LocalRecipeExport],
    conan_cmd: str,
) -> bool:
    had_errors = False
    for export in exports:
        try:
            subprocess.run(
                [
                    conan_cmd,
                    "export",
                    str(export.recipe_dir),
                    f"--version={export.version}",
                ],
                check=True,
            )
            print(f"Successfully exported {export.name} version {export.version}.")
        except subprocess.CalledProcessError as e:
            print(f"Error exporting {export.name} version {export.version}: {e}")
            had_errors = True
    return not had_errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export dependencies using Conan.")
    parser.add_argument(
        "--conan_cmd",
        type=str,
        default="conan",
        help="Command to invoke Conan (default: 'conan').",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    try:
        exports = local_recipe_exports(
            repo_root / "conandata.yml",
            repo_root / "dependencies",
        )
    except Exception as e:
        print(f"Error reading conandata.yml: {e}")
        return 1

    if export_local_recipes(exports, args.conan_cmd):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
