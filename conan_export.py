#!/usr/bin/python3

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Parse arguments
parser = argparse.ArgumentParser(description="Export dependencies using Conan.")
parser.add_argument(
    "--conan_cmd",
    type=str,
    default="conan",
    help="Command to invoke Conan (default: 'conan').",
)
args = parser.parse_args()

# Conan command
conan_cmd = args.conan_cmd


# Read versions from conandata.yml
def read_versions_from_conandata(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"conandata.yml not found at {file_path}")

    dependencies = {}
    in_requirements = False
    for line in file_path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "requirements:":
            in_requirements = True
            continue
        if not in_requirements:
            continue
        if not stripped:
            continue
        match = re.match(r'-\s*["\']?([^/"\']+)/([^"\']+)["\']?\s*$', stripped)
        if match is None:
            raise ValueError(f"Malformed requirement line in conandata.yml: {line}")
        dependencies[match.group(1).strip()] = match.group(2).strip()
    return dependencies


repo_root = Path(__file__).resolve().parent
conandata_path = repo_root / "conandata.yml"


try:
    dependency_versions = read_versions_from_conandata(conandata_path)
except Exception as e:
    print(f"Error reading conandata.yml: {e}")
    exit(1)


# export dependencies
had_errors = False
for dep, version in dependency_versions.items():
    recipe_dir = repo_root / "dependencies" / dep
    if not (recipe_dir / "conanfile.py").is_file():
        continue

    try:
        subprocess.run(
            [conan_cmd, "export", str(recipe_dir), f"--version={version}"],
            check=True,
        )
        print(f"Successfully exported {dep} version {version}.")
    except subprocess.CalledProcessError as e:
        print(f"Error exporting {dep} version {version}: {e}")
        had_errors = True

if had_errors:
    sys.exit(1)
