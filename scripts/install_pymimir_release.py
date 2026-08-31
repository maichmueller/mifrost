"""Install an ABI-matched Pymimir wheel from the maintained Mimir release.

Mifrost's Pymimir adapter and Pymimir exchange nanobind objects through a
shared type registry. The Pymimir wheels on the maintained
``maichmueller/mimir`` release are therefore installed explicitly instead of
asking PyPI to resolve a possibly older public package. The version is pinned
to the release selected here, while the project metadata documents the
compatible minimum version.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

DEFAULT_REPOSITORY = "maichmueller/mimir"
DEFAULT_VERSION = "0.15.0"
MINIMUM_VERSION = "0.15.0"


def _normalize_version(version: str) -> str:
    normalized = version.strip().removeprefix("v")
    if not re.fullmatch(r"\d+\.\d+\.\d+", normalized):
        raise ValueError(
            f"Pymimir release version must be a stable x.y.z version, got {version!r}"
        )
    return normalized


def _version_key(version: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in _normalize_version(version).split("."))  # type: ignore[return-value]


def _python_tag(
    major: int | None = None,
    minor: int | None = None,
    free_threaded: bool | None = None,
) -> str:
    if major is None:
        major = sys.version_info.major
    if minor is None:
        minor = sys.version_info.minor
    if free_threaded is None:
        gil_probe = getattr(sys, "_is_gil_enabled", None)
        free_threaded = callable(gil_probe) and not gil_probe()
    suffix = "t" if free_threaded else ""
    return f"cp{major}{minor}{suffix}"


def _platform_tag(system: str | None = None, machine: str | None = None) -> str:
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower().replace("-", "_")

    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "macosx_11_0_arm64"
    if system == "linux" and machine in {"x86_64", "amd64"}:
        return "manylinux_2_27_x86_64.manylinux_2_28_x86_64"

    raise RuntimeError(
        "No published pymimir release wheel for "
        f"{platform.system() if system == 'darwin' else system}/{machine}; "
        "the maintained release currently provides macOS arm64 and Linux x86_64 wheels."
    )


def wheel_filename(
    version: str,
    *,
    python_tag: str | None = None,
    system: str | None = None,
    machine: str | None = None,
    free_threaded: bool | None = None,
) -> str:
    version = _normalize_version(version)
    python_tag = python_tag or _python_tag(free_threaded=free_threaded)
    platform_tag = _platform_tag(system, machine)
    return f"pymimir-{version}-{python_tag}-{python_tag}-{platform_tag}.whl"


def release_wheel_url(
    version: str,
    *,
    repository: str = DEFAULT_REPOSITORY,
    python_tag: str | None = None,
    system: str | None = None,
    machine: str | None = None,
    free_threaded: bool | None = None,
) -> str:
    version = _normalize_version(version)
    repository = repository.strip().strip("/")
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", repository):
        raise ValueError(f"Repository must be in OWNER/NAME form, got {repository!r}")
    filename = wheel_filename(
        version,
        python_tag=python_tag,
        system=system,
        machine=machine,
        free_threaded=free_threaded,
    )
    return f"https://github.com/{repository}/releases/download/v{version}/{filename}"


def _selected_release(
    version: str | None = None, repository: str | None = None
) -> tuple[str, str]:
    selected_version = _normalize_version(
        version or os.environ.get("MIFROST_PYMIMIR_RELEASE_VERSION", DEFAULT_VERSION)
    )
    if _version_key(selected_version) < _version_key(MINIMUM_VERSION):
        raise ValueError(
            f"Pymimir {selected_version} is below the supported minimum "
            f"{MINIMUM_VERSION}"
        )
    selected_repository = (
        repository
        or os.environ.get("MIFROST_PYMIMIR_RELEASE_REPOSITORY", DEFAULT_REPOSITORY)
    ).strip()
    return selected_version, selected_repository


def install_release(
    version: str | None = None,
    repository: str | None = None,
    *,
    runner: Callable[[Sequence[str]], int] = subprocess.check_call,
) -> str:
    """Install and verify the selected release wheel, returning its URL."""
    version, repository = _selected_release(version, repository)
    url = release_wheel_url(version, repository=repository)
    print(f"Installing pymimir {version} from {url}")
    runner(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            url,
        ]
    )

    installed_version = importlib.metadata.version("pymimir")
    if installed_version != version:
        raise RuntimeError(
            f"Expected pymimir {version} after installation, found {installed_version}"
        )
    try:
        import pymimir

        cmake_dir = Path(pymimir.get_cmake_dir())
    except Exception as exc:
        raise RuntimeError(
            "The selected pymimir release installed, but its CMake package "
            "could not be loaded"
        ) from exc
    if not cmake_dir.is_dir():
        raise RuntimeError(f"pymimir.get_cmake_dir() is not a directory: {cmake_dir}")
    print(f"Verified pymimir {installed_version}; CMake package: {cmake_dir}")
    return url


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install the ABI-matched Pymimir wheel from a Mimir release."
    )
    parser.add_argument(
        "--version", help=f"release version (default: {DEFAULT_VERSION})"
    )
    parser.add_argument(
        "--repository",
        help=f"GitHub OWNER/NAME (default: {DEFAULT_REPOSITORY})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the selected wheel URL without installing it",
    )
    args = parser.parse_args(argv)

    try:
        version, repository = _selected_release(args.version, args.repository)
        url = release_wheel_url(version, repository=repository)
        if args.dry_run:
            print(url)
        else:
            install_release(version, repository)
    except (RuntimeError, ValueError, importlib.metadata.PackageNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
