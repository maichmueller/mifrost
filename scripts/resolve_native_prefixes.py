"""Resolve native prefixes for planner Python packages during CMake setup.

The planner packages expose ``native_prefix()`` at runtime, but importing the
top-level package also loads its extension modules.  That is not reliable while
building an extension in an isolated wheel environment: the package's native
libraries may not be on the loader path until the wheel is repaired.  Package
locations are sufficient for CMake and can be discovered without importing the
extensions.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


def _package_dir(package_name: str) -> Path:
    """Return an installed package directory without requiring native imports."""
    try:
        module = importlib.import_module(package_name)
    except Exception:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            raise ModuleNotFoundError(package_name)
        if spec.submodule_search_locations:
            return Path(next(iter(spec.submodule_search_locations))).resolve()
        if spec.origin and spec.origin not in {"built-in", "frozen"}:
            return Path(spec.origin).resolve().parent
        raise ImportError(f"Could not locate package directory for {package_name}")
    else:
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise ImportError(f"Could not locate package directory for {package_name}")
        return Path(module_file).resolve().parent


def _native_prefix(package_name: str, marker: str) -> Path:
    package_dir = _package_dir(package_name)
    candidates = (package_dir / "native", package_dir)
    for candidate in candidates:
        if (candidate / "include" / marker).is_dir():
            return candidate
    raise FileNotFoundError(
        f"No native prefix for {package_name} under {package_dir}; "
        f"expected include/{marker}"
    )


def main() -> int:
    packages = (
        ("pytyr", "tyr"),
        ("pyyggdrasil", "yggdrasil"),
        ("pypddl", "loki"),
    )
    try:
        for package_name, marker in packages:
            print(_native_prefix(package_name, marker))
    except (ImportError, ModuleNotFoundError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
