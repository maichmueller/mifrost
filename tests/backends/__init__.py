"""Cross-backend semantic and integration test helpers."""

from pathlib import Path
import sys


def isolated_subprocess_package_parent(source_parent: Path) -> Path:
    """Select a complete package tree for ``python -S`` isolation checks."""
    neutral_core = sys.modules.get("mifrost._neutral_core")
    native_file = getattr(neutral_core, "__file__", None)
    if native_file is None:
        return source_parent

    installed_parent = Path(native_file).resolve().parent.parent
    if (installed_parent / "mifrost" / "__init__.py").is_file():
        return installed_parent
    return source_parent
