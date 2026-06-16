from __future__ import annotations

STANDARD_LOCAL_BUILD_DIRS = (
    "build/local-release",
    "build/local-debug",
    "build/stubs",
    "build/ci",
)

DEFAULT_LOCAL_BUILD_DIR = STANDARD_LOCAL_BUILD_DIRS[0]


def local_build_dir_help_text() -> str:
    return (
        "Build directory; prefer "
        + ", ".join(STANDARD_LOCAL_BUILD_DIRS[:-1])
        + f", or {STANDARD_LOCAL_BUILD_DIRS[-1]} (see AGENTS.md)"
    )
