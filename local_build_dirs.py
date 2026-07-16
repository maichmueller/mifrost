from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalBuildMode:
    name: str
    build_dir: str
    config: str = "Release"
    default_target: str = "all"
    with_benchmarks: bool = False
    description: str = ""


LOCAL_BUILD_MODES = (
    LocalBuildMode(
        name="local-release",
        build_dir="build/local-release",
        description="normal local development build",
    ),
    LocalBuildMode(
        name="local-debug",
        build_dir="build/local-debug",
        config="Debug",
        description="debug local development build",
    ),
    LocalBuildMode(
        name="stubs",
        build_dir="build/stubs",
        default_target="mifrost_module_stubs",
        description="generated stub build",
    ),
    LocalBuildMode(
        name="ci",
        build_dir="build/ci",
        default_target="mifrost_tests",
        description="CI-like local test build",
    ),
    LocalBuildMode(
        name="bench",
        build_dir="build/bench-release",
        default_target="mifrost_bench_hgraph",
        with_benchmarks=True,
        description="benchmark-enabled local build",
    ),
)

LOCAL_BUILD_MODE_BY_NAME = {mode.name: mode for mode in LOCAL_BUILD_MODES}
STANDARD_LOCAL_BUILD_DIRS = tuple(mode.build_dir for mode in LOCAL_BUILD_MODES)
DEFAULT_LOCAL_BUILD_MODE = LOCAL_BUILD_MODES[0]
DEFAULT_LOCAL_BUILD_DIR = DEFAULT_LOCAL_BUILD_MODE.build_dir


def local_build_mode_names() -> tuple[str, ...]:
    return tuple(mode.name for mode in LOCAL_BUILD_MODES)


def get_local_build_mode(name: str) -> LocalBuildMode:
    try:
        return LOCAL_BUILD_MODE_BY_NAME[name]
    except KeyError as exc:
        choices = ", ".join(local_build_mode_names())
        raise ValueError(
            f"Unknown build mode {name!r}; choose one of: {choices}"
        ) from exc


def local_build_mode_help_text() -> str:
    return (
        "Build mode; choose one of "
        + ", ".join(local_build_mode_names())
        + " (see docs/development/build-and-test.md)"
    )


def local_build_dir_help_text() -> str:
    return (
        "Build directory; prefer "
        + ", ".join(STANDARD_LOCAL_BUILD_DIRS[:-1])
        + f", or {STANDARD_LOCAL_BUILD_DIRS[-1]} (see docs/development/build-and-test.md)"
    )
