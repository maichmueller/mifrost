from __future__ import annotations

import os
import shlex
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Any

import scikit_build_core.build as _sbc

_CONAN_PREPARED: set[tuple[str, str]] = set()


def _first_setting(config_settings: dict[str, Any] | None, key: str) -> str | None:
    if not config_settings:
        return None
    value = config_settings.get(key)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None
    return str(value)


def _get_build_type(config_settings: dict[str, Any] | None) -> str:
    return (
        _first_setting(config_settings, "cmake.build-type")
        or os.environ.get("MIFROST_BUILD_TYPE")
        or os.environ.get("CMAKE_BUILD_TYPE")
        or "Release"
    )


def _get_conan_cmd() -> str:
    return os.environ.get("CONAN_COMMAND") or os.environ.get("CONAN_CMD") or "conan"


def _get_mimir_prefix() -> str | None:
    env_value = os.environ.get("MIFROST_MIMIR_CMAKE_DIR") or os.environ.get(
        "MIMIR_CMAKE_DIR"
    )
    if env_value:
        return env_value
    try:
        import pymimir

        if hasattr(pymimir, "get_cmake_dir"):
            return str(pymimir.get_cmake_dir())
    except Exception:
        pass
    return None


def _set_env_prefix_path(prefix: str) -> None:
    current = os.environ.get("CMAKE_PREFIX_PATH")
    if current:
        parts = current.split(os.pathsep)
        if prefix in parts:
            return
        os.environ["CMAKE_PREFIX_PATH"] = f"{prefix}{os.pathsep}{current}"
    else:
        os.environ["CMAKE_PREFIX_PATH"] = prefix


def _append_cmake_args(arg: str) -> None:
    existing = os.environ.get("CMAKE_ARGS", "").strip()
    if existing:
        os.environ["CMAKE_ARGS"] = f"{arg} {existing}"
    else:
        os.environ["CMAKE_ARGS"] = arg


def _ensure_cmake_arg(key: str, value: str) -> None:
    existing = os.environ.get("CMAKE_ARGS", "")
    if f"-D{key}=" in existing:
        return
    _append_cmake_args(f"-D{key}={value}")


def _set_default_rpath_mode(mode: str) -> None:
    os.environ.setdefault("MIFROST_RPATH_MODE", mode)


def _prepare_conan(config_settings: dict[str, Any] | None) -> None:
    if os.environ.get("MIFROST_SKIP_CONAN_PREP") == "1":
        return

    build_type = _get_build_type(config_settings)
    rpath_mode = os.environ.get("MIFROST_RPATH_MODE", "dev")
    _ensure_cmake_arg("MIFROST_RPATH_MODE", rpath_mode)
    repo_root = Path(__file__).resolve().parent
    toolchain_override = os.environ.get("MIFROST_CONAN_TOOLCHAIN")
    if toolchain_override:
        toolchain_path = Path(toolchain_override)
        if toolchain_path.exists():
            if "CMAKE_TOOLCHAIN_FILE" not in os.environ.get("CMAKE_ARGS", ""):
                _append_cmake_args(f"-DCMAKE_TOOLCHAIN_FILE={toolchain_path}")
            _set_env_prefix_path(str(toolchain_path.parent))
            mimir_prefix = _get_mimir_prefix()
            if mimir_prefix:
                _set_env_prefix_path(mimir_prefix)
            return

    default_toolchain = (
        repo_root
        / "build"
        / "conan"
        / "build"
        / build_type
        / "generators"
        / "conan_toolchain.cmake"
    )
    if default_toolchain.exists():
        if "CMAKE_TOOLCHAIN_FILE" not in os.environ.get("CMAKE_ARGS", ""):
            _append_cmake_args(f"-DCMAKE_TOOLCHAIN_FILE={default_toolchain}")
        _set_env_prefix_path(str(default_toolchain.parent))
        mimir_prefix = _get_mimir_prefix()
        if mimir_prefix:
            _set_env_prefix_path(mimir_prefix)
        return

    build_root = Path(
        os.environ.get("MIFROST_CONAN_BUILD_DIR", "build/conan_prep")
    ).resolve()
    if "CONAN_HOME" not in os.environ:
        os.environ["CONAN_HOME"] = str(build_root / "conan_home")
    cache_key = (build_type, str(build_root))
    if cache_key in _CONAN_PREPARED:
        return

    conan_cmd = _get_conan_cmd()
    conan_home = Path(os.environ["CONAN_HOME"])
    default_profile = conan_home / "profiles" / "default"
    if not default_profile.exists():
        subprocess.check_call([conan_cmd, "profile", "detect", "--force"])
    configure_cmd = [
        sys.executable,
        str(repo_root / "configure.py"),
        "--only_install",
        "--build_dir",
        str(build_root),
        "--source_dir",
        str(repo_root),
        "--config",
        build_type,
        "--conan_cmd",
        conan_cmd,
    ]

    extra_args = os.environ.get("CONAN_EXTRA_ARGS")
    if extra_args:
        configure_cmd.extend(shlex.split(extra_args))

    subprocess.check_call(configure_cmd, cwd=str(repo_root))

    toolchain = (
        build_root
        / "conan"
        / "build"
        / build_type
        / "generators"
        / "conan_toolchain.cmake"
    )
    if not toolchain.exists():
        fallback = build_root / "conan_toolchain.cmake"
        if fallback.exists():
            toolchain = fallback
        else:
            raise FileNotFoundError(f"Conan toolchain not found at {toolchain}")

    if "CMAKE_TOOLCHAIN_FILE" not in os.environ.get("CMAKE_ARGS", ""):
        _append_cmake_args(f"-DCMAKE_TOOLCHAIN_FILE={toolchain}")

    mimir_prefix = _get_mimir_prefix()
    if mimir_prefix:
        _set_env_prefix_path(mimir_prefix)

    _CONAN_PREPARED.add(cache_key)


def get_requires_for_build_wheel(config_settings: dict[str, Any] | None = None):
    return _sbc.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_editable(config_settings: dict[str, Any] | None = None):
    return _sbc.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_wheel(
    metadata_directory: str, config_settings: dict[str, Any] | None = None
):
    _set_default_rpath_mode("wheel")
    _prepare_conan(config_settings)
    return _sbc.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(
    metadata_directory: str, config_settings: dict[str, Any] | None = None
):
    _set_default_rpath_mode("dev")
    _prepare_conan(config_settings)
    return _sbc.prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
):
    _set_default_rpath_mode("wheel")
    _prepare_conan(config_settings)
    wheel_path = _sbc.build_wheel(wheel_directory, config_settings, metadata_directory)
    if sys.platform == "darwin" and os.environ.get("MIFROST_SKIP_DELOCATE") != "1":
        delocate = shutil.which("delocate-wheel")
        if not delocate:
            raise RuntimeError(
                "delocate-wheel not found. Install it with: pip install delocate "
                "or set MIFROST_SKIP_DELOCATE=1 to skip."
            )
        subprocess.check_call([delocate, "-w", wheel_directory, wheel_path])
    return wheel_path


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
):
    _set_default_rpath_mode("dev")
    _prepare_conan(config_settings)
    return _sbc.build_editable(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory: str, config_settings: dict[str, Any] | None = None):
    return _sbc.build_sdist(sdist_directory, config_settings)
