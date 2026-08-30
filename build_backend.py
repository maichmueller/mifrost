from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import scikit_build_core.build as _sbc

_CONAN_PREPARED: set[tuple[str, str]] = set()
_BACKEND_BUILD_REQUIREMENTS = {
    "pymimir": "pymimir>=0.14.3",
    "pytyr": "pytyr==0.0.34",
}


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
        entries = current.split(os.pathsep)
        if prefix in entries:
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


def _set_cmake_arg(key: str, value: str) -> None:
    """Force a -Dkey=value entry in CMAKE_ARGS, overriding any prior value.

    GitHub Actions images and local shells occasionally carry a stale CMAKE_ARGS
    value across steps. For CI determinism, treat env vars owned by this backend
    as authoritative.
    """

    existing = os.environ.get("CMAKE_ARGS", "").strip()
    if not existing:
        os.environ["CMAKE_ARGS"] = f"-D{key}={value}"
        return

    tokens = shlex.split(existing)
    prefix_eq = f"-D{key}="
    prefix_typed = f"-D{key}:"
    tokens = [
        t for t in tokens if not (t.startswith(prefix_eq) or t.startswith(prefix_typed))
    ]
    tokens.insert(0, f"-D{key}={value}")
    os.environ["CMAKE_ARGS"] = shlex.join(tokens)


def _set_default_rpath_mode(mode: str) -> None:
    os.environ.setdefault("MIFROST_RPATH_MODE", mode)
    if sys.platform == "darwin" and mode == "wheel":
        os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "11.0")


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _cmake_definition(name: str) -> str | None:
    """Return the last value assigned to a definition in ``CMAKE_ARGS``."""
    value: str | None = None
    prefix = f"-D{name}"
    for token in shlex.split(os.environ.get("CMAKE_ARGS", "")):
        if not token.startswith(prefix):
            continue
        suffix = token[len(prefix) :]
        if suffix.startswith(":"):
            _, separator, candidate = suffix.partition("=")
        elif suffix.startswith("="):
            separator, candidate = "=", suffix[1:]
        else:
            continue
        if separator:
            value = candidate
    return value


def _parse_backend_selection(value: str) -> frozenset[str]:
    normalized = value.strip().lower()
    if normalized in {"", "none", "core", "neutral"}:
        return frozenset()
    if normalized in {"all", "both"}:
        return frozenset(_BACKEND_BUILD_REQUIREMENTS)

    requested = {
        item.strip() for item in normalized.replace("+", ",").split(",") if item.strip()
    }
    unknown = requested.difference(_BACKEND_BUILD_REQUIREMENTS)
    if unknown:
        choices = "core, pymimir, pytyr, or both"
        raise RuntimeError(
            "Unsupported MIFROST_BUILD_BACKENDS value "
            f"{value!r}: unknown backend(s) {sorted(unknown)!r}. Use {choices}."
        )
    return frozenset(requested)


def _selected_backends() -> frozenset[str]:
    """Resolve adapter selection without process-global runtime state.

    Package builds contain both adapters by default. Explicit CMake definitions
    remain authoritative for existing build invocations, while
    ``MIFROST_BUILD_BACKENDS`` provides the concise packaging-facing control.
    """
    selection = os.environ.get("MIFROST_BUILD_BACKENDS")
    if selection is not None:
        return _parse_backend_selection(selection)

    pymimir_value = _cmake_definition("MIFROST_BUILD_PYMIMIR_ADAPTER")
    pytyr_value = _cmake_definition("MIFROST_BUILD_PYTYR_ADAPTER")
    if pymimir_value is not None or pytyr_value is not None:
        enabled: set[str] = set()
        if _as_bool(pymimir_value, default=True):
            enabled.add("pymimir")
        if _as_bool(pytyr_value, default=False):
            enabled.add("pytyr")
        return frozenset(enabled)

    return frozenset(_BACKEND_BUILD_REQUIREMENTS)


def _with_backend_build_requirements(requirements: list[str]) -> list[str]:
    result = list(requirements)
    normalized = {requirement.lower() for requirement in result}
    for backend in sorted(_selected_backends()):
        requirement = _BACKEND_BUILD_REQUIREMENTS[backend]
        if requirement.lower() not in normalized:
            result.append(requirement)
            normalized.add(requirement.lower())
    return result


def _exclude_unbuilt_adapter_stubs(backends: frozenset[str]) -> None:
    excluded = {
        item.strip()
        for item in os.environ.get("SKBUILD_WHEEL_EXCLUDE", "").split(";")
        if item.strip()
    }
    if "pymimir" not in backends:
        excluded.update({"mifrost/_core.pyi", "mifrost/_pymimir_adapter.pyi"})
    if "pytyr" not in backends:
        excluded.add("mifrost/_pytyr_adapter.pyi")
    if excluded:
        os.environ["SKBUILD_WHEEL_EXCLUDE"] = ";".join(sorted(excluded))


def _required_wheel_stubs() -> tuple[Path, ...]:
    package_dir = Path(__file__).resolve().parent / "src" / "mifrost"
    names = ["_neutral_core.pyi"]
    backends = _selected_backends()
    if "pymimir" in backends:
        names.extend(("_core.pyi", "_pymimir_adapter.pyi"))
    if "pytyr" in backends:
        names.append("_pytyr_adapter.pyi")
    return tuple(package_dir / name for name in names)


def _in_cibuildwheel_repair_pipeline() -> bool:
    # cibuildwheel sets CIBUILDWHEEL=1 for the whole build it drives, including
    # the "pip install --no-build-isolation" invocation that reaches this hook.
    # That is the only context where generating stubs during build_wheel() is
    # unsafe: repair (auditwheel/delocate) relocates the extension's planner
    # shared libraries after this hook returns, so importing the extension to
    # introspect it here -- which stub generation requires -- would see
    # libraries that are not yet resolvable. A plain "pip install ." (local
    # dev, editable installs, tests.yml CI) never goes through repair, so it
    # keeps the default nanobind/CMake behavior: stubs are generated inline,
    # like any other build output, with no separate manual step.
    return _as_bool(os.environ.get("CIBUILDWHEEL"), default=False)


def _generate_wheel_stubs() -> None:
    """Generate nanobind stubs via a clean dev-mode CMake build.

    Mirrors the CI ``generate_stubs`` job so a bare local ``cibuildwheel``
    invocation is self-contained. The stub build lives in its own
    ``build/stubs`` tree, which defaults to the CMake cache's "dev" RPATH
    mode, so the freshly compiled extension resolves its planner libraries
    directly -- unlike the "wheel"-mode extension this same process is
    building, which only resolves them after repair (see
    ``_in_cibuildwheel_repair_pipeline``).
    """
    repo_root = Path(__file__).resolve().parent
    backends = "+".join(sorted(_selected_backends())) or "core"
    subprocess.check_call(
        [
            sys.executable,
            str(repo_root / "configure.py"),
            "--mode",
            "stubs",
            "--backends",
            backends,
        ],
        cwd=str(repo_root),
    )
    subprocess.check_call(
        [sys.executable, str(repo_root / "cbuild.py"), "--mode", "stubs"],
        cwd=str(repo_root),
    )


def _ensure_wheel_stubs_generated() -> None:
    missing = [path for path in _required_wheel_stubs() if not path.is_file()]
    if not missing:
        return
    _generate_wheel_stubs()
    missing = [path for path in _required_wheel_stubs() if not path.is_file()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise RuntimeError(
            "cibuildwheel builds require nanobind stubs, and automatic "
            f"generation did not produce: {formatted}. Run `python configure.py "
            "--mode stubs --backends both` and `python cbuild.py --mode stubs` "
            "manually to see the underlying error, or build from the release "
            "sdist."
        )


def _prepare_common_cmake_env() -> None:
    rpath_mode = os.environ.get("MIFROST_RPATH_MODE", "dev")
    _set_cmake_arg("MIFROST_RPATH_MODE", rpath_mode)
    backends = _selected_backends()
    _exclude_unbuilt_adapter_stubs(backends)
    _set_cmake_arg(
        "MIFROST_BUILD_PYMIMIR_ADAPTER", "ON" if "pymimir" in backends else "OFF"
    )
    _set_cmake_arg(
        "MIFROST_BUILD_PYTYR_ADAPTER", "ON" if "pytyr" in backends else "OFF"
    )
    if "pymimir" in backends:
        mimir_prefix = _get_mimir_prefix()
        if mimir_prefix:
            _set_env_prefix_path(mimir_prefix)


def _get_conan_mode() -> str:
    return os.environ.get("MIFROST_CONAN_MODE", "provider").strip().lower()


def _maybe_prepare_conan(config_settings: dict[str, Any] | None) -> None:
    _prepare_common_cmake_env()
    mode = _get_conan_mode()
    if mode == "provider":
        return
    if mode == "toolchain":
        _prepare_conan(config_settings)
        return
    raise RuntimeError(
        "Unsupported MIFROST_CONAN_MODE. Use 'provider' (default) or 'toolchain'."
    )


def _prepare_conan(config_settings: dict[str, Any] | None) -> None:
    if os.environ.get("MIFROST_SKIP_CONAN_PREP") == "1":
        return

    build_type = _get_build_type(config_settings)
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
    return _with_backend_build_requirements(
        list(_sbc.get_requires_for_build_wheel(config_settings))
    )


def get_requires_for_build_editable(config_settings: dict[str, Any] | None = None):
    return _with_backend_build_requirements(
        list(_sbc.get_requires_for_build_editable(config_settings))
    )


def prepare_metadata_for_build_wheel(
    metadata_directory: str, config_settings: dict[str, Any] | None = None
):
    _set_default_rpath_mode("wheel")
    _set_cmake_arg("BUILD_TESTING", "OFF")
    _set_cmake_arg("MIFROST_BUILD_BENCHMARKS", "OFF")
    # Metadata prep should stay cheap and avoid any stub-related build work.
    # Actual wheel generation controls whether stubs are produced.
    _set_cmake_arg("MIFROST_GENERATE_STUBS", "OFF")
    _maybe_prepare_conan(config_settings)
    return _sbc.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(
    metadata_directory: str, config_settings: dict[str, Any] | None = None
):
    _set_default_rpath_mode("dev")
    _maybe_prepare_conan(config_settings)
    return _sbc.prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
):
    _set_default_rpath_mode("wheel")
    _set_cmake_arg("BUILD_TESTING", "OFF")
    _set_cmake_arg("MIFROST_BUILD_BENCHMARKS", "OFF")
    if _in_cibuildwheel_repair_pipeline():
        # CI's generate_stubs job pre-stages nanobind stubs as a shared
        # artifact so every matrix leg skips regenerating them. A bare local
        # cibuildwheel run has no such job, so this call self-heals by
        # running the same stub build (_generate_wheel_stubs) the first time
        # it finds stubs missing.
        _ensure_wheel_stubs_generated()
        _set_cmake_arg("MIFROST_GENERATE_STUBS", "OFF")
    else:
        _set_cmake_arg("MIFROST_GENERATE_STUBS", "ON")
    _maybe_prepare_conan(config_settings)
    return _sbc.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
):
    _set_default_rpath_mode("dev")
    _maybe_prepare_conan(config_settings)
    return _sbc.build_editable(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory: str, config_settings: dict[str, Any] | None = None):
    return _sbc.build_sdist(sdist_directory, config_settings)
