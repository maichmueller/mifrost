from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_configure_defaults_to_local_release_build_dir(monkeypatch) -> None:
    module = _load_script_module("configure_defaults_test", "configure.py")
    build_paths = _load_script_module("build_modes_test", "local_build_dirs.py")
    calls: list[list[str]] = []

    monkeypatch.setattr(sys, "argv", ["configure.py", "--noconan"])
    monkeypatch.setattr(module, "find_mimir_prefix", lambda: None)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda cmd, *, check: calls.append(cmd),
    )

    module.main()

    assert calls
    assert "-B" in calls[0]
    assert calls[0][calls[0].index("-B") + 1] == str(
        (ROOT / build_paths.DEFAULT_LOCAL_BUILD_DIR).resolve()
    )


def test_configure_ci_mode_uses_ci_build_dir_and_release(monkeypatch) -> None:
    module = _load_script_module("configure_ci_mode_test", "configure.py")
    build_paths = _load_script_module("build_modes_test_ci", "local_build_dirs.py")
    calls: list[list[str]] = []

    monkeypatch.setattr(sys, "argv", ["configure.py", "--noconan", "--mode", "ci"])
    monkeypatch.setattr(module, "find_mimir_prefix", lambda: None)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda cmd, *, check: calls.append(cmd),
    )

    module.main()

    mode = build_paths.get_local_build_mode("ci")
    assert calls
    assert calls[0][calls[0].index("-B") + 1] == str((ROOT / mode.build_dir).resolve())
    assert f"-DCMAKE_BUILD_TYPE={mode.config}" in calls[0]


def test_configure_bench_mode_enables_benchmarks(monkeypatch) -> None:
    module = _load_script_module("configure_bench_mode_test", "configure.py")
    build_paths = _load_script_module("build_modes_test_bench", "local_build_dirs.py")
    calls: list[list[str]] = []

    monkeypatch.setattr(sys, "argv", ["configure.py", "--noconan", "--mode", "bench"])
    monkeypatch.setattr(module, "find_mimir_prefix", lambda: None)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda cmd, *, check: calls.append(cmd),
    )

    module.main()

    mode = build_paths.get_local_build_mode("bench")
    assert calls
    assert calls[0][calls[0].index("-B") + 1] == str((ROOT / mode.build_dir).resolve())
    assert "-DMIFROST_BUILD_BENCHMARKS=ON" in calls[0]


def test_cbuild_defaults_to_local_release_build_dir(monkeypatch) -> None:
    module = _load_script_module("cbuild_defaults_test", "cbuild.py")
    build_paths = _load_script_module("build_paths_test_cbuild", "local_build_dirs.py")
    calls: list[list[str]] = []

    monkeypatch.setattr(sys, "argv", ["cbuild.py"])
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda cmd, *, check: calls.append(cmd),
    )

    module.main()

    assert calls == [
        [
            "cmake",
            "--build",
            build_paths.DEFAULT_LOCAL_BUILD_MODE.build_dir,
            "--target",
            build_paths.DEFAULT_LOCAL_BUILD_MODE.default_target,
        ]
    ]


def test_cbuild_mode_uses_mode_build_dir_and_target(monkeypatch) -> None:
    module = _load_script_module("cbuild_mode_test", "cbuild.py")
    build_paths = _load_script_module(
        "build_paths_test_cbuild_mode", "local_build_dirs.py"
    )
    calls: list[list[str]] = []

    monkeypatch.setattr(sys, "argv", ["cbuild.py", "--mode", "stubs"])
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda cmd, *, check: calls.append(cmd),
    )

    module.main()

    mode = build_paths.get_local_build_mode("stubs")
    assert calls == [
        ["cmake", "--build", mode.build_dir, "--target", mode.default_target]
    ]
