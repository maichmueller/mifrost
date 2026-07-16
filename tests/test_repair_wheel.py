from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_repair_module():
    path = ROOT / "scripts" / "repair_wheel.py"
    spec = importlib.util.spec_from_file_location("mifrost_repair_wheel", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_macos_repair_excludes_generic_python_runtime_libraries() -> None:
    module = _load_repair_module()

    assert module._iter_python_runtime_lib_dirs("darwin") == []


def test_delocate_uses_package_relative_library_subdirectory(monkeypatch) -> None:
    module = _load_repair_module()
    commands: list[list[str]] = []
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/tmp/{name}")
    monkeypatch.setattr(
        module,
        "_run",
        lambda command, *, env=None: commands.append(command),
    )
    wheel = Path("/tmp/mifrost.whl")

    module._repair_macos(wheel, Path("/tmp/repaired"), [])

    assert commands[0][commands[0].index("--lib-sdir") + 1] == ".dylibs"
