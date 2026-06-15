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
        (ROOT / "build/local-release").resolve()
    )


def test_cbuild_defaults_to_local_release_build_dir(monkeypatch) -> None:
    module = _load_script_module("cbuild_defaults_test", "cbuild.py")
    calls: list[list[str]] = []

    monkeypatch.setattr(sys, "argv", ["cbuild.py"])
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda cmd, *, check: calls.append(cmd),
    )

    module.main()

    assert calls == [["cmake", "--build", "build/local-release", "--target", "all"]]
