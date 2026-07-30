from __future__ import annotations

import importlib

from scripts import resolve_native_prefixes


def test_native_prefix_falls_back_to_package_layout_when_import_fails(
    tmp_path, monkeypatch
) -> None:
    package_dir = tmp_path / "planner_stub"
    (package_dir / "native" / "include" / "planner").mkdir(parents=True)
    (package_dir / "__init__.py").write_text("raise RuntimeError('native import')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    assert (
        resolve_native_prefixes._native_prefix("planner_stub", "planner")
        == (package_dir / "native").resolve()
    )
