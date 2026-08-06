from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import conan_export


def _write_recipe(root: Path, name: str) -> Path:
    recipe_dir = root / name
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "conanfile.py").write_text("from conan import ConanFile\n")
    return recipe_dir


def test_local_recipe_exports_uses_conandata_requirements(tmp_path: Path) -> None:
    conandata = tmp_path / "conandata.yml"
    dependencies = tmp_path / "dependencies"
    dependencies.mkdir()
    loki_dir = _write_recipe(dependencies, "loki")
    nauty_dir = _write_recipe(dependencies, "nauty")
    _write_recipe(dependencies, "unused")
    conandata.write_text(
        "\n".join(
            [
                "requirements:",
                '  - "dlpack/1.2"',
                '  - "loki/abc123"',
                '  - "nauty/2.8.8"',
            ]
        )
    )

    exports = conan_export.local_recipe_exports(conandata, dependencies)

    assert exports == [
        conan_export.LocalRecipeExport("loki", "abc123", loki_dir),
        conan_export.LocalRecipeExport("nauty", "2.8.8", nauty_dir),
    ]


def test_module_runs_without_the_conan_package(tmp_path: Path) -> None:
    # CMake runs this script with the interpreter that builds the project, which
    # is not required to have the ``conan`` package importable -- Conan itself is
    # located separately as an executable via --conan_cmd. Re-import the module
    # in a subprocess with ``conan`` blocked to keep that guarantee enforced.
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        "import sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'conan' or name.startswith('conan.'):\n"
        "            raise ImportError('conan is intentionally unavailable')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
    )
    repo_root = Path(conan_export.__file__).resolve().parent
    env = dict(os.environ, PYTHONPATH=f"{tmp_path}{os.pathsep}{repo_root}")

    completed = subprocess.run(
        [sys.executable, str(repo_root / "conan_export.py"), "--help"],
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("dlpack/1.2", ("dlpack", "1.2")),
        ("  loki/abc123  ", ("loki", "abc123")),
        ("nauty/2.8.8@user/channel", ("nauty", "2.8.8")),
        ("valla/1.0#rev0123", ("valla", "1.0")),
        ("strong_type/v10@user/channel#rev0123", ("strong_type", "v10")),
    ],
)
def test_recipe_reference_parsing(reference: str, expected: tuple[str, str]) -> None:
    parsed = conan_export.RecipeReference.loads(reference)

    assert (parsed.name, parsed.version) == expected


@pytest.mark.parametrize("reference", ["", "   ", "nameonly", "/1.0", "name/"])
def test_recipe_reference_rejects_malformed(reference: str) -> None:
    with pytest.raises(ValueError):
        conan_export.RecipeReference.loads(reference)


def test_local_recipe_exports_rejects_missing_requirements(tmp_path: Path) -> None:
    conandata = tmp_path / "conandata.yml"
    conandata.write_text("sources: {}\n")

    with pytest.raises(ValueError, match="requirements"):
        conan_export.local_recipe_exports(conandata, tmp_path / "dependencies")


def test_export_local_recipes_invokes_conan_export(monkeypatch, tmp_path: Path) -> None:
    recipe_dir = _write_recipe(tmp_path, "loki")
    calls: list[list[str]] = []

    def fake_run(args: list[str], *, check: bool) -> None:
        assert check is True
        calls.append(args)

    monkeypatch.setattr(subprocess, "run", fake_run)

    ok = conan_export.export_local_recipes(
        [conan_export.LocalRecipeExport("loki", "abc123", recipe_dir)],
        "conan",
    )

    assert ok is True
    assert calls == [["conan", "export", str(recipe_dir), "--version=abc123"]]


def test_export_local_recipes_reports_failures(monkeypatch, tmp_path: Path) -> None:
    recipe_dir = _write_recipe(tmp_path, "loki")

    def fake_run(args: list[str], *, check: bool) -> None:
        raise subprocess.CalledProcessError(returncode=1, cmd=args)

    monkeypatch.setattr(subprocess, "run", fake_run)

    ok = conan_export.export_local_recipes(
        [conan_export.LocalRecipeExport("loki", "abc123", recipe_dir)],
        "conan",
    )

    assert ok is False
