from __future__ import annotations

from pathlib import Path

import pytest
import tomllib

import build_backend


@pytest.fixture(autouse=True)
def _clean_backend_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MIFROST_BUILD_BACKENDS", raising=False)
    monkeypatch.delenv("CMAKE_ARGS", raising=False)
    monkeypatch.delenv("SKBUILD_WHEEL_EXCLUDE", raising=False)


def test_package_builds_include_both_adapters_by_default() -> None:
    assert build_backend._selected_backends() == frozenset({"pymimir", "pytyr"})


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ("core", frozenset()),
        ("pymimir", frozenset({"pymimir"})),
        ("pytyr", frozenset({"pytyr"})),
        ("pymimir+pytyr", frozenset({"pymimir", "pytyr"})),
        ("both", frozenset({"pymimir", "pytyr"})),
    ],
)
def test_explicit_backend_selection(
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
    expected: frozenset[str],
) -> None:
    monkeypatch.setenv("MIFROST_BUILD_BACKENDS", selection)
    assert build_backend._selected_backends() == expected


def test_explicit_cmake_adapter_flags_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CMAKE_ARGS",
        "-DMIFROST_BUILD_PYMIMIR_ADAPTER:BOOL=OFF -DMIFROST_BUILD_PYTYR_ADAPTER=ON",
    )
    assert build_backend._selected_backends() == frozenset({"pytyr"})


def test_unknown_backend_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIFROST_BUILD_BACKENDS", "pymimir,unknown")
    with pytest.raises(RuntimeError, match="unknown backend"):
        build_backend._selected_backends()


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ("core", ["wheel"]),
        ("pymimir", ["wheel", "pymimir>=0.13.60"]),
        ("pytyr", ["wheel", "pytyr>=0.0.30"]),
        ("both", ["wheel", "pymimir>=0.13.60", "pytyr>=0.0.30"]),
    ],
)
def test_build_requirements_follow_backend_selection(
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
    expected: list[str],
) -> None:
    monkeypatch.setenv("MIFROST_BUILD_BACKENDS", selection)
    assert build_backend._with_backend_build_requirements(["wheel"]) == expected


def test_cmake_environment_disables_unselected_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MIFROST_BUILD_BACKENDS", "pytyr")
    build_backend._prepare_common_cmake_env()
    cmake_args = build_backend.os.environ["CMAKE_ARGS"]
    assert "-DMIFROST_BUILD_PYMIMIR_ADAPTER=OFF" in cmake_args
    assert "-DMIFROST_BUILD_PYTYR_ADAPTER=ON" in cmake_args
    assert build_backend.os.environ["SKBUILD_WHEEL_EXCLUDE"] == (
        "mifrost/_core.pyi;mifrost/_pymimir_adapter.pyi"
    )


def test_project_metadata_keeps_planners_optional() -> None:
    with open("pyproject.toml", "rb") as stream:
        project = tomllib.load(stream)

    assert project["project"]["dependencies"] == []
    build_requirements = project["build-system"]["requires"]
    assert not any("pymimir" in requirement for requirement in build_requirements)
    assert not any("pytyr" in requirement for requirement in build_requirements)
    extras = project["project"]["optional-dependencies"]
    assert extras["pymimir"] == ["pymimir>=0.13.60"]
    assert extras["pytyr"] == ["pytyr>=0.0.30"]
    assert set(extras["backends"]) == {"pymimir>=0.13.60", "pytyr>=0.0.30"}


def test_wheel_rows_use_one_pinned_both_backend_build_environment() -> None:
    base_requirements = Path("requirements/base-build.txt").read_text()
    assert "pymimir" not in base_requirements.lower()
    assert "pytyr" not in base_requirements.lower()

    default_requirements = Path("requirements/build.txt").read_text()
    assert "-r base-build.txt" in default_requirements
    assert "pymimir==" in default_requirements
    assert "pytyr==" in default_requirements

    workflow = Path(".github/workflows/wheels.yml").read_text()
    before_build_lines = [
        line.strip()
        for line in workflow.splitlines()
        if line.strip().startswith("CIBW_BEFORE_BUILD_")
    ]
    assert len(before_build_lines) == 2
    assert all("requirements/build.txt" in line for line in before_build_lines)
    assert 'CIBW_BUILD_FRONTEND: "pip; args: --no-build-isolation"' in workflow
    assert 'CIBW_TEST_REQUIRES: "pymimir==0.13.63 pytyr==0.0.30"' in workflow
    assert workflow.count("flavor:") == 2
    assert "MIFROST_BUILD_BACKENDS: both" in workflow
    assert "backend: core" not in workflow
    assert "backend: pymimir" not in workflow
    assert "backend: pytyr" not in workflow


def test_cibuildwheel_build_requires_generated_stubs_and_disables_regeneration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CIBUILDWHEEL", "1")
    monkeypatch.setattr(
        build_backend,
        "_required_wheel_stubs",
        lambda: (tmp_path / "_neutral_core.pyi",),
    )
    with pytest.raises(RuntimeError, match="pre-generated nanobind stubs"):
        build_backend.build_wheel(str(tmp_path))

    (tmp_path / "_neutral_core.pyi").write_text("# generated\n")
    monkeypatch.setattr(build_backend, "_maybe_prepare_conan", lambda _settings: None)
    monkeypatch.setattr(
        build_backend._sbc,
        "build_wheel",
        lambda *args: "mifrost.whl",
    )
    assert build_backend.build_wheel(str(tmp_path)) == "mifrost.whl"
    assert "-DMIFROST_GENERATE_STUBS=OFF" in build_backend.os.environ["CMAKE_ARGS"]


def test_plain_wheel_build_generates_stubs_without_a_pre_generated_requirement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("CIBUILDWHEEL", raising=False)
    monkeypatch.setattr(
        build_backend,
        "_required_wheel_stubs",
        lambda: (tmp_path / "_neutral_core.pyi",),
    )
    monkeypatch.setattr(build_backend, "_maybe_prepare_conan", lambda _settings: None)
    monkeypatch.setattr(
        build_backend._sbc,
        "build_wheel",
        lambda *args: "mifrost.whl",
    )
    # No stub files exist on disk; a plain (non-cibuildwheel) build must not
    # raise and must let CMake generate them inline, like any other dev
    # install or CI's tests.yml `pip install .`.
    assert build_backend.build_wheel(str(tmp_path)) == "mifrost.whl"
    assert "-DMIFROST_GENERATE_STUBS=ON" in build_backend.os.environ["CMAKE_ARGS"]
