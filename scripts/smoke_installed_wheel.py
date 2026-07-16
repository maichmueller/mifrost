import importlib
from pathlib import Path
from typing import Any

import mifrost


def _assert_encoding(encoding: Any, backend: str) -> None:
    if encoding.num_graphs != 1:
        raise SystemExit(f"{backend} smoke produced {encoding.num_graphs} graphs")
    if encoding.num_nodes <= 0:
        raise SystemExit(f"{backend} smoke produced an empty encoding")


def _smoke_pymimir(domain_path: Path, problem_path: Path) -> None:
    import pymimir

    domain = pymimir.Domain(domain_path)
    problem = pymimir.Problem(domain, problem_path, mode="grounded")
    encoder = mifrost.FlatRelationEncoder(domain, backend="pymimir")
    _assert_encoding(encoder.encode(problem.get_initial_state()), "Pymimir")


def _smoke_pytyr(domain_path: Path, problem_path: Path) -> None:
    from pypddl.formalism import ParserOptions
    from pytyr.formalism.planning import Parser
    from pytyr.planning import ExecutionContext
    from pytyr.planning.lifted import (
        AxiomEvaluatorFactory,
        StateRepositoryFactory,
        SuccessorGeneratorFactory,
        Task,
    )

    options = ParserOptions()
    planning_task = Parser(str(domain_path), options).parse_task(
        str(problem_path), options
    )
    task = Task(planning_task)
    context = ExecutionContext(1)
    evaluator = AxiomEvaluatorFactory().create(task, context)
    repository = StateRepositoryFactory().create(task, evaluator)
    generator = SuccessorGeneratorFactory().create(task, context, repository)
    encoder = mifrost.FlatRelationEncoder(planning_task, backend="pytyr")
    _assert_encoding(encoder.encode(generator.get_initial_node().get_state()), "PyTyr")


def main() -> None:
    package_root = Path(mifrost.__file__).resolve().parent
    include_dir = Path(mifrost.get_include_dir()).resolve()
    library_dir = Path(mifrost.get_library_dir()).resolve()
    cmake_dir = Path(mifrost.get_cmake_dir()).resolve()

    if include_dir != (package_root / "include").resolve():
        raise SystemExit(
            f"expected installed include dir under {package_root}, got {include_dir}"
        )

    expected_library_dirs = {
        (package_root / "lib").resolve(),
        (package_root / "lib64").resolve(),
    }
    if library_dir not in expected_library_dirs:
        raise SystemExit(
            f"expected installed library dir under {package_root}, got {library_dir}"
        )

    if cmake_dir != (library_dir / "cmake" / "mifrost").resolve():
        raise SystemExit(
            f"expected installed CMake dir under {library_dir}, got {cmake_dir}"
        )

    if mifrost.get_include() != str(include_dir):
        raise SystemExit("mifrost.get_include() did not match get_include_dir()")

    batch_builder_header = include_dir / "mifrost" / "core" / "batch_builder.hpp"
    if not batch_builder_header.is_file():
        raise SystemExit("installed wheel is missing exported SDK headers")

    if "nanobind" in batch_builder_header.read_text():
        raise SystemExit("installed SDK headers still expose nanobind")

    if (include_dir / "mifrost" / "core" / "map_view.hpp").exists():
        raise SystemExit("installed wheel still ships internal Python-only SDK headers")

    if not any(
        path.name.startswith("libmifrost_neutral_core")
        for path in library_dir.iterdir()
    ):
        raise SystemExit("installed wheel is missing the reusable neutral core library")

    config_path = cmake_dir / "mifrostConfig.cmake"
    targets_path = cmake_dir / "mifrostTargets.cmake"

    if not config_path.is_file():
        raise SystemExit("installed wheel is missing mifrostConfig.cmake")

    if not targets_path.is_file():
        raise SystemExit("installed wheel is missing mifrostTargets.cmake")

    config_text = config_path.read_text()
    with_pymimir = "set(mifrost_WITH_PYMIMIR_ADAPTER ON)" in config_text
    with_pytyr = "set(mifrost_WITH_PYTYR_ADAPTER ON)" in config_text
    if "find_dependency(nanobind" in config_text:
        raise SystemExit(
            "installed wheel still requires nanobind in mifrostConfig.cmake"
        )

    targets_text = targets_path.read_text()
    if "nanobind::nanobind" in targets_text:
        raise SystemExit(
            "installed wheel still exports nanobind as a public target dependency"
        )

    if ".conan" in targets_text:
        raise SystemExit(
            "installed wheel still leaks Conan build paths in mifrostTargets.cmake"
        )

    neutral_module = importlib.import_module("mifrost._neutral_core")
    if not hasattr(neutral_module, "BatchEncoding"):
        raise SystemExit("neutral extension is missing BatchEncoding")

    if with_pymimir:
        importlib.import_module("mifrost._pymimir_adapter")
        if not any(
            path.name.startswith("libmifrost_pymimir_adapter")
            for path in library_dir.iterdir()
        ):
            raise SystemExit("Pymimir wheel is missing its reusable adapter library")
    elif any(package_root.glob("_pymimir_adapter*.so")) or any(
        package_root.glob("_pymimir_adapter*.pyd")
    ):
        raise SystemExit("wheel unexpectedly contains the Pymimir extension")

    if with_pytyr:
        importlib.import_module("mifrost._pytyr_adapter")
        if not any(
            path.name.startswith("libmifrost_pytyr_adapter")
            for path in library_dir.iterdir()
        ):
            raise SystemExit("PyTyr wheel is missing its reusable adapter library")
    elif any(package_root.glob("_pytyr_adapter*.so")) or any(
        package_root.glob("_pytyr_adapter*.pyd")
    ):
        raise SystemExit("wheel unexpectedly contains the PyTyr extension")

    fixture_dir = Path(__file__).resolve().parent.parent / "data" / "pddl" / "blocks"
    domain_path = fixture_dir / "domain.pddl"
    problem_path = fixture_dir / "small.pddl"
    if with_pymimir:
        _smoke_pymimir(domain_path, problem_path)
    if with_pytyr:
        _smoke_pytyr(domain_path, problem_path)

    print(
        "installed wheel smoke test passed "
        f"(pymimir={with_pymimir}, pytyr={with_pytyr})"
    )


if __name__ == "__main__":
    main()
