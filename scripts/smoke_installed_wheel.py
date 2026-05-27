from pathlib import Path

import mifrost


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
        path.name.startswith("libmifrost_core") for path in library_dir.iterdir()
    ):
        raise SystemExit("installed wheel is missing the reusable mifrost core library")

    config_path = cmake_dir / "mifrostConfig.cmake"
    targets_path = cmake_dir / "mifrostTargets.cmake"

    if not config_path.is_file():
        raise SystemExit("installed wheel is missing mifrostConfig.cmake")

    if not targets_path.is_file():
        raise SystemExit("installed wheel is missing mifrostTargets.cmake")

    config_text = config_path.read_text()
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

    print("installed wheel smoke test passed")


if __name__ == "__main__":
    main()
