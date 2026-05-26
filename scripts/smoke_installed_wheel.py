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

    if not (include_dir / "mifrost" / "core" / "batch_builder.hpp").is_file():
        raise SystemExit("installed wheel is missing exported SDK headers")

    if not any(
        path.name.startswith("libmifrost_core") for path in library_dir.iterdir()
    ):
        raise SystemExit("installed wheel is missing the reusable mifrost core library")

    if not (cmake_dir / "mifrostConfig.cmake").is_file():
        raise SystemExit("installed wheel is missing mifrostConfig.cmake")

    if not (cmake_dir / "mifrostTargets.cmake").is_file():
        raise SystemExit("installed wheel is missing mifrostTargets.cmake")

    print("installed wheel smoke test passed")


if __name__ == "__main__":
    main()
