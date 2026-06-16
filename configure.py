#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

from local_build_dirs import (
    DEFAULT_LOCAL_BUILD_MODE,
    get_local_build_mode,
    local_build_dir_help_text,
    local_build_mode_help_text,
    local_build_mode_names,
)


def find_mimir_prefix():
    """Try to find mimir cmake directory via python import."""
    try:
        import pymimir as mimir

        # Assuming typical wheel structure or exposed API
        # If mimir exposes get_cmake_dir():
        if hasattr(mimir, "get_cmake_dir"):
            return mimir.get_cmake_dir()
        # Fallback inspection
        mimir_path = Path(mimir.__file__).parent
        cmake_path = mimir_path / "lib" / "cmake" / "mimir"
        if cmake_path.exists():
            return str(cmake_path)
    except ImportError:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Configure Mifrost Build")
    parser.add_argument("--noconan", action="store_true", help="Disable Conan")
    parser.add_argument(
        "--only_install", action="store_true", help="Only install dependencies"
    )
    parser.add_argument("--deps_policy", default="missing", help="Conan build policy")
    parser.add_argument("--cmake_cmd", default="cmake", help="CMake executable")
    parser.add_argument("--conan_cmd", default="conan", help="Conan executable")
    parser.add_argument(
        "--mode",
        choices=local_build_mode_names(),
        default=DEFAULT_LOCAL_BUILD_MODE.name,
        help=local_build_mode_help_text(),
    )
    parser.add_argument(
        "--build_dir",
        default=None,
        help=local_build_dir_help_text(),
    )
    parser.add_argument("--source_dir", default=".", help="Source directory")
    parser.add_argument(
        "--config", default=None, help="Build type (Debug, Release, etc.)"
    )
    parser.add_argument("--toolchain_file", help="CMake toolchain file")
    parser.add_argument(
        "--with_benchmarks",
        action="store_true",
        help="Enable benchmark dependencies and CMake targets",
    )

    args, extra_args = parser.parse_known_args()
    build_mode = get_local_build_mode(args.mode)

    use_conan = not args.noconan
    build_dir = Path(args.build_dir or build_mode.build_dir).resolve()
    source_dir = Path(args.source_dir).resolve()
    script_dir = Path(__file__).parent.resolve()
    build_config = args.config or build_mode.config
    with_benchmarks = args.with_benchmarks or build_mode.with_benchmarks

    print(
        f"Configuring Mifrost: mode={build_mode.name}, config={build_config}, conan={use_conan}"
    )

    # 1. Conan Install
    cmake_toolchain_file = args.toolchain_file

    if use_conan:
        # Conan 2 does not guarantee a default profile exists. Create one if needed.
        conan_home = Path(
            os.environ.get("CONAN_HOME", Path.home() / ".conan2")
        ).resolve()
        default_profile = conan_home / "profiles" / "default"
        if not default_profile.exists():
            subprocess.run([args.conan_cmd, "profile", "detect", "--force"], check=True)

        # Check custom deps (cista/loki/nauty) are exported?
        # conan_export.py logic
        conan_export_script = script_dir / "conan_export.py"
        if conan_export_script.exists():
            # Check if we need to export (naive check: just run it, it's fast)
            cmd = [
                sys.executable,
                str(conan_export_script),
                f"--conan_cmd={args.conan_cmd}",
            ]
            subprocess.run(cmd, check=True)

        conan_install_dir = build_dir / "conan"
        conan_args = [
            "-s",
            f"build_type={build_config}",
            "-s:h",
            "compiler.cppstd=gnu23",
            "-s:b",
            "compiler.cppstd=gnu23",
            "--profile:host=default",
            "--profile:build=default",
            "--options=loki/*:fPIC=True",
            "--options=nauty/*:fPIC=True",
            f"--build={args.deps_policy}",
        ]
        if with_benchmarks:
            conan_args.append("--options=mifrost/*:with_benchmarks=True")

        install_cmd = [
            args.conan_cmd,
            "install",
            str(source_dir),
            f"-of={conan_install_dir}",
            "-s",
            f"&:build_type={build_config}",  # Sets build type for consumer
        ] + conan_args

        print(f"Running Conan: {' '.join(install_cmd)}")
        subprocess.run(install_cmd, check=True)

        if not cmake_toolchain_file:
            # Check standard conan toolchain locations
            possible_path = (
                conan_install_dir
                / "build"
                / build_config
                / "generators"
                / "conan_toolchain.cmake"
            )
            possible_path_flat = conan_install_dir / "conan_toolchain.cmake"

            if possible_path.exists():
                cmake_toolchain_file = str(possible_path)
            elif possible_path_flat.exists():
                cmake_toolchain_file = str(possible_path_flat)

    if args.only_install:
        return

    # 2. CMake Configure
    cmake_args = [
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        f"-DCMAKE_BUILD_TYPE={build_config}",
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DPython_ROOT_DIR={Path(sys.executable).resolve().parent.parent}",
    ]
    if with_benchmarks:
        cmake_args.append("-DMIFROST_BUILD_BENCHMARKS=ON")

    if cmake_toolchain_file:
        cmake_args.append(f"-DCMAKE_TOOLCHAIN_FILE={cmake_toolchain_file}")

    # Inject Mimir Prefix Path
    mimir_prefix = find_mimir_prefix()
    if mimir_prefix:
        print(f"Found Mimir CMake at: {mimir_prefix}")
        # Add to CMAKE_PREFIX_PATH
        # We pass it as a define or env var.
        # CMAKE_PREFIX_PATH can be passed as argument.
        cmake_args.append(f"-DCMAKE_PREFIX_PATH={mimir_prefix}")
    else:
        print(
            "Warning: Could not auto-detect specific Mimir CMake path. Relying on env CMAKE_PREFIX_PATH."
        )

    print(
        f"Running CMake: {args.cmake_cmd} {' '.join(cmake_args)} {' '.join(extra_args)}"
    )
    subprocess.run([args.cmake_cmd] + cmake_args + extra_args, check=True)


if __name__ == "__main__":
    main()
