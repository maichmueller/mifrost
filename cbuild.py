#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

from local_build_dirs import (
    DEFAULT_LOCAL_BUILD_MODE,
    get_local_build_mode,
    local_build_dir_help_text,
    local_build_mode_help_text,
    local_build_mode_names,
)


def main():
    parser = argparse.ArgumentParser(description="Build Mifrost")
    parser.add_argument(
        "--mode",
        choices=local_build_mode_names(),
        default=DEFAULT_LOCAL_BUILD_MODE.name,
        help=local_build_mode_help_text(),
    )
    parser.add_argument(
        "build_dir",
        nargs="?",
        default=None,
        help=local_build_dir_help_text(),
    )
    parser.add_argument("--target", default=None, help="Build target")
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Build the mifrost_bench_hgraph target",
    )
    parser.add_argument("--clean", action="store_true", help="Clean before building")

    args, extra_args = parser.parse_known_args()
    build_mode = get_local_build_mode(args.mode)

    build_dir = Path(args.build_dir or build_mode.build_dir)
    target = args.target or build_mode.default_target

    if args.clean:
        cmd = ["cmake", "--build", str(build_dir), "--target", "clean"]
        subprocess.run(cmd, check=True)

    if args.bench:
        target = "mifrost_bench_hgraph"

    cmd = ["cmake", "--build", str(build_dir), "--target", target] + extra_args
    print(f"Building: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
