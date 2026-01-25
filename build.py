#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Build Mifrost")
    parser.add_argument("build_dir", nargs="?", default="build", help="Build directory")
    parser.add_argument("--target", default="all", help="Build target")
    parser.add_argument("--clean", action="store_true", help="Clean before building")
    
    args, extra_args = parser.parse_known_args()
    
    build_dir = Path(args.build_dir)
    
    if args.clean:
        cmd = ["cmake", "--build", str(build_dir), "--target", "clean"]
        subprocess.run(cmd, check=True)

    cmd = ["cmake", "--build", str(build_dir), "--target", args.target] + extra_args
    print(f"Building: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
