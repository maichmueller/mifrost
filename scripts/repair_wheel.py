#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

try:
    from resolve_native_prefixes import _native_prefix
except ImportError:  # pragma: no cover - package import path in unit tests
    from scripts.resolve_native_prefixes import _native_prefix


def _split_path_list(value: str | None) -> list[Path]:
    if not value:
        return []
    out: list[Path] = []
    for entry in value.split(os.pathsep):
        if entry:
            out.append(Path(entry))
    return out


def _iter_conan_lib_dirs() -> list[Path]:
    conan_home = os.environ.get("CONAN_HOME")
    if not conan_home:
        return []
    root = Path(conan_home).resolve()
    if not root.is_dir():
        return []

    candidates: list[Path] = []
    candidates.extend(root.glob("p/*/p/lib"))
    candidates.extend(root.glob("p/b/*/p/lib"))
    return [path for path in candidates if path.is_dir()]


def _iter_pymimir_lib_dirs() -> list[Path]:
    env_dir = os.environ.get("MIFROST_PYMIMIR_LIB_DIR")
    if env_dir:
        path = Path(env_dir).resolve()
        if path.is_dir():
            return [path]

    try:
        import pymimir  # type: ignore
    except Exception:
        return []

    candidates: list[Path] = []
    get_library_dir = getattr(pymimir, "get_library_dir", None)
    if callable(get_library_dir):
        try:
            candidates.append(Path(str(get_library_dir())).resolve())
        except Exception:
            pass

    get_cmake_dir = getattr(pymimir, "get_cmake_dir", None)
    if callable(get_cmake_dir):
        try:
            cmake_dir = Path(str(get_cmake_dir())).resolve()
            candidates.append(cmake_dir.parent.parent)
        except Exception:
            pass

    pymimir_file = getattr(pymimir, "__file__", None)
    if pymimir_file:
        try:
            pkg_dir = Path(str(pymimir_file)).resolve().parent
            site_dir = pkg_dir.parent
            candidates.append(pkg_dir / "lib")

            # Wheels repaired by auditwheel often place vendored deps into
            # <dist>.libs next to the package (e.g. pymimir.libs/libgcc_s-*.so.1).
            candidates.append(site_dir / "pymimir.libs")
            candidates.extend(site_dir.glob("pymimir*.libs"))
            # Be generous: some projects/vendor toolchains choose a different
            # dist name (e.g. pymimir_foo.libs) or we may need deps from other
            # wheels (e.g. libgcc_s-*.so.1 bundled into somepkg.libs).
            candidates.extend(site_dir.glob("*.libs"))

            # delocate sometimes uses <dist>.dylibs or in-package .dylibs.
            candidates.append(site_dir / "pymimir.dylibs")
            candidates.extend(site_dir.glob("pymimir*.dylibs"))
            candidates.extend(site_dir.glob("*.dylibs"))
            candidates.append(pkg_dir / ".dylibs")
            candidates.append(pkg_dir / ".libs")
        except Exception:
            pass

    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if resolved.is_dir() and resolved not in seen:
            out.append(resolved)
            seen.add(resolved)
    return out


def _iter_pytyr_lib_dirs() -> list[Path]:
    candidates = _split_path_list(os.environ.get("MIFROST_PYTYR_LIB_DIR"))
    package_markers = {
        "pytyr": "tyr",
        "pyyggdrasil": "yggdrasil",
        "pypddl": "loki",
    }
    for package_name, marker in package_markers.items():
        # Locate bundled native libraries from package files even when importing
        # the package fails because its extensions are not loader-resolvable yet.
        try:
            prefix = _native_prefix(package_name, marker)
            candidates.extend((prefix / "lib", prefix / "native" / "lib"))
        except Exception:
            pass
        try:
            module = importlib.import_module(package_name)
        except Exception:
            continue

        native_prefix = getattr(module, "native_prefix", None)
        if callable(native_prefix):
            try:
                prefix = Path(str(native_prefix())).resolve()
                candidates.extend((prefix / "lib", prefix / "native" / "lib"))
            except Exception:
                pass

        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            package_dir = Path(str(module_file)).resolve().parent
            site_dir = package_dir.parent
            candidates.extend(
                (
                    package_dir / "lib",
                    package_dir / "native" / "lib",
                    package_dir / ".libs",
                    package_dir / ".dylibs",
                    site_dir / f"{package_name}.libs",
                    site_dir / f"{package_name}.dylibs",
                )
            )
            candidates.extend(site_dir.glob(f"{package_name}*.libs"))
            candidates.extend(site_dir.glob(f"{package_name}*.dylibs"))
        except Exception:
            pass

    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if resolved.is_dir() and resolved not in seen:
            out.append(resolved)
            seen.add(resolved)
    return out


def _iter_python_runtime_lib_dirs(platform: str | None = None) -> list[Path]:
    """Return generic interpreter library paths only where they are safe.

    On macOS, prepending a Conda ``lib`` directory can make dyld resolve the
    system ``/usr/lib/libc++`` dependency to Conda's newer ``libc++``. Delocate
    then vendors that unrelated runtime and raises the wheel's minimum macOS
    version. Planner and Conan directories already cover Mifrost's non-system
    dependencies, so generic interpreter paths must stay out of macOS repair.
    """
    if (platform or sys.platform) == "darwin":
        return []

    candidates: list[Path] = []
    for key in ("LIBDIR", "LIBPL"):
        try:
            libdir = sysconfig.get_config_var(key)
        except Exception:
            libdir = None
        if libdir:
            candidates.append(Path(str(libdir)))

    try:
        python_prefix = Path(sys.executable).resolve().parent.parent
        for sub in ("lib", "lib64"):
            libdir = python_prefix / sub
            if libdir.is_dir():
                candidates.append(libdir)
    except Exception:
        pass
    return candidates


def _iter_candidate_lib_dirs() -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(_iter_pymimir_lib_dirs())
    candidates.extend(_iter_pytyr_lib_dirs())
    candidates.extend(_iter_conan_lib_dirs())
    candidates.extend(_split_path_list(os.environ.get("DYLD_LIBRARY_PATH")))
    candidates.extend(_split_path_list(os.environ.get("LD_LIBRARY_PATH")))

    candidates.extend(_iter_python_runtime_lib_dirs())

    cmake_prefixes = _split_path_list(os.environ.get("CMAKE_PREFIX_PATH"))
    for prefix in cmake_prefixes:
        if prefix.is_dir():
            candidates.append(prefix)
            lib_dir = prefix / "lib"
            if lib_dir.is_dir():
                candidates.append(lib_dir)
            pymimir_lib = prefix / "pymimir" / "lib"
            if pymimir_lib.is_dir():
                candidates.append(pymimir_lib)
            for relative in (
                "pytyr/native/lib",
                "pyyggdrasil/lib",
                "pypddl/native/lib",
            ):
                planner_lib = prefix / relative
                if planner_lib.is_dir():
                    candidates.append(planner_lib)

    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if resolved.is_dir() and resolved not in seen:
            out.append(resolved)
            seen.add(resolved)
    return out


def _verify_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.resolve()


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, env=env)


def _repair_macos(wheel: Path, outdir: Path, lib_dirs: list[Path]) -> None:
    delocate = shutil.which("delocate-wheel")
    listdeps = shutil.which("delocate-listdeps")
    if not delocate:
        raise RuntimeError("delocate-wheel not found on PATH")

    env = dict(os.environ)
    existing = env.get("DYLD_LIBRARY_PATH", "")
    prefix = os.pathsep.join(str(path) for path in lib_dirs)
    env["DYLD_LIBRARY_PATH"] = (
        f"{prefix}{os.pathsep}{existing}" if existing and prefix else prefix or existing
    )
    # Some macOS setups ignore DYLD_LIBRARY_PATH in restricted contexts, but
    # delocate's internal resolution can consult the fallback path too.
    existing_fb = env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    env["DYLD_FALLBACK_LIBRARY_PATH"] = (
        f"{prefix}{os.pathsep}{existing_fb}"
        if existing_fb and prefix
        else prefix or existing_fb
    )

    # IMPORTANT: delocate-wheel's `-L/--lib-sdir` sets the destination directory
    # for copied libraries inside the wheel; it is NOT a search path flag.
    # We use DYLD_LIBRARY_PATH / DYLD_FALLBACK_LIBRARY_PATH above to help dyld
    # resolve @rpath dependencies when delocate inspects binaries.
    cmd = [
        delocate,
        "-w",
        str(outdir),
        "-v",
        "--lib-sdir",
        ".dylibs",
        str(wheel),
    ]
    _run(cmd, env=env)

    repaired_wheel = outdir / wheel.name
    if listdeps and repaired_wheel.exists():
        _run([listdeps, "--all", str(repaired_wheel)], env=env)


def _repair_linux(wheel: Path, outdir: Path, lib_dirs: list[Path]) -> None:
    auditwheel = shutil.which("auditwheel")
    if not auditwheel:
        raise RuntimeError("auditwheel not found on PATH")

    env = dict(os.environ)
    existing = env.get("LD_LIBRARY_PATH", "")
    prefix = os.pathsep.join(str(path) for path in lib_dirs)
    env["LD_LIBRARY_PATH"] = (
        f"{prefix}{os.pathsep}{existing}" if existing and prefix else prefix or existing
    )

    _run([auditwheel, "repair", "-w", str(outdir), str(wheel)], env=env)

    repaired_wheel = outdir / wheel.name
    if repaired_wheel.exists():
        _run([auditwheel, "show", str(repaired_wheel)])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair mifrost wheels in a single place."
    )
    parser.add_argument("wheel", type=Path, help="Path to wheel to repair.")
    parser.add_argument(
        "outdir", type=Path, help="Output directory for repaired wheel."
    )
    args = parser.parse_args()

    wheel = _verify_exists(args.wheel)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    lib_dirs = _iter_candidate_lib_dirs()
    if not lib_dirs:
        raise RuntimeError(
            "Could not locate any runtime library directories. "
            "Set CONAN_HOME, MIFROST_PYMIMIR_LIB_DIR, or "
            "MIFROST_PYTYR_LIB_DIR; alternatively install the selected planner "
            "packages in the wheel build environment."
        )

    print("Using library search dirs:")
    for d in lib_dirs:
        print(" -", d)

    if sys.platform == "darwin":
        _repair_macos(wheel, outdir, lib_dirs)
        return 0

    if sys.platform.startswith("linux"):
        _repair_linux(wheel, outdir, lib_dirs)
        return 0

    raise RuntimeError(f"Unsupported platform for wheel repair: {sys.platform}")


if __name__ == "__main__":
    raise SystemExit(main())
