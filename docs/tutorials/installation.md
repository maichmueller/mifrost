# Installation

## Requirements

- Python `>=3.12`
- A working C++ toolchain
- At least one planner for encoder use: `pymimir>=0.13.60` or `pytyr>=0.0.30`
- Conan 2 (for source builds via `configure.py`)
- Optional for PyG conversion: `torch` and `torch-geometric`

## Install From PyPI

Choose one planner or install both for same-process interoperability:

```bash
python -m pip install "mifrost[pymimir]"
python -m pip install "mifrost[pytyr]"
python -m pip install "mifrost[backends]"
```

The base package contains the planner-neutral API. Planner extras are optional;
installing one does not install the other.

## Install From Source

### Minimal

```bash
git clone https://github.com/maichmueller/mifrost.git
cd mifrost
MIFROST_BUILD_BACKENDS=pytyr python -m pip install ".[pytyr]"
```

`MIFROST_BUILD_BACKENDS` accepts `core`, `pymimir`, `pytyr`, or `both`.
Reduced builds omit the unused native adapter entirely. Package builds default
to `both`; runtime extras still determine which planners are installed.

### Recommended (explicit configure/build)

```bash
git clone https://github.com/maichmueller/mifrost.git
cd mifrost
python configure.py --mode local-release
python cbuild.py --mode local-release
```

### Alternative: CMake Presets

```bash
cmake --preset local-release
cmake --build build/local-release
```

In CLion, use `local-release` / `local-debug`. If CMake does not auto-detect
the intended Python environment, set
`Python_EXECUTABLE=/path/to/env/bin/python` in the profile's CMake cache
variables.

### With Benchmarks Enabled

```bash
git clone https://github.com/maichmueller/mifrost.git
cd mifrost
python configure.py --mode bench
python cbuild.py --mode bench
```

If `conan` is not on `PATH`, pass it explicitly:

```bash
python configure.py --config Release --conan_cmd /path/to/conan
```

If a selected planner cannot be found, use its package-provided CMake prefix:

```bash
export CMAKE_PREFIX_PATH="$(python -c 'import pymimir; print(pymimir.get_cmake_dir())'):${CMAKE_PREFIX_PATH:-}"
export CMAKE_PREFIX_PATH="$(python -c 'import pytyr; print(pytyr.cmake_prefix())'):${CMAKE_PREFIX_PATH:-}"
```

If you switch benchmark mode (`--with_benchmarks` on/off), use a fresh build directory or re-run `configure.py` with a clean build tree to avoid stale CMake cache settings.

## Linux cibuildwheel (podman) with external Conan cache

```bash
export CIBW_CONTAINER_ENGINE=podman
export CIBW_CONTAINER_ENGINE_OPTIONS="--volume ${CONAN_HOME_HOST:?set CONAN_HOME_HOST}:/conan_home"
export CIBW_ENVIRONMENT="CONAN_HOME=/conan_home CONAN_NON_INTERACTIVE=1"
export CIBW_BEFORE_BUILD_LINUX="rm -rf /project/build && python -m pip install conan"
```

## Editable Development Install

```bash
export MIFROST_BUILD_BACKENDS=both
python -m pip install --no-build-isolation \
  --config-settings=editable.rebuild=true \
  -Cbuild-dir=build_editable \
  -e .
```

## Reusable C++ SDK from an installed package

Installed wheels expose the native SDK alongside the Python package:

```bash
python -c 'import mifrost; print(mifrost.get_include_dir())'
python -c 'import mifrost; print(mifrost.get_cmake_dir())'
python -c 'import mifrost; print(mifrost.get_library_dir())'
```

For downstream CMake projects, either prepend `mifrost.get_cmake_dir()` to
`CMAKE_PREFIX_PATH` or pass `-Dmifrost_DIR="$(python -c 'import mifrost; print(mifrost.get_cmake_dir())')"`.

For a C++-only SDK, disable Python independently of the selected adapters:

```bash
cmake -S . -B build/native -DMIFROST_BUILD_PYTHON=OFF \
  -DMIFROST_BUILD_PYMIMIR_ADAPTER=OFF \
  -DMIFROST_BUILD_PYTYR_ADAPTER=ON
cmake --build build/native
cmake --install build/native --prefix /path/to/mifrost-sdk
```

An installed PyTyr adapter first looks for planner libraries in the wheel-style
sibling layout. Standalone SDK consumers can instead set
`MIFROST_PYTYR_NATIVE_PREFIX`, `MIFROST_PYYGGDRASIL_NATIVE_PREFIX`, and
`MIFROST_PYPDDL_NATIVE_PREFIX` in their CMake cache (or the corresponding
environment variables without the `MIFROST_` prefix).
