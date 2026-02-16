# Installation

## Requirements

- Python `>=3.10`
- A working C++ toolchain
- `pymimir>=0.13.60` available in the active environment
- Conan 2 (for source builds via `configure.py`)
- Optional for PyG conversion: `torch` and `torch-geometric`

## Install From PyPI

```bash
python -m pip install mifrost
```

## Install From Source

### Minimal

```bash
git clone https://github.com/maichmueller/mifrost.git
cd mifrost
python -m pip install .
```

### Recommended (explicit configure/build)

```bash
git clone https://github.com/maichmueller/mifrost.git
cd mifrost
python configure.py --config Release --build_dir build
python cbuild.py build
```

### Alternative: CMake Presets

```bash
cmake --preset local-release
cmake --build build/local-release
```

In CLion, use `local-release` / `local-debug`. If your conda env is not
`rgnet_new`, set `Python_EXECUTABLE=/path/to/env/bin/python` in the profile's
CMake cache variables.

### With Benchmarks Enabled

```bash
git clone https://github.com/maichmueller/mifrost.git
cd mifrost
python configure.py --config Release --build_dir build_bench --with_benchmarks
python cbuild.py build_bench
```

If `conan` is not on `PATH`, pass it explicitly:

```bash
python configure.py --config Release --build_dir build --conan_cmd /path/to/conan
```

If CMake cannot find `pymimir`, extend `CMAKE_PREFIX_PATH`:

```bash
export CMAKE_PREFIX_PATH="$(python -c 'import pymimir; print(pymimir.get_cmake_dir())'):${CMAKE_PREFIX_PATH:-}"
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
export CMAKE_PREFIX_PATH="$(python -c 'import pymimir; print(pymimir.get_cmake_dir())'):${CMAKE_PREFIX_PATH:-}"
python -m pip install --no-build-isolation \
  --config-settings=editable.rebuild=true \
  -Cbuild-dir=build_editable \
  -e .
```
