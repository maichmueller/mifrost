# Installation

## Requirements

- Python `>=3.10`
- A working C++ toolchain
- `pymimir` available in the active environment
- Optional for PyG conversion: `torch` and `torch-geometric`

## Install From PyPI

```bash
python -m pip install mifrost
```

## Install From Source

```bash
git clone https://github.com/maichmueller/mifrost.git
cd mifrost
python -m pip install .
```

If Conan is not available on `PATH`:

```bash
export CONAN_COMMAND=/path/to/conan
```

If CMake cannot find `pymimir`:

```bash
export MIFROST_MIMIR_CMAKE_DIR="$(python -c 'import pymimir; print(pymimir.get_cmake_dir())')"
```

## Editable Development Install

```bash
python -m pip install --no-build-isolation \
  --config-settings=editable.rebuild=true \
  -Cbuild-dir=build_editable \
  -e .
```
