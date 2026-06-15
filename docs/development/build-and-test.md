# Build and Test

This project uses a Python build backend that configures and builds the C++ core with CMake.

## Local setup

Use the `beiw` conda environment and the editable-install workflow described in
[AGENTS.md](../../AGENTS.md) before running import checks or tests.

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

If your environment does not provide extras, install the minimal tools directly:

```bash
python -m pip install -e .
python -m pip install pytest mkdocs mkdocs-material
```

## Build and install

Use a dedicated subdirectory under `build/` for each local purpose:

- `build/local-release` for normal local development builds
- `build/local-debug` for debug builds
- `build/stubs` for stub-generation or typing-oriented work
- `build/ci` for CI-like local reproductions

Avoid ad hoc root-level build directories like `build_*_probe` unless they are
short-lived ignored experiments.

Install the package (this triggers a full native build):

```bash
python -m pip install -v .
```

For editable development:

```bash
python -m pip install -e .
```

For explicit CMake-driven local builds, the blessed layout is:

```bash
python configure.py --config Release --build_dir build/local-release
python cbuild.py build/local-release
```

Use `build/local-debug` with `--config Debug` when you need a debug tree.

## Run tests

Run the full Python test suite:

```bash
python -m pytest
```

Run a single test module while iterating:

```bash
python -m pytest tests/encoding/test_color_encoder.py -q
```

## Build docs

Strict docs build (fails on warnings):

```bash
mkdocs build --strict
```
