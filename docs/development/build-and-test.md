# Build and Test

This project uses a Python build backend that configures and builds the C++ core with CMake.

## Local setup

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

Install the package (this triggers a full native build):

```bash
python -m pip install -v .
```

For editable development:

```bash
python -m pip install -e .
```

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
