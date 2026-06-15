# Architecture Overview

`mifrost` uses a `src/` layout with a thin public Python package on top of an
internal native core.

## Repository Layout

- `src/mifrost/` is the public Python package imported by users.
- `src/_core/mifrost/` contains the native implementation, nanobind bindings,
  schema helpers, and encoder engines.
- `src/CMakeLists.txt` ties the native sources into the build.
- `src/mifrost/_core.pyi` provides the public stub for the extension module.

The Python layer stays intentionally small: it exposes the user-facing API,
package-level helpers, and convenience wrappers, while the native layer owns
the actual encoding and batching logic.

## Runtime Shape

1. Python callers construct encoder objects from `src/mifrost/`.
2. Those wrappers delegate to the native core in `src/_core/mifrost/`.
3. Native encoders produce `BatchEncoding` as the primary output.
4. Python helpers convert that native output to PyTorch Geometric objects only
   when needed.

This keeps the package native-first without changing the boundary between the
Python API and the C++ implementation.

## Tests

- Python test suites live in `tests/encoding/`, `tests/python/`, and
  `tests/native/`.
- Native behavior and binding coverage live in `tests/cpp/`.
- Shared test helpers live at `tests/conftest.py` and `tests/parity_utils.py`.

## Documentation

- User-facing dynamic graph field behavior: `docs/how-to/dynamic-graph-fields.md`
- Historical design note for dynamic attr collation:
  `docs/development/dynamic_graph_fields_plan.md`
- Build and test workflow: `docs/development/build-and-test.md`
