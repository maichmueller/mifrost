# Mifrost Repository Quality and Architecture Improvement Plan

## Context

This plan addresses repository hygiene, packaging drift, quality gates, and
module structure issues found during the June 2026 review.

Important local environment note: day-to-day development should happen from the
conda environment `beiw`, which has an editable install of this repository. Use
that interpreter for import checks and test runs. Do not use
`SKBUILD_EDITABLE_SKIP`; editable behavior should rebuild C++ changes so Python
runtime checks reflect the current native code.

## Goals

- Make dependency and version metadata single-source-of-truth.
- Stop local build products from making the source package stale or
  unimportable.
- Ensure default CI and local commands exercise all important Python and C++
  behavior.
- Add lint/type gates that match the public API surface this package ships.
- Reduce duplicated public API export tables and duplicated input-normalization
  paths.
- Keep the native core reusable while making nanobind binding files thinner and
  easier to test.

## Non-Goals

- Do not redesign encoder semantics in the same change set as repo hygiene.
- Do not remove existing public APIs without a deprecation path.
- Do not weaken editable install behavior to speed up local commands.

## Phase 0: Baseline and Safety

1. Record the intended local workflow in `AGENTS.md`.
   - Status: done in this plan update.
   - Acceptance: future agents know to use `conda activate beiw` and avoid
     bypassing editable rebuilds.

2. Capture a baseline from the correct environment.
   - Commands:
     - `conda activate beiw`
     - `python -c "import sys, mifrost; print(sys.executable); print(mifrost.__file__)"`
     - `python -m pytest -q`
     - `python configure.py --build_dir build_ci --config Release`
     - `cmake --build build_ci --target mifrost_tests`
     - `ctest --test-dir build_ci -R mifrost_tests --output-on-failure`
   - Acceptance: baseline failures, if any, are documented before fixes start.

3. Create a tracked cleanup checklist for ignored local outputs.
   - Include root `build*`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`,
     `.benchmarks`, `.serena`, `src/mifrost/*.so`, `src/mifrost/*.dylib`,
     `src/mifrost/lib/`, and `src/mifrost/ops/__pycache__/`.
   - Acceptance: `git status --ignored=matching` becomes readable enough for
     agents and IDEs.

## Phase 1: Repository Hygiene and Legal Correctness

1. Add license and project governance basics.
   - Add `LICENSE` with GPL-3.0-only text.
   - Add a minimal `SECURITY.md`.
   - Add a minimal `CHANGELOG.md` or explicitly document that release notes live
     only in GitHub releases.
   - Acceptance: `twine check dist/*` still passes, and package metadata has a
     corresponding license file.

2. Tighten `.gitignore`.
   - Add explicit entries for cache and local-agent directories:
     `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`, `.serena/`,
     `.benchmarks/`, nested cache directories, and generated editable outputs
     under `src/mifrost`.
   - Keep `src/mifrost/_core.pyi` handling deliberate, because the wheel and
     sdist workflows currently rely on generated stubs.
   - Status: expanded to cover nested Python/tool caches, editable native
     outputs, local Conan caches, CMake user presets, and generated example
     plots.
   - Acceptance: no tool cache remains visible unless intentionally untracked.

3. Remove stale local artifacts from the working tree.
   - Delete the orphaned `src/mifrost/ops/` directory if it contains only
     `__pycache__`.
   - Remove ad hoc probe scripts or move useful probes into named tests.
   - Acceptance: no source package directories exist only because of bytecode.

## Phase 2: Version and Dependency Source of Truth

1. Remove the hardcoded Conan fallback dependency set.
   - Replace the fallback in `conanfile.py` with a fatal error when
     `conandata.yml` is missing or malformed.
   - Acceptance: there is no path that silently builds against a Loki revision
     different from `conandata.yml`.

2. Unify project version flow.
   - Choose `pyproject.toml` as the release version source.
   - Either derive the Conan recipe version from package metadata during export
     or document Conan package versioning as intentionally independent.
   - Remove the stale standalone CMake fallback version or set it to a clearly
     non-release dev value.
   - Status: implemented; Conan reads `pyproject.toml`, and standalone CMake
     config falls back to `0.0.0.dev0` instead of a stale release version.
   - Acceptance: `pyproject.toml`, Conan export behavior, generated CMake
     package versions, and release tags cannot drift silently.

3. Replace hand-written dependency export parsing.
   - Update `conan_export.py` to parse `conandata.yml` with a YAML parser or
     Conan metadata APIs.
   - Derive the local export list from directories that contain local recipes and
     are actually referenced by `conandata.yml`.
   - Decide whether `strong_type` is intentionally external or should be
     exported locally like `loki`, `nauty`, `cista`, and `valla`.
   - Status: implemented; `conan_export.py` now uses PyYAML plus Conan
     `RecipeReference`, and it exports every local recipe that is referenced by
     `conandata.yml`, including `strong_type`.
   - Acceptance: adding or removing a local dependency requires changing one
     metadata file, not a script string literal.

## Phase 3: Test Discovery and CI Authority

1. Move native binding Python tests out of `src/_core`.
   - Move `src/_core/mifrost/tests/test_bindings.py`,
     `test_transition_dag.py`, and `test_unordered_dense_typecaster.py` into
     `tests/native/` or `tests/python/native/`.
   - Keep any C++-only tests under `tests/cpp/`.
   - Acceptance: `python -m pytest -q` from `beiw` runs these tests by default.

2. Consolidate pytest config into `pyproject.toml`.
   - Move `pytest.ini` settings into `[tool.pytest.ini_options]`.
   - Use one `testpaths` entry for `tests`.
   - Add markers if slow/perf/native tests need opt-in treatment.
   - Acceptance: no redundant `tests/python` path, and no orphaned Python tests.

3. Rationalize CI test workflows.
   - Make `tests.yml` the authoritative source build and test workflow.
   - Delete `pip.yml` or reduce it to a targeted install-smoke job not already
     covered by `tests.yml`.
   - Keep wheel validation in `wheels.yml`.
   - Acceptance: pull requests show one clear source-test signal, one wheel
     signal, one docs signal, and optional performance signal.

4. Decide performance gate semantics.
   - Either keep it explicitly non-blocking and rename it to "Performance
     Monitor", or make boundary failures block on protected branches.
   - Acceptance: the workflow name and check conclusion match its enforcement.

## Phase 4: Lint, Type, and API Surface Gates

1. Enable Ruff lint.
   - Add `[tool.ruff]` and `[tool.ruff.lint]` to `pyproject.toml`.
   - Re-enable `ruff-check` in pre-commit.
   - Add a CI lint job or fold it into the source-test workflow before builds.
   - Acceptance: formatting and linting are both enforced.

2. Add a type-checking gate.
   - Choose mypy or pyright for Python wrappers and tests.
   - Add config to `pyproject.toml` or a single dedicated config file.
   - Include generated `_core.pyi` in the type-check path only after stubs are
     available.
   - Acceptance: wrapper API typing is checked in CI, and cache directories are
     not the only evidence of local type checking.

3. Add public API export tests.
   - Assert that top-level `mifrost.__all__`, `mifrost.encoders.__all__`, and
     lazy export tables agree for intended public encoder names.
   - Add a test for `mifrost.FlatRootedHorizonEncoder` if top-level export is
     intended, or document that it is intentionally under `mifrost.encoders`.
   - Acceptance: export-table drift fails fast.

## Phase 5: Build Output Isolation

1. Stop writing compiled editable artifacts directly into `src/mifrost`.
   - Investigate scikit-build-core editable install modes that preserve normal
     rebuild behavior while keeping compiled outputs in a controlled build
     directory or redirect directory.
   - If direct source output is retained, add a cleanup target that removes stale
     extensions and dylibs before rebuild.
   - Status: scoped out as an intentional scikit-build editable behavior for
     now; the repo now ignores those generated artifacts and validates editable
     rebuilds through the `beiw` workflow.
   - Acceptance: an old `_core.cpython-*.so` cannot load against a newer
     `libmifrost_core.dylib` and break imports.

2. Define one blessed local build layout.
   - Prefer `build/local-debug`, `build/local-release`, `build/stubs`, and
     `build/ci` over ad hoc root directories like `build_*_probe`.
   - Update `configure.py`, `cbuild.py`, docs, and CMake presets to point at
     those locations.
   - Status: implemented for the Python helper defaults and docs; explicit CI,
     stub, and benchmark commands still pass purpose-specific subdirectories
     under `build/`.
   - Acceptance: normal workflows no longer create arbitrary root-level build
     directories.

3. Make stub generation explicit.
   - Decide whether `_core.pyi` is source-controlled, generated in CI, or
     generated on demand for dev.
   - Align `.gitignore`, `pyproject.toml` `sdist.include`, and wheel workflows
     with that decision.
   - Acceptance: stub freshness has one validation path.

## Phase 6: Python Wrapper Module Deepening

Candidate A: encoder lane contract.

- Cluster: `src/mifrost/encoders/base.py`, `common.py`, `_lane_specs.py`,
  `_action_contract.py`, `_batch_contract.py`, `_root_policy.py`,
  `_target_sources.py`, and callers in `flat.py`, `flat_horizon.py`,
  `flat_transition.py`, `hgraph.py`, `horizon.py`, and `transition.py`.
- Why coupled: they jointly own user input normalization, batch semantics,
  optional goals/actions/history payloads, root policy, and native engine call
  shape.
- Dependency category: in-process.
- Proposed direction: introduce one internal "lane input contract" module that
  returns typed prepared payload objects for single, batch, and stream paths.
- Status: partially implemented. Existing lane/batch/action contract modules are
  now the internal owners for parsed batch plans and batch payload conversion;
  `common.py` keeps compatibility wrappers for older private imports.
- Test impact: replace narrow helper tests with boundary tests that assert each
  encoder lane accepts/rejects public input shapes consistently.
- Acceptance: adding a new encoder lane requires defining a lane spec and native
  engine adapter, not duplicating validation and stream glue across large files.

Candidate B: PyG conversion boundary.

- Cluster: `common.py`, `flat_data.py`, native `BatchEncoding.as_pyg`, and
  tests around `encoding_to_tensors`, `flat_relation_data_from_pyg`, and dynamic
  graph fields.
- Why coupled: conversion behavior is split between native views, dict payloads,
  flat carrier metadata, and Python wrappers.
- Dependency category: in-process with optional torch/PyG dependencies.
- Proposed direction: isolate conversion into a public conversion module with a
  small interface:
  `to_pyg(encoding, *, as_batch, include_metadata)` and
  `to_tensor_payload(encoding)`.
- Status: implemented for the Python wrapper boundary. Conversion now lives in
  `mifrost.encoders.conversion`, encoder bases use that boundary for the
  default path, and legacy encoder namespace helpers remain as compatibility
  exports.
- Test impact: boundary tests cover hetero, homo, flat, metadata included or
  excluded, and optional dependency degradation.
- Acceptance: encoder classes no longer need to know low-level schema conversion
  details.

Candidate C: public export manifest.

- Cluster: `mifrost/__init__.py`, `mifrost/encoders/__init__.py`, generated
  stubs, docs, and API tests.
- Why coupled: the same public names are listed in multiple places.
- Dependency category: in-process.
- Proposed direction: one manifest owns public encoder exports, optional
  dependency behavior, lazy imports, and top-level re-export policy.
- Test impact: add export-manifest tests and remove duplicated name assertions.
- Acceptance: `FlatRootedHorizonEncoder`-style drift cannot happen.

## Phase 7: Native Binding Deepening

Candidate D: `BatchEncoding` binding split.

- Cluster: `src/_core/mifrost/init_batch_encoding.cpp`,
  `src/_core/mifrost/batch_encoding_python_collation.*`,
  `src/_core/mifrost/batch_encoding_graph_field_access.*`,
  `src/_core/mifrost/schema_bindings.cpp`, and Python tests currently in
  `tests/native/`.
- Why coupled: one binding file owns serialization, schema conversion, tensor
  conversion, graph fields, Python attribute collation, repr behavior, and
  nanobind exposure.
- Dependency category: in-process.
- Proposed direction: move behavior into native units with plain C++ tests where
  possible, leaving nanobind files as thin adapters.
- Status: partially implemented. Repr/string formatting for `BatchEncoding` now
  lives in a dedicated native source file, reducing `init_batch_encoding.cpp`
  while keeping the binding registration in place. Owner target-device and
  tensor-cache helpers now live in a shared native helper module used by graph
  field access, PyG views, and binding methods. Schema fingerprinting and PyG
  tensor-key lookup now live in a dedicated schema helper module.
- Test impact: use C++ tests for schema/collation/serialization core behavior
  and Python tests for binding-level lifetime and conversion behavior.
- Acceptance: binding files become mostly declarations and small adapters.

Candidate E: reusable build/package support.

- Cluster: `build_backend.py`, `configure.py`, `cbuild.py`, `CMakePresets.json`,
  `src/CMakeLists.txt`, and workflows.
- Why coupled: editable, wheel, Conan provider, toolchain, stub generation, and
  RPATH behavior are distributed across several scripts.
- Dependency category: local-substitutable by using temp build dirs and isolated
  virtual or conda environments.
- Proposed direction: one build orchestration module or script owns build modes:
  `editable-dev`, `ci-source`, `wheel`, `stubs`, and `bench`.
- Test impact: add script-level smoke tests for command construction and one CI
  smoke per build mode.
- Acceptance: adding a new build mode does not require editing multiple scripts
  plus workflow snippets independently.

## Suggested Commit Order

1. Add `LICENSE`, ignore updates, and remove stale local-only artifacts.
2. Move orphaned Python tests into `tests/native/` and simplify pytest config.
3. Remove Conan fallback and fix version/dependency source of truth.
4. Consolidate CI test workflows and add Ruff lint config.
5. Add type-check config and public export tests.
6. Fix top-level export drift or document intended scope.
7. Isolate editable build outputs or add robust cleanup before rebuild.
8. Deepen the Python encoder lane contract.
9. Deepen PyG conversion boundary.
10. Split `init_batch_encoding.cpp` into behavior modules plus thin bindings.

## Validation Matrix

- Local baseline from `beiw`:
  - `python -c "import mifrost; print(mifrost.__file__)"`
  - `python -m pytest -q`
  - `python configure.py --build_dir build_ci --config Release`
  - `cmake --build build_ci --target mifrost_tests`
  - `ctest --test-dir build_ci -R mifrost_tests --output-on-failure`
- Quality gates:
  - `pre-commit run --all-files`
  - `python -m ruff check .`
  - type checker command chosen in Phase 4
- Packaging:
  - `python -m build --sdist`
  - wheel workflow or local `pip wheel . -w wheelhouse --no-deps`
  - `python scripts/smoke_installed_wheel.py` against an installed wheel

## Open Decisions

- Should `FlatRootedHorizonEncoder` be top-level public API or only
  `mifrost.encoders` API?
- Should generated `_core.pyi` be committed, generated in CI, or generated only
  for release artifacts?
- Should the performance workflow block protected branches?
- Should `strong_type` be exported locally, or is the Conan Center package
  deliberately sufficient?
