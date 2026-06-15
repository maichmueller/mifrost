# Agent Instructions for `mifrost`

## Python Build/Test Workflow
- Preferred local development environment: activate the conda environment
  `beiw` before import checks, test runs, or editable-install work. That
  environment has an editable install of this repository in place.
- Do not treat a plain shell `python` import failure as a repository failure
  until the `beiw` environment has been activated or the intended interpreter has
  been confirmed.
- Never use `SKBUILD_EDITABLE_SKIP` for commands in this repository.
- Run Python tests/import checks with normal editable behavior so C++ changes are rebuilt and reflected in actual Python runtime behavior.
- Prefer `python -m pytest ...` (or equivalent) without bypass flags for validation.
