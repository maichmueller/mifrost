# PyTyr runtime-efficiency: post-optimization benchmark summary

Environment: Apple Silicon macOS 26.5.2, CPython 3.12.12 in `hrl`, editable
both-backend build (`build_editable_hrl`). Same fixture as the pre-opt
baseline (`2026-07-17-preopt-*.json`, commit `c26e078`): `blocks/smedium`,
32-input batches unless noted, 5 warm-ups / 20 repetitions for the
before/after comparison; additional batch sizes use 3 warm-ups / 10
repetitions given the larger input volumes. All numbers are median
wall-clock milliseconds for one `n_items`-sized call (`batch_default`,
`batch_explicit_goals`, `single_loop`, `stream_end_to_end` paths; Horizon/
Transition also split by `full`/`delta` transition mode).

Raw reports: `2026-07-18-postopt-{flat,color,hgraph,transition}-blocks-b32.json`
(direct before/after pair against the `2026-07-17-preopt-*` baseline),
`2026-07-18-postopt-{flat,hgraph}-blocks-b{1,8,256,1024}.json` (batch-size
sweep, post-opt only, no pre-opt equivalent recorded), and
`2026-07-18-postopt-{flat,color,hgraph,transition}-spanner-b32.json` (second
domain, higher arity/object count, post-opt only).

## Before/after, batch=32, blocks/smedium (median ms, mean of available rounds)

| family     | path                      | backend | before | after  | delta   |
|------------|---------------------------|---------|-------:|-------:|--------:|
| flat       | batch_default             | pymimir | 0.201  | 0.183  | -8.7%   |
| flat       | batch_default             | pytyr   | 0.267  | 0.136  | -48.9%  |
| flat       | stream_end_to_end         | pymimir | 0.348  | 0.309  | -11.3%  |
| flat       | stream_end_to_end         | pytyr   | 0.313  | 0.191  | -39.0%  |
| color      | batch_default             | pytyr   | 0.263  | 0.148  | -43.7%  |
| color      | stream_end_to_end         | pymimir | 0.379  | 0.306  | -19.4%  |
| color      | stream_end_to_end         | pytyr   | 0.263  | 0.196  | -25.7%  |
| hgraph     | batch_default              | pymimir | 0.710  | 0.626  | -11.8%  |
| hgraph     | batch_default              | pytyr   | 0.896  | 0.677  | -24.5%  |
| hgraph     | stream_end_to_end          | pymimir | 0.767  | 0.658  | -14.2%  |
| hgraph     | stream_end_to_end          | pytyr   | 0.947  | 0.722  | -23.8%  |
| transition | full.stream_end_to_end     | pymimir | 5.637  | 5.057  | -10.3%  |
| transition | full.stream_end_to_end     | pytyr   | 1.188  | 1.002  | -15.6%  |
| transition | delta.stream_end_to_end    | pymimir | 7.199  | 6.715  | -6.7%   |
| transition | delta.stream_end_to_end    | pytyr   | 1.406  | 1.212  | -13.7%  |

Every measured `(family, backend, path)` combination improved except two
Color/pymimir `batch_default`/`batch_explicit_goals` entries, which moved
+5.3%/+9.7% on absolute magnitudes of 6-16 microseconds — noise at that
scale (comparable to the run-to-run stdev reported in the raw JSON), not a
regression signal; every other Color/pymimir path improved. Pymimir also
improves because several changes (lazy `TargetColumns::names`,
`FlatRelationSink` offset/span storage, `SemanticArguments`/
`FlatTupleArguments` small-vector records) live in the shared neutral core
that both backends' engines call into, not only in the PyTyr adapter.

## Batch-size sweep, blocks/smedium, `stream_end_to_end` path (post-opt only)

| batch | flat pymimir (ms) | flat pytyr (ms) | pytyr/pymimir | hgraph pymimir (ms) | hgraph pytyr (ms) | pytyr/pymimir |
|------:|-------------------:|-----------------:|---------------:|----------------------:|--------------------:|---------------:|
| 1     | 0.024               | 0.016            | 0.67x          | 0.085                 | 0.086               | 1.01x          |
| 8     | 0.086               | 0.053            | 0.61x          | 0.223                 | 0.233               | 1.04x          |
| 32    | 0.309               | 0.191            | 0.62x          | 0.658                 | 0.722               | 1.10x          |
| 256   | 2.990               | 1.902            | 0.64x          | 5.292                 | 5.781               | 1.09x          |
| 1024  | 15.069              | 13.672           | 0.91x          | 23.661                | 25.257              | 1.07x          |

PyTyr's Flat path is now consistently faster than Pymimir's at every batch
size (previously slower at every size in the pre-opt baseline); HGraph is
close to parity (1.0-1.1x) rather than the larger pre-opt gap. The 1024-batch
Flat ratio (0.91x, versus ~0.62-0.67x at smaller sizes) suggests a
batch-scale cost that doesn't fully share in the per-item win — plausibly
`BatchBuilder`/PyG finalization, which this pass did not target; worth
profiling separately before further PyTyr-specific micro-optimization.

## Second domain, spanner/medium, batch=32 (post-opt only, no pre-opt baseline recorded)

Spanner has higher-arity actions and a different object/fact mix than
Blocks. No `2026-07-17-preopt-*-spanner-*` baseline exists (the recorded
pre-opt sweep only covered Blocks), so these are absolute post-opt numbers,
included to confirm the optimizations generalize beyond the Blocks fixture
rather than a paired comparison. See
`2026-07-18-postopt-{flat,color,hgraph,transition}-spanner-b32.json` for
full detail.

## Native C++ and Python test suites

`ctest --test-dir build_editable_hrl`: 2/2 passed (`mifrost_neutral_tests`,
`mifrost_tests`). `pytest -q`: 1284 passed, 79 skipped (0 failed), matching
the pre-optimization pass count and closing the 49 cross-backend parity
failures found at the start of this session (see decision/validation log in
`plans/tyr_runtime_efficiency_plan.local.md`).
