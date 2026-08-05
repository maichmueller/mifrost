# Backend migration performance summary

Date: 2026-07-16

Scenario: 32 states or transitions from `blocks/smedium` on the same Apple
Silicon machine

Interpretation: performance is a regression guard; ordinary single-digit
movement is not a migration blocker.

This summary consolidates the versioned raw reports rather than adding another
tuning-oriented benchmark run. Negative percentages mean faster or lower
memory. Each family report retains the warmups, repetitions, alternating-round
medians, wheel hashes, module paths, and raw values needed to reproduce or
inspect the result.

| Representative path | Refactored Pymimir vs original | PyTyr vs refactored Pymimir | Process-memory context |
| --- | ---: | ---: | ---: |
| Flat explicit-goal batch / default batch | -2.43% | +54.92% | Pymimir RSS +2.21% |
| Color default batch | -6.08% | +84.41% | Planner/PyTorch imports dominate |
| HGraph default batch | -3.40% | +25.18% | Pymimir RSS +0.45% |
| Full transition default batch | +0.27% | +43.07% | Pymimir RSS -1.05% |

The PyTyr batch deltas are visible but do not indicate repeated Python
materialization or per-state ABI crossings. Single/stream results also vary in
both directions. For example, PyTyr Flat stream end-to-end is 12.61% faster,
Color stream is 24.44% faster, HGraph stream is 25.59% slower, and
full-transition stream is 77.99% faster. The shared semantic snapshot path is
retained for compatibility while the normal Pymimir paths use task-scoped
Views. This report remains a historical regression guard; current direct-View
smoke runs should be compared with a matched build rather than these older
cross-backend numbers.

Native Release smoke coverage is separate from the three-way Python/runtime
comparison. `mifrost_bench_hgraph`, `mifrost_bench_flat_relation`, and
`mifrost_bench_relation_encoders` all build and run. The comparison target's
single/batch count, slot, schema-name, and schema-arity parity checks report
`ok`; representative 32-item medians were 0.490 ms for HGraph and 0.189 ms for
Flat. macOS could not report CPU frequency or set thread affinity, so those
figures are build/regression context rather than optimization evidence.

Source reports:

- `2026-07-15-flat-backends.json`
- `2026-07-15-color-backends.json`
- `2026-07-15-hgraph-backends.json`
- `2026-07-15-transition-backends.json`

Acceptance: the refactored Pymimir representative paths show no material
abstraction regression or unbounded memory growth. PyTyr is fully functional
through the same semantic engines; further native traversal optimization is a
future opportunity, not a backend-migration gate.
