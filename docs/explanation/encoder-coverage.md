# Encoder Coverage Map

| Encoder | Intended Scenario | What It Encodes | Primary Output | Cited Example/Test |
| --- | --- | --- | --- | --- |
| `HGraphEncoder` | General state encoding with optional goals/actions/history | Objects, facts, optional goal/action/history structures | `BatchEncoding` (hetero) | `examples/encoders/hetero_encoder_example.py`, `tests/encoding/test_batch_encoding.py` |
| `FlatRelationEncoder` | Compact packed state/goal/action/history encoding | Packed relation tensors plus target-entity metadata | `BatchEncoding` (flat-on-homo carrier) | `examples/encoders/flat_relation_encoder_example.py`, `tests/encoding/test_flat_relation_encoder.py` |
| `HorizonEncoder` | Root + lookahead candidate reasoning | Root state with `TransitionDAG` target/candidate relations | `BatchEncoding` (hetero) | `examples/encoders/horizon_hetero_encoder_example.py`, `tests/encoding/test_horizon_encoder.py` |
| `FlatHorizonEncoder` | Compact root + lookahead candidate reasoning | Packed flat relations with state-target carrier rows from `TransitionDAG` | `BatchEncoding` (flat-on-homo carrier) | `tests/encoding/test_flat_horizon_encoder.py` |
| `TransitionHGraphEncoder` | Full transition modeling | Source + successor structures in one transition graph | `BatchEncoding` (hetero) | `examples/encoders/transition_hetero_encoder_example.py`, `tests/encoding/test_transition_hetero_encoder.py` |
| `TransitionEffectsHGraphEncoder` | Effect/delta transition learning | Transition-focused deltas/effects representation | `BatchEncoding` (hetero) | `examples/encoders/transition_change_hetero_encoder_example.py`, `tests/encoding/test_transition_change_encoder.py` |
| `FlatTransitionEncoder` | Compact full transition modeling | One-step successor encoding over the flat state-target substrate | `BatchEncoding` (flat-on-homo carrier) | `examples/encoders/flat_transition_encoder_example.py`, `tests/encoding/test_flat_transition_encoder.py` |
| `FlatTransitionEffectsEncoder` | Compact effect/delta transition learning | One-step delta/effects encoding over the flat state-target substrate | `BatchEncoding` (flat-on-homo carrier) | `examples/encoders/flat_transition_encoder_example.py`, `tests/encoding/test_flat_transition_encoder.py` |
| `ColorEncoder` | Compact homogeneous baselines | Integer-style color/object/predicate/action features | `BatchEncoding` (homo) | `examples/encoders/color_encoder_example.py`, `tests/encoding/test_color_encoder.py` |
| `StarGraphEncoder` | Vanilla-GNN state encoding with reified atoms | Object nodes plus fact/goal/subgoal/history/action nodes joined by position-aware star edges; arity-0 literals carry a `nullary_self` loop | PyG `Data` (`x_ids` `[N, 6]` / `edge_attr` `[E, 9]` integer channels) | `examples/encoders/derived_graph_example.py`, `tests/encoding/test_gnn_conformance.py`, `tests/encoding/test_derived_facades.py`, `tests/encoding/test_derived_backends_parity.py` |
| `ObjectGraphEncoder` | Compact objects-only projections (`clique` / `chain` / `star_first`) | Argument-pair edges between objects, each labeled with its instance's relation/role/sign/goal-level/history/category; arity-1 literals emit a `unary_self` loop and arity-0 literals a `nullary_self` loop on a trailing `anchor` node; grounded actions are still reified as nodes | PyG `Data` (`x_ids` integer channels) plus `anchor_index` and the always-on tuple instance table | `examples/encoders/derived_graph_example.py`, `tests/encoding/test_gnn_conformance.py`, `tests/encoding/test_derived_facades.py`, `tests/encoding/test_derived_backends_parity.py` |
| `AtomLineGraphEncoder` | Atom co-occurrence structure over the star universe | Star view plus fact-fact `line_share` edges bounded by `line_graph_max_degree` and honouring `include_reverse_edges`; `line_share` rows carry zero instance labels by design | PyG `Data` (`x_ids` integer channels) | `tests/encoding/test_gnn_conformance.py`, `tests/encoding/test_derived_facades.py`, `tests/encoding/test_derived_backends_parity.py` |
| `HypergraphIncidenceEncoder` | Native hypergraph layers (PyG `HypergraphConv`) | Literal instances as hyperedges over object nodes; incidence restacked into `hyperedge_index` / `hyperedge_attr_ids` (`[M, 6]`, all six node channels); never empty, so PyG's `num_edges` inference matches | PyG `Data` + hyperedge fields | `tests/encoding/test_gnn_conformance.py`, `tests/encoding/test_derived_facades.py`, `tests/encoding/test_derived_backends_parity.py` |
| `TransformerBiasEncoder` | Transformer attention-bias inputs from planning structure | Objects-only clique projection (anchor and tuple instance table included) plus sparse shortest-path fields `spd_src` / `spd_dst` / `spd_dist` over object pairs only | PyG `Data` + spd, anchor and tuple fields | `tests/encoding/test_gnn_conformance.py`, `tests/encoding/test_derived_facades.py`, `tests/encoding/test_derived_backends_parity.py` |
| `TupleTensorEncoder` | Padding-free tuple views for sequence-style consumers | CSR tuple view on the star-view carrier: `tuple_args` / `tuple_sizes` / `tuple_ptr` plus six id vectors (`tuple_rel_ids`, `tuple_role_ids`, `tuple_sign_ids`, `tuple_level_ids`, `tuple_dt_ids`, `tuple_category_ids`), also stacked as `tuple_attr_ids` (`[T, 6]`) | PyG `Data` + tuple fields (see class docstring) | `tests/encoding/test_gnn_conformance.py`, `tests/encoding/test_derived_backends_parity.py` |
| `LiftedTaskEncoder` | Task-structure graphs for task-level predictions (ASG-style, planner-selection-style readouts) | State-independent schema graph: object/predicate/action/parameter nodes with lifted precondition/effect/goal wiring and optional grounded state facts | PyG `Data` (`x_ids` integer channels) | `tests/encoding/test_lifted_task_encoder.py` |
| `ILGEncoder` | Python ILG feature construction | ILG atom/object/action topology and statuses | `BatchEncoding` (hetero) | `examples/encoders/ilg_hetero_encoder_example.py`, `tests/encoding/test_custom_python_encoder_example.py` |
| custom toolkit (`mifrost.encoders.custom`) | Python-only new encoding schemes | User-defined graphs built with `GraphWriter` over planner-neutral `StateView` inputs | User-defined homo graphs via `GraphWriter` | `docs/how-to/write-your-own-encoder.md` |

## Backend Support

All encoders in the table accept Pymimir and PyTyr inputs through a per-instance
runtime. Either planner can be installed independently, or both can coexist in
one process. Compatible outputs from different backends share semantic schema
fingerprints and can be passed together to `mifrost.batch_encodings`.

| Capability | Pymimir | PyTyr |
| --- | :---: | :---: |
| Single and batch encoding | Yes | Yes |
| Stream encoding | Yes | Yes |
| HGraph relation-schema replacement | Yes | Yes |
| Explicit or inferred selection | Yes | Yes |
| Independent wheel/install | Yes | Yes |
| Same-process coexistence | Yes | Yes |
| Mixed compatible `BatchEncoding` / PyG / serialization | Yes | Yes |

`transition_dag_from_rustworkx(...)` remains a Pymimir-returning standalone
helper for backward compatibility. For either backend, passing the raw
`rustworkx.PyDiGraph` directly to the selected `HorizonEncoder` or
`FlatHorizonEncoder` uses that runtime's native conversion path.

## Streams and Dynamic Fields Coverage

- Stream semantics/parity: `tests/encoding/test_stream_encoder.py`
- Flat stream semantics/parity: `tests/encoding/test_flat_stream_encoder.py`
- Dynamic graph fields/collation: `tests/encoding/test_dynamic_graph_fields.py`

## LGAN Surface

- `HGraphEncoder` and `FlatRelationEncoder` share the same main-lane LGAN API:
  `include_lgan_edges`, `lgan_*_edge_pos`, and `lgan_anchor_sources`.
- `HorizonEncoder` and `FlatHorizonEncoder` use candidate state rows as LGAN
  anchors and do not expose `lgan_anchor_sources`.
- Transition encoders on both lanes follow the horizon rule.
