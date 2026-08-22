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
| `StarGraphEncoder` | Vanilla-GNN state encoding with reified atoms | Object nodes plus fact/goal/action anchor nodes joined by position-aware star edges | PyG `Data` (`x_ids` integer channels) | `examples/encoders/derived_graph_example.py`, `tests/encoding/test_gnn_conformance.py`, `tests/encoding/test_derived_facades.py` |
| `ObjectGraphEncoder` | Compact objects-only projections (`clique` / `chain` / `star_first`) | Directed argument-pair edges between objects; no atom nodes materialized | PyG `Data` (`x_ids` integer channels) | `examples/encoders/derived_graph_example.py`, `tests/encoding/test_gnn_conformance.py`, `tests/encoding/test_derived_facades.py` |
| `AtomLineGraphEncoder` | Atom co-occurrence structure over the star universe | Star view plus fact-fact `line_share` edges bounded by `line_graph_max_degree` | PyG `Data` (`x_ids` integer channels) | `tests/encoding/test_gnn_conformance.py`, `tests/encoding/test_derived_facades.py` |
| `HypergraphIncidenceEncoder` | Native hypergraph layers (PyG `HypergraphConv`) | Literal instances as hyperedges; incidence restacked into `hyperedge_index` / `hyperedge_attr_ids` | PyG `Data` + hyperedge fields | `tests/encoding/test_gnn_conformance.py`, `tests/encoding/test_derived_facades.py` |
| `TransformerBiasEncoder` | Transformer attention-bias inputs from planning structure | Objects-only clique projection plus sparse shortest-path fields `spd_src` / `spd_dst` / `spd_dist` | PyG `Data` + spd fields | `tests/encoding/test_gnn_conformance.py`, `tests/encoding/test_derived_facades.py` |
| `TupleTensorEncoder` | Padded-free tuple views for sequence-style consumers | CSR tuple view (`tuple_args` / `tuple_ptr` / `tuple_rel_ids`) on the derived-graph carrier | PyG `Data` + tuple fields (see class docstring) | see class docstring |
| `ILGEncoder` | Python ILG feature construction | ILG atom/object/action topology and statuses | `BatchEncoding` (hetero) | `examples/encoders/ilg_hetero_encoder_example.py`, `tests/encoding/test_custom_python_encoder_example.py` |

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
