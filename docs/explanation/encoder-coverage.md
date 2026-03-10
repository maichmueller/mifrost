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
| `ILGEncoder` | Python ILG feature construction | ILG atom/object/action topology and statuses | `BatchEncoding` (hetero) | `examples/encoders/ilg_hetero_encoder_example.py`, `tests/encoding/test_custom_python_encoder_example.py` |

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
