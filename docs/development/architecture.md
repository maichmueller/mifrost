# Architecture Notes

This project keeps encoder layering explicit:

- `HGraphEncoderEngine` owns shared heterogeneous encoding lifecycle.
- `HorizonHGraphEncoderEngine` and `SuccessorHGraphEncoderEngine` own specialized graph semantics.
- Python wrappers keep a native-first API and explicit conversion utilities.

Reference design notes in source tree:

- `src/ENCODER_INHERITANCE_ARCHITECTURE.md`
- `src/STREAM_ENCODER_PLAN.md`
- `src/dynamic_graph_fields_plan.md`
