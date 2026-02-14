from __future__ import annotations

import mifrost
from mifrost.graph_fields import DType, GraphFieldSpec, Mode

from ._helpers import first_successor, load_problem, print_encoding_summary, to_py_list


def main() -> None:
    space, domain, _problem, state0 = load_problem("blocks", "smedium")
    _action, state1 = first_successor(space, state0)

    encoder = mifrost.HGraphEncoder(domain)
    encoder.register_graph_fields(
        {
            "goal_distance": GraphFieldSpec(mode=Mode.STACK, dtype=DType.F32),
            "target_indices": GraphFieldSpec(mode=Mode.RAGGED_CAT, dtype=DType.I64),
        }
    )

    g0 = encoder.encode_graph(state0)
    g1 = encoder.encode_graph(state1)
    g0.goal_distance = 0
    g0.target_indices = [0, 1, 2]
    g1.goal_distance = 1
    g1.target_indices = [0]

    batch_enc = encoder.batch_graphs([g0, g1])
    print_encoding_summary(
        batch_enc, label="Dynamic graph fields via encode_graph + batch_graphs"
    )

    payload = batch_enc.as_dict()
    tensors = payload.get("tensors", {})
    goal_distance = tensors.get("__graph__/goal_distance")
    target_indices = tensors.get("__graph__/target_indices")
    target_ptr = tensors.get("__graph__/target_indices/ptr")

    print(
        "graph_field_keys_present:",
        sorted(k for k in tensors.keys() if str(k).startswith("__graph__/")),
    )
    print("goal_distance_head:", to_py_list(goal_distance, max_items=10))
    print("target_indices_head:", to_py_list(target_indices, max_items=10))
    print("target_indices_ptr:", to_py_list(target_ptr, max_items=10))


if __name__ == "__main__":
    main()
