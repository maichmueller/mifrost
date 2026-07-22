from __future__ import annotations

import mifrost
from mifrost.graph_fields import DType, Mode

from ._helpers import first_successor, load_problem, print_encoding_summary, to_py_list


def main() -> None:
    space, domain, _problem, state0 = load_problem("blocks", "smedium")
    _action, state1 = first_successor(space, state0)

    encoder = mifrost.HGraphEncoder(domain)

    enc0 = encoder.encode(state0)
    enc1 = encoder.encode(state1)
    enc0.goal_distance = 0
    enc0.target_indices = [0, 1, 2]
    enc1.goal_distance = 1
    enc1.target_indices = [0]

    batch_enc = mifrost.batch_encodings(
        [enc0, enc1],
        collate_spec={
            "goal_distance": {"mode": Mode.STACK.value, "dtype": DType.F32.value},
            "target_indices": {
                "mode": Mode.RAGGED_CAT.value,
                "dtype": DType.I64.value,
            },
        },
    )
    print_encoding_summary(
        batch_enc, label="Dynamic attrs via encode + batch_encodings(collate_spec=...)"
    )

    payload = batch_enc.as_dict()
    tensors = payload.get("tensors", {})
    print(
        "native_graph_field_keys_present:",
        sorted(k for k in tensors.keys() if str(k).startswith("__graph__/")),
    )
    print("goal_distance:", to_py_list(batch_enc.goal_distance, max_items=10))
    print("target_indices:", to_py_list(batch_enc.target_indices, max_items=10))
    print("target_indices_ptr:", to_py_list(batch_enc.target_indices_ptr, max_items=10))
    print("collate_spec:", batch_enc.collate_spec())


if __name__ == "__main__":
    main()
