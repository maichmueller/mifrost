"""Custom flat encoder example that adds dense target-row annotations."""

from __future__ import annotations

from _custom_flat_target_mask_encoder import ExampleTargetMaskFlatEncoder

from _helpers import first_transition, load_space


def main() -> None:
    space, domain, problem = load_space(domain="blocks", problem="smedium")
    state = problem.get_initial_state()
    action, _ = first_transition(space, state)

    encoder = ExampleTargetMaskFlatEncoder(domain)
    data = encoder.encode_pyg(state, actions=[action])

    print("relation schema:", data.schema)
    print("target entity names:", data.graph_target_entity_names(0))
    print("entity_is_target:", data.entity_is_target.tolist())
    print("target_entity_count:", data.target_entity_count.tolist())


if __name__ == "__main__":
    main()
