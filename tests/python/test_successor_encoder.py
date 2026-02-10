import os

import mifrost
import pymimir
import pytest


DOMAIN_CASES = [
    ("blocks", "smedium"),
    ("gripper", "gripper_b-5"),
    ("delivery", "instance_2x2_p-2_0"),
]


@pytest.mark.parametrize(
    ("domain_name", "problem_name"),
    DOMAIN_CASES,
    ids=[f"{domain}:{problem}" for domain, problem in DOMAIN_CASES],
)
def test_successor_encoder(domain_name: str, problem_name: str):
    # 1. Load PDDL
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    domain_path = os.path.join(root, "data", "pddl", domain_name, "domain.pddl")
    problem_path = os.path.join(
        root, "data", "pddl", domain_name, f"{problem_name}.pddl"
    )

    print(f"Loading domain from {domain_path}")
    domain = pymimir.Domain(domain_path)
    # Using 'grounded' mode to allow applying actions easily
    problem = pymimir.Problem(domain, problem_path, mode="grounded")

    # 2. Get initial state
    state = problem.get_initial_state()

    # 3. Create a successor state
    # Use brfs to find a goal state as a successor
    print("Running BRFS to find a successor...")
    result = pymimir.brfs(problem, state)
    assert result.status == "solved"
    successor = result.goal_state
    print(f"Found successor (goal state) with index {successor.get_index()}")

    # Also find an immediate successor if possible for a smaller delta
    applicable = state.generate_applicable_actions()
    if applicable:
        # If we can't apply, we just use the goal state as 'successor'
        # To make it a 'real' successor test
        pass

    # 4. Setup goals
    goal_cond = problem.get_goal_condition()
    goals = [
        l._advanced_ground_literal
        for l in goal_cond._static_ground_literals
        + goal_cond._fluent_ground_literals
        + goal_cond._derived_ground_literals
    ]
    goal_inputs = mifrost.GoalInputs(goals)

    # 5. Full Mode Test
    config_full = mifrost.SuccessorEncoderConfig()
    config_full.successor_mode = mifrost.SuccessorEncoderMode.Full
    config_full.successor_suffix = "[suc]"

    encoder_full = mifrost.SuccessorHGraphEncoderEngine(
        domain._advanced_domain, config_full
    )
    data_full = encoder_full.encode(
        state._advanced_state, successor._advanced_state, goal_inputs
    ).as_pyg(as_batch=False)
    node_types = list(data_full.node_types)
    print(f"Node types (Full): {node_types}")

    # Check for [suc] nodes
    suc_nodes = [t for t in node_types if "[suc]" in t]
    assert len(suc_nodes) > 0, "No successor nodes found in Full mode"
    print(f"Successor nodes (Full): {suc_nodes}")

    # 6. Delta Mode Test
    config_delta = mifrost.SuccessorEncoderConfig()
    config_delta.successor_mode = mifrost.SuccessorEncoderMode.Delta
    config_delta.successor_suffix = "[suc]"

    encoder_delta = mifrost.SuccessorHGraphEncoderEngine(
        domain._advanced_domain, config_delta
    )
    data_delta = encoder_delta.encode(
        state._advanced_state, successor._advanced_state, goal_inputs
    ).as_pyg(as_batch=False)
    node_types_delta = list(data_delta.node_types)
    print(f"Node types (Delta): {node_types_delta}")

    suc_nodes_delta = [t for t in node_types_delta if "[suc]" in t]
    assert len(suc_nodes_delta) > 0, "No successor nodes found in Delta mode"
    print(f"Successor nodes (Delta): {suc_nodes_delta}")

    # In Delta mode, we expect at least one positive and potentially one negative if we unstack something
    # But blocks often just has additions.

    print("Test passed!")


if __name__ == "__main__":
    for domain_name, problem_name in DOMAIN_CASES:
        test_successor_encoder(domain_name, problem_name)
