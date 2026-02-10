from pathlib import Path

import mifrost
import pymimir
import pytest
from pymimir.wrapper_formalism import Domain, Problem


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
def test_horizon_encoder(domain_name: str, problem_name: str):
    # 1. Setup Domain & Problem
    root = Path(__file__).resolve().parents[2]
    domain_path = root / "data" / "pddl" / domain_name / "domain.pddl"
    problem_path = root / "data" / "pddl" / domain_name / f"{problem_name}.pddl"

    print(f"Loading domain from {domain_path}")
    domain = Domain(domain_path)
    # Using 'grounded' mode to allow applying actions easily
    problem = Problem(domain, problem_path, mode="grounded")
    state = problem.get_initial_state()

    # 2. Setup DAG
    dag = mifrost.TransitionDAG(state._advanced_state)

    # Add a real transition if possible
    applicable = state.generate_applicable_actions()
    if applicable:
        action = applicable[0]
        next_state = action.apply(state)
        dag.register_transition(
            dag.root(), next_state._advanced_state, action._advanced_ground_action
        )
        print(f"Registered transition with action {action}")

    # 3. Setup Encoder
    enc_config = mifrost.HorizonEncoderConfig()
    enc_config.max_goal_level = 1
    encoder = mifrost.HorizonHGraphEncoderEngine(domain._advanced_domain, enc_config)

    # 4. Encode
    goal_cond = problem.get_goal_condition()
    static_goals = [
        l._advanced_ground_literal for l in goal_cond._static_ground_literals
    ]
    fluent_goals = [
        l._advanced_ground_literal for l in goal_cond._fluent_ground_literals
    ]
    derived_goals = [
        l._advanced_ground_literal for l in goal_cond._derived_ground_literals
    ]

    goals = mifrost.GoalInputs(static_goals + fluent_goals + derived_goals)

    encoding = encoder.encode(state._advanced_state, dag, goals)
    data = encoding.as_pyg(as_batch=False)

    # Verify we have target nodes
    symbol_names = list(getattr(data[enc_config.symbol_type_id], "node_names", []))
    target_nodes = [s for s in symbol_names if "target" in s]
    print(f"Target nodes found: {target_nodes}")
    assert len(target_nodes) >= 1  # At least root target

    # Verify graph connectivity is present.
    edge_types = list(data.edge_types)
    print(f"Edge types found: {edge_types}")
    assert len(edge_types) >= 1

    print("Test passed!")


if __name__ == "__main__":
    for domain_name, problem_name in DOMAIN_CASES:
        test_horizon_encoder(domain_name, problem_name)
