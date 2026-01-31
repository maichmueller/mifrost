import mifrost
import pymimir
from pymimir.wrapper_formalism import Domain, Problem
from pathlib import Path
import os


def test_horizon_encoder():
    # 1. Setup Domain & Problem
    root = Path(__file__).resolve().parents[2]
    domain_path = root / "data" / "pddl" / "blocks" / "domain.pddl"
    problem_path = root / "data" / "pddl" / "blocks" / "probBLOCKS-4-0.pddl"

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

    parts = encoder.encode(state._advanced_state, dag, goals)

    print("Encoded Parts Keys:", parts.keys())

    # Verify we have target nodes
    node_names = parts.get("node_names", {})
    symbol_names = node_names.get(enc_config.symbol_type_id, [])
    target_nodes = [s for s in symbol_names if "target" in s]
    print(f"Target nodes found: {target_nodes}")
    assert len(target_nodes) >= 1  # At least root target

    # Verify edge types
    edge_index = parts.get("edge_index", {})
    parent_edges = [k for k in edge_index.keys() if "parent" in k[1]]
    print(f"Parent edges found: {parent_edges}")

    print("Test passed!")


if __name__ == "__main__":
    test_horizon_encoder()
