import mifrost
import pymimir
import os


def test_horizon_encoder():
    # 1. Setup Domain
    pddl_dir = "/Users/maichmueller/GitHub/bifrost/tests/pddl"
    domain_path = os.path.join(pddl_dir, "gripper.pddl")
    problem_path = os.path.join(pddl_dir, "gripper-4.pddl")

    # We need a grounded problem for states
    # Using pymimir directly
    config = pymimir.GroundedEvaluator.Config()
    evaluator = pymimir.GroundedEvaluator.from_pddl(domain_path, problem_path, config)
    problem = evaluator.problem
    root_state = problem.initial_state

    # 2. Setup DAG
    dag = mifrost.TransitionDAG(root_state)

    # Add a transition
    actions = list(problem.ground_actions)
    # Find a pick action
    pick_action = next(a for a in actions if "pick" in str(a))

    next_state = evaluator.apply(root_state, pick_action)
    dag.register_transition(
        dag.root_index(), 0, pick_action
    )  # Wait, register_transition needs parent_idx, child_node_idx, action
    # Actually TransitionDAG.register_transition(parent_idx, child_state, action)
    # My C++ implementation: int register_transition(int parent_idx, mimir::search::State child_state, std::optional<mimir::formalism::GroundAction> action = std::nullopt)

    idx1 = dag.register_transition(dag.root_index(), next_state, pick_action)
    print(f"Registered transition to node {idx1}")

    # 3. Setup Encoder
    enc_config = mifrost.HorizonEncoderConfig()
    enc_config.max_goal_level = 1
    encoder = mifrost.HorizonHGraphEncoderEngine(problem.domain, enc_config)

    # 4. Encode
    goals = mifrost.GoalInputs(list(problem.goal_literals))
    parts = encoder.encode(root_state, dag, goals)

    print("Encoded Parts:")
    for k, v in parts.items():
        if isinstance(v, dict):
            print(f"  {k}: {v.keys()}")
        else:
            print(f"  {k}: {len(v)}")

    # Verify we have target nodes
    symbol_names = parts.get("node_names", {}).get("symbol", [])
    target_nodes = [s for s in symbol_names if "target" in s]
    print(f"Target nodes found: {target_nodes}")
    assert len(target_nodes) >= 2

    # Verify edge types
    edge_index = parts.get("edge_index", {})
    parent_edges = [k for k in edge_index.keys() if "parent" in k[1]]
    print(f"Parent edges found: {parent_edges}")

    print("Test passed!")


if __name__ == "__main__":
    test_horizon_encoder()
