"""FlatRelationEncoder inspection example with synthetic relation-instance nodes."""

from __future__ import annotations

import mifrost
from mifrost.encoders import FlatRelationEncoder

from _helpers import begin_plot, false_problem_literals, load_problem, save_plot


def main() -> None:
    domain, problem, state = load_problem(domain="blocks", problem="small")
    subgoals = false_problem_literals(problem, state)
    encoder = FlatRelationEncoder(
        domain,
        max_goal_level=1,
        target_sources=[mifrost.TargetSource.subgoals],
    )
    data = encoder.encode_pyg(state, subgoal_layers=[subgoals])
    graph = encoder.to_networkx(data)

    print("schema:", data.schema)
    print("subgoals:", len(subgoals))
    print("target groups:", list(data.target_groups))
    print(
        "subgoal target entities:", data.graph_target_entity_names(0, group="subgoal")
    )
    print("relation counts:", data.relation_instance_counts_total().tolist())
    print("nodes:", graph.number_of_nodes(), "edges:", graph.number_of_edges())

    try:
        from matplotlib import pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not installed; skipped plot export")
        return

    fig = begin_plot(figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1)
    encoder.draw(data, ax=ax, with_labels=True, edge_labels=True)
    plt.title("FlatRelationEncoder graph with false atoms as subgoal targets")
    out_path = save_plot("flat_relation_encoder.png")
    print(f"Saved flat relation encoder plot to {out_path}")


if __name__ == "__main__":
    main()
