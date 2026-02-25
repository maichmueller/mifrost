"""
Example: Encode a (state, successor) pair with TransitionHeteroGraphEncoder and plot the graph.

Uses the test PDDL problems bundled in this repository.
"""

from __future__ import annotations

import mifrost
from mifrost.encoders import TransitionHGraphEncoder

from _helpers import begin_plot, first_transition, load_space, save_plot


def main():
    space, domain, problem = load_space(domain="gripper", problem="gripper_b-5")
    state = problem.get_initial_state()
    # pick a meaningful single successor via a greedy step
    _, successor = first_transition(space, state)

    encoder = TransitionHGraphEncoder(domain)
    graph = encoder.encode_pyg(state=state, successor=successor)

    # Plot the transition-augmented heterogeneous graph
    begin_plot(figsize=(14, 10))
    encoder.draw(
        graph,
        with_labels=True,
        edge_labels=True,
        label_node_types=[mifrost.DEFAULT_SYMBOL_TYPE_ID],
    )
    out_path = save_plot("transition_hetero_encoder.png")
    print(f"Saved transition encoder plot to {out_path}")


if __name__ == "__main__":
    main()
