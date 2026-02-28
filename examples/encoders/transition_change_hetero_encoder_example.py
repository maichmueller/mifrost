"""
Example: Encode a (state, successor) pair with TransitionChangeHeteroGraphEncoder and plot the graph.

Only the changed literals between the states are represented for the successor.
Uses the test PDDL problems bundled in this repository.
"""

from __future__ import annotations

import mifrost
from mifrost.encoders import TransitionEffectsHGraphEncoder

from _helpers import begin_plot, first_transition, load_space, save_plot


def main():
    space, domain, problem = load_space(domain="blocks", problem="smedium")
    state = problem.get_initial_state()
    _, successor = first_transition(space, state)

    encoder = TransitionEffectsHGraphEncoder(domain)
    graph = encoder.encode_pyg(state=state, successor=successor)

    # Plot the transition-delta heterogeneous graph
    fig = begin_plot(figsize=(14, 10))
    ax = fig.add_subplot(111)
    encoder.draw(
        graph,
        ax=ax,
        with_labels=True,
        edge_labels=True,
        label_node_types=[mifrost.DEFAULT_SYMBOL_TYPE_ID],
    )
    out_path = save_plot("transition_change_hetero_encoder.png")
    print(f"Saved transition effects encoder plot to {out_path}")


if __name__ == "__main__":
    main()
