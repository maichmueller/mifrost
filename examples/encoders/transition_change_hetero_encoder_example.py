"""
Example: Encode a (state, successor) pair with TransitionChangeHeteroGraphEncoder and plot the graph.

Only the changed literals between the states are represented for the successor.
Uses the test PDDL problems bundled in this repository.
"""

from __future__ import annotations

from matplotlib import pyplot as plt

import mifrost
from mifrost.encoders import TransitionEffectsHGraphEncoder

from _helpers import first_transition, load_space


def main():
    space, domain, problem = load_space(domain="blocks", problem="smedium")
    state = problem.get_initial_state()
    _, successor = first_transition(space, state)

    encoder = TransitionEffectsHGraphEncoder(domain)
    graph = encoder.encode(state=state, successor=successor)

    # Plot the transition-delta heterogeneous graph
    encoder.draw(
        graph,
        with_labels=True,
        edge_labels=True,
        label_node_types=[mifrost.DEFAULT_SYMBOL_TYPE_ID],
    )
    plt.show()


if __name__ == "__main__":
    main()
