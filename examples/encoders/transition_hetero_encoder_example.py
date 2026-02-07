"""
Example: Encode a (state, successor) pair with TransitionHeteroGraphEncoder and plot the graph.

Uses the test PDDL problems bundled in this repository.
"""

from __future__ import annotations

from matplotlib import pyplot as plt

import mifrost
from mifrost.encoders import TransitionHGraphEncoder

from _helpers import first_transition, load_space


def main():
    space, domain, problem = load_space(domain="gripper", problem="gripper_b-1")
    state = problem.get_initial_state()
    # pick a meaningful single successor via a greedy step
    _, successor = first_transition(space, state)

    encoder = TransitionHGraphEncoder(domain)
    graph = encoder.encode(state=state, successor=successor)

    # Plot the transition-augmented heterogeneous graph
    encoder.draw(
        graph,
        with_labels=True,
        edge_labels=True,
        label_node_types=[mifrost.DEFAULT_SYMBOL_TYPE_ID],
    )
    plt.show()


if __name__ == "__main__":
    main()
