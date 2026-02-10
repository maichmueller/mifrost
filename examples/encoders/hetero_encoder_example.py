"""
Example: Encode a PDDL state with HeteroGraphEncoder and plot the graph.

Uses the test PDDL problems bundled in this repository.
"""

from __future__ import annotations

from matplotlib import pyplot as plt

import mifrost
from mifrost.encoders import HGraphEncoder

from _helpers import load_space


def main():
    # Choose a small but meaningful instance
    _, domain, problem = load_space(domain="blocks", problem="medium")
    encoder = HGraphEncoder(domain)

    # Encode the current state; goals default to the problem goal
    graph = encoder.encode(problem.get_initial_state())

    # Plot the produced heterogeneous graph
    encoder.draw(
        graph,
        with_labels=True,
        edge_labels=True,
        label_node_types=[
            mifrost.DEFAULT_SYMBOL_TYPE_ID
        ],  # show object names for clarity
    )
    plt.show()


if __name__ == "__main__":
    main()
