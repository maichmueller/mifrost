"""
Example: Encode a PDDL state with ColorGraphEncoder and plot the graph.

Uses the test PDDL problems bundled in this repository.
"""

from __future__ import annotations

from matplotlib import pyplot as plt

from mifrost.encoders import ColorEncoder

from _helpers import load_problem


def main():
    domain, problem, state = load_problem(domain="blocks", problem="probBLOCKS-4-0")
    # Demonstrate edge_features=True so positional/color info appears on edges
    encoder = ColorEncoder(
        domain,
        edge_features=True,
        enable_global_predicate_nodes=True,
    )

    graph = encoder.encode(state)

    # Plot the produced colored graph
    encoder.draw(
        graph,
        with_labels=True,
        edge_labels=True,
    )
    plt.show()


if __name__ == "__main__":
    main()
