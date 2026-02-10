"""
Example: Encode a PDDL state with ILGHeteroGraphEncoder and plot the graph.

Uses the test PDDL problems bundled in this repository.
"""

from __future__ import annotations

from matplotlib import pyplot as plt

from mifrost.encoders import HGraphEncoder, ILGEncoder

from _helpers import load_space


def main():
    _, domain, problem = load_space(domain="blocks", problem="medium")
    encoder = ILGEncoder(domain)

    graph = encoder.encode_pyg(problem.get_initial_state(), goals=[])
    drawer = HGraphEncoder(domain)

    # Plot the ILG heterogeneous graph
    drawer.draw(
        graph,
        with_labels=True,
        edge_labels=True,
        label_node_types=[encoder.symbol_type_id],
    )
    plt.show()


if __name__ == "__main__":
    main()
