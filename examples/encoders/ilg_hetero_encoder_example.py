"""
Example: Encode a PDDL state with ILGHeteroGraphEncoder and plot the graph.

Uses the test PDDL problems bundled in this repository.
"""

from __future__ import annotations

from mifrost.encoders import HGraphEncoder, ILGEncoder

from _helpers import begin_plot, load_space, save_plot


def main():
    _, domain, problem = load_space(domain="blocks", problem="medium")
    encoder = ILGEncoder(domain)

    graph = encoder.encode_pyg(problem.get_initial_state(), goals=[])
    drawer = HGraphEncoder(domain)

    # Plot the ILG heterogeneous graph
    fig = begin_plot(figsize=(14, 10))
    ax = fig.add_subplot(111)
    drawer.draw(
        graph,
        ax=ax,
        with_labels=True,
        edge_labels=True,
        label_node_types=[encoder.symbol_type_id],
    )
    out_path = save_plot("ilg_hetero_encoder.png")
    print(f"Saved ILG encoder plot to {out_path}")


if __name__ == "__main__":
    main()
