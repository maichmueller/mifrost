"""
Example: Encode a PDDL state with ColorGraphEncoder and plot the graph.

Uses the test PDDL problems bundled in this repository.
"""

from __future__ import annotations

from mifrost.encoders import ColorEncoder

from _helpers import begin_plot, load_problem, save_plot


def main():
    domain, problem, state = load_problem(domain="blocks", problem="smedium")
    # Demonstrate edge_features=True so positional/color info appears on edges
    encoder = ColorEncoder(
        domain,
        edge_features=True,
        enable_global_predicate_nodes=True,
    )

    graph = encoder.encode_pyg(state)

    # Plot the produced colored graph
    begin_plot(figsize=(14, 10))
    encoder.draw(
        graph,
        with_labels=True,
        edge_labels=True,
    )
    out_path = save_plot("color_encoder.png")
    print(f"Saved color encoder plot to {out_path}")


if __name__ == "__main__":
    main()
