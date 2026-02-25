"""
Example: Encode a PDDL state with HeteroGraphEncoder and plot the graph.

Uses the test PDDL problems bundled in this repository.
"""

from __future__ import annotations

import mifrost
from mifrost.encoders import HGraphEncoder

from _helpers import begin_plot, first_transition, load_space, save_plot


def main():
    # Choose a small but meaningful instance
    space, domain, problem = load_space(domain="blocks", problem="medium")
    state = problem.get_initial_state()
    action, _ = first_transition(space, state)

    variants = [
        ("state_only", True, None),
        ("state_with_action", False, [action]),
    ]
    for name, ignore_actions, actions in variants:
        begin_plot(figsize=(14, 10))
        encoder = HGraphEncoder(domain, ignore_actions=ignore_actions)
        graph = encoder.encode_pyg(state, actions=actions)

        # Plot the produced heterogeneous graph
        encoder.draw(
            graph,
            with_labels=True,
            edge_labels=True,
            label_node_types=[
                mifrost.DEFAULT_SYMBOL_TYPE_ID
            ],  # show object names for clarity
        )
        out_path = save_plot(f"hetero_encoder_{name}.png")
        print(f"Saved {name} plot to {out_path}")


if __name__ == "__main__":
    main()
