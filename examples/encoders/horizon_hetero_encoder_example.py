"""HorizonEncoder drawing example with a few useful config variants."""

from __future__ import annotations

from collections import deque

import pymimir
from matplotlib import pyplot as plt

import mifrost
from mifrost.encoders import HorizonEncoder
from mifrost.encoders.types import to_advanced_action, to_advanced_state
from _helpers import begin_plot, load_problem, save_plot


def _build_dag(
    problem: pymimir.Problem, *, max_depth: int = 2, max_branch: int = 2
) -> tuple[pymimir.State, mifrost.TransitionDAG]:
    """Build a compact bounded tree horizon for visualization."""
    root = problem.get_initial_state()
    state_space, _ = pymimir.advanced.datasets.StateSpace.create(
        problem._search_context, pymimir.advanced.datasets.StateSpaceOptions()
    )
    space = pymimir.wrapper_datasets.StateSpaceSampler(
        pymimir.advanced.datasets.StateSpaceSampler(state_space), problem
    )

    adv_root = getattr(root, "_advanced_state", root)
    dag = mifrost.TransitionDAG(adv_root)
    queue: deque[tuple[pymimir.State, int]] = deque([(root, 0)])
    seen = {root}
    while queue:
        state, depth = queue.popleft()
        if depth >= max_depth:
            continue
        transitions = list(space.get_forward_transitions(state))
        for action, successor in transitions[:max_branch]:
            if successor in seen:
                continue
            dag.register_transition(
                to_advanced_state(state),
                to_advanced_state(successor),
                to_advanced_action(action),
            )
            seen.add(successor)
            queue.append((successor, depth + 1))
    return root, dag


def main() -> None:
    domain, problem, _ = load_problem(domain="blocks", problem="medium")
    root, dag = _build_dag(problem, max_depth=2, max_branch=3)
    goals = list(problem.get_goal_condition().get_literals())

    variants = [
        ("full", mifrost.HorizonEncoderMode.Full),
        ("delta", mifrost.HorizonEncoderMode.Delta),
    ]
    relation_flags = [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ]

    for mode_name, mode in variants:
        for use_parent, use_sibling, use_cousin in relation_flags:
            fig = begin_plot(figsize=(16, 12))
            ax = fig.add_subplot(1, 1, 1)
            encoder = HorizonEncoder(
                domain,
                transition_mode=mode,
                ignore_actions=False,
                enable_parent_relation=use_parent,
                enable_sibling_relation=use_sibling,
                enable_cousin_relation=use_cousin,
            )
            data = encoder.encode_pyg(root, dag=dag, goals=goals)
            relation_label = (
                f"P={'on' if use_parent else 'off'} "
                f"S={'on' if use_sibling else 'off'} "
                f"C={'on' if use_cousin else 'off'}"
            )
            encoder.draw(
                data,
                ax=ax,
                with_labels=True,
                edge_labels=True,
                label_node_types=[encoder.symbol_type_id],
                align_target_nodes=True,
                target_x_spacing=4.0,
                target_y_spacing=2.0,
            )
            plt.title(f"HorizonEncoder mode={mode_name} {relation_label}")
            filename = (
                f"horizon_encoder_{mode_name}_"
                f"p{int(use_parent)}_s{int(use_sibling)}_c{int(use_cousin)}.png"
            )
            out_path = save_plot(filename)
            print(f"Saved horizon plot to {out_path}")


if __name__ == "__main__":
    main()
