"""HorizonEncoder drawing example with a few useful config variants."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import pymimir
from matplotlib import pyplot as plt

import mifrost
from mifrost.encoders import HorizonEncoder
from mifrost.encoders.types import to_advanced_state


def _load_problem() -> tuple[pymimir.Domain, pymimir.Problem]:
    repo_root = Path(__file__).resolve().parents[2]
    domain_path = repo_root / "data" / "pddl" / "blocks" / "domain.pddl"
    problem_path = repo_root / "data" / "pddl" / "blocks" / "medium.pddl"
    domain = pymimir.Domain(domain_path)
    problem = pymimir.Problem(domain, problem_path, mode="lifted")
    return domain, problem


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
        for _, successor in transitions[:max_branch]:
            if successor in seen:
                continue
            dag.register_transition(
                to_advanced_state(state),
                to_advanced_state(successor),
            )
            seen.add(successor)
            queue.append((successor, depth + 1))
    return root, dag


def main() -> None:
    domain, problem = _load_problem()
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

    for name, mode in variants:
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        fig.suptitle(f"HorizonEncoder mode={name}", fontsize=12)
        for ax, (use_parent, use_sibling, use_cousin) in zip(
            axes.flat, relation_flags, strict=True
        ):
            encoder = HorizonEncoder(
                domain,
                transition_mode=mode,
                enable_parent_relation=use_parent,
                enable_sibling_relation=use_sibling,
                enable_cousin_relation=use_cousin,
            )
            data = encoder.encode_pyg(root, dag=dag, goals=goals)
            label = (
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
            ax.set_title(label, fontsize=10)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
