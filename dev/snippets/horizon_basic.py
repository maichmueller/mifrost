from __future__ import annotations

import mifrost

from ._helpers import (
    _as_advanced_action,
    _as_advanced_state,
    load_problem,
    print_encoding_summary,
)


def main() -> None:
    space, domain, problem, root = load_problem("blocks", "small")
    goals = list(problem.get_goal_condition().get_literals())

    transitions = list(space.get_forward_transitions(root))[:3]
    if not transitions:
        raise RuntimeError("No transitions available for horizon example")

    adv_root = _as_advanced_state(root)
    dag = mifrost.TransitionDAG(adv_root)
    for action, target in transitions:
        if target is None:
            continue
        dag.register_transition(
            adv_root, _as_advanced_state(target), _as_advanced_action(action)
        )

    encoder = mifrost.HorizonEncoder(domain)
    encoding = encoder.encode(root, dag=dag, goals=goals)
    print_encoding_summary(encoding, label="HorizonEncoder.encode(root, dag, goals)")


if __name__ == "__main__":
    main()
