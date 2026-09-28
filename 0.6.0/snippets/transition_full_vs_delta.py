from __future__ import annotations

import mifrost

from ._helpers import first_successor, load_problem, print_encoding_summary


def main() -> None:
    space, domain, problem, state = load_problem("blocks", "small")
    goals = list(problem.get_goal_condition().get_literals())
    _action, succ = first_successor(space, state)

    full = mifrost.TransitionHGraphEncoder(domain)
    delta = mifrost.TransitionEffectsHGraphEncoder(domain)

    enc_full = full.encode(state, successor=succ, goals=goals)
    enc_delta = delta.encode(state, successor=succ, goals=goals)

    print_encoding_summary(
        enc_full, label="TransitionHGraphEncoder (full) encode(state, successor)"
    )
    print_encoding_summary(
        enc_delta,
        label="TransitionEffectsHGraphEncoder (delta) encode(state, successor)",
    )


if __name__ == "__main__":
    main()
