from __future__ import annotations

import mifrost

from ._helpers import load_problem, print_encoding_summary


def main() -> None:
    _space, domain, problem, state = load_problem("blocks", "small")
    goals = list(problem.get_goal_condition().get_literals())
    encoder = mifrost.ILGEncoder(domain)
    encoding = encoder.encode(state, goals=goals)
    print_encoding_summary(encoding, label="ILGEncoder.encode(state, goals)")


if __name__ == "__main__":
    main()
