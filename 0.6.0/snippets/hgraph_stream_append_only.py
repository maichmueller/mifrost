from __future__ import annotations

import mifrost

from ._helpers import first_successor, load_problem, print_encoding_summary


def main() -> None:
    space, domain, _problem, state = load_problem("blocks", "smedium")
    _action, succ = first_successor(space, state)

    encoder = mifrost.HGraphEncoder(domain)
    stream = encoder.stream()
    stream.append(state)
    stream.append(succ)
    encoding = stream.flush()

    print_encoding_summary(
        encoding, label="HGraphEncoder.stream().append(...).flush() (2 graphs)"
    )


if __name__ == "__main__":
    main()
