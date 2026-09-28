from __future__ import annotations

import mifrost

from ._helpers import first_successor, load_problem, print_encoding_summary


def main() -> None:
    space, domain, _problem, state = load_problem("blocks", "smedium")
    _action, succ = first_successor(space, state)
    encoder = mifrost.HGraphEncoder(domain)
    encoding = encoder.encode_batch([state, succ])
    print_encoding_summary(
        encoding, label="HGraphEncoder.encode_batch([state, successor])"
    )


if __name__ == "__main__":
    main()
