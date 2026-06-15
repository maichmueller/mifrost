from __future__ import annotations

import mifrost

from ._helpers import load_problem, print_encoding_summary


def main() -> None:
    _space, domain, _problem, state = load_problem("blocks", "smedium")
    encoder = mifrost.HGraphEncoder(domain)
    encoding = encoder.encode(state)
    print_encoding_summary(encoding, label="HGraphEncoder.encode(initial_state)")


if __name__ == "__main__":
    main()
