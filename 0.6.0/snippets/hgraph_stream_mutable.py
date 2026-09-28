from __future__ import annotations

import mifrost

from ._helpers import first_successor, load_problem, print_encoding_summary


def main() -> None:
    space, domain, _problem, state = load_problem("blocks", "smedium")
    _action, succ = first_successor(space, state)

    encoder = mifrost.HGraphEncoder(domain)
    mstream = encoder.mutable_stream()
    sid0 = mstream.append(state)
    sid1 = mstream.append(succ)
    mstream.update(sid0, succ)
    mstream.remove(sid1)
    encoding = mstream.flush()

    print_encoding_summary(
        encoding,
        label="HGraphEncoder.mutable_stream() append/update/remove/flush (1 graph)",
    )


if __name__ == "__main__":
    main()
