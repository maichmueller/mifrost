"""Pure-Python composable encoder toolkit for mifrost.

This package lets Python-only users define planning-graph encoders without
writing any C++: a `StateView` exposes a pymimir problem or pytyr task
through planner-neutral records (`Atom`, `Literal`, `PredicateInfo`,
`ActionInfo`), subclasses of `CustomGraphEncoder` draw each state into a
`GraphWriter` using the interning helpers in `tables` (Vocabulary, NodeTable,
EdgeSink), and batching plus streaming come free from the shared encoder
machinery. See the custom-encoder how-to guide in the project documentation
and the bundled showcase encoder for worked examples.
"""

from .base import CustomGraphEncoder, CustomStream
from .harness import assert_backend_parity, channel_summary, conformance_smoke
from .state_view import (
    ActionInfo,
    ActionStructure,
    Atom,
    Effect,
    Literal,
    PredicateInfo,
    StateView,
)
from .tables import EdgeSink, NodeTable, Vocabulary
from .writer import GraphWriter

__all__ = [
    "ActionInfo",
    "ActionStructure",
    "Atom",
    "CustomGraphEncoder",
    "CustomStream",
    "EdgeSink",
    "Effect",
    "GraphWriter",
    "Literal",
    "NodeTable",
    "PredicateInfo",
    "StateView",
    "Vocabulary",
    "assert_backend_parity",
    "channel_summary",
    "conformance_smoke",
]
