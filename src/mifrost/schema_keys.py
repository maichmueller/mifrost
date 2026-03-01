"""Shared schema-key separators and helpers for normalized encoding keys."""

from __future__ import annotations

TYPE_ATTR_SEPARATOR = "/"
EDGE_TYPE_SEPARATOR = "|"

EDGE_INDEX_ATTR_PREFIX = "edge_index_"
EDGE_INDEX_KEY_PREFIX = f"{TYPE_ATTR_SEPARATOR}{EDGE_INDEX_ATTR_PREFIX}"
EDGE_INDEX_SRC_COMPONENT = "0"
EDGE_INDEX_DST_COMPONENT = "1"

PTR_ATTR = "ptr"
BATCH_ATTR = "batch"


def make_type_attr_key(type_key: str, attr: str) -> str:
    """Build `<type>/<attr>` keys used by normalized encoding tensors."""
    return f"{type_key}{TYPE_ATTR_SEPARATOR}{attr}"


def make_edge_type_key(src: str, rel: str, dst: str) -> str:
    """Build `<src>|<rel>|<dst>` edge-type key prefixes."""
    return f"{src}{EDGE_TYPE_SEPARATOR}{rel}{EDGE_TYPE_SEPARATOR}{dst}"


__all__ = [
    "TYPE_ATTR_SEPARATOR",
    "EDGE_TYPE_SEPARATOR",
    "EDGE_INDEX_ATTR_PREFIX",
    "EDGE_INDEX_KEY_PREFIX",
    "EDGE_INDEX_SRC_COMPONENT",
    "EDGE_INDEX_DST_COMPONENT",
    "PTR_ATTR",
    "BATCH_ATTR",
    "make_type_attr_key",
    "make_edge_type_key",
]
