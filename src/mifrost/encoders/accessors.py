"""Backward-compatible imports for Pymimir-specific object accessors."""

from ..backends.pymimir_accessors import (
    action_arity,
    action_name,
    action_objects,
    atom_objects,
    atom_signature,
    atoms_equal,
    literal_atom,
    literal_polarity,
    object_name,
    predicate,
    predicate_arity,
    predicate_name,
)

__all__ = [
    "action_arity",
    "action_name",
    "action_objects",
    "atom_objects",
    "atom_signature",
    "atoms_equal",
    "literal_atom",
    "literal_polarity",
    "object_name",
    "predicate",
    "predicate_arity",
    "predicate_name",
]
