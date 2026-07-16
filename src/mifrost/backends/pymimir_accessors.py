from __future__ import annotations

from collections.abc import Iterable

import pymimir.advanced.formalism as af
import pymimir.wrapper_formalism as wf

from .pymimir_types import (
    ATOM_TYPES,
    GROUND_ACTION_TYPES,
    GOAL_LITERAL_TYPES,
    OBJECT_TYPES,
    PREDICATE_TYPES,
    GroundAtomInput,
    GroundActionInput,
    GoalLiteralInput,
    ObjectInput,
    PredicateInput,
)


def predicate(atom: GroundAtomInput) -> PredicateInput:
    """Return an atom's predicate object."""
    if not isinstance(atom, ATOM_TYPES):
        raise TypeError(f"Expected ground atom, got {type(atom)!r}")
    return atom.get_predicate()


def atom_objects(atom: GroundAtomInput) -> Iterable[ObjectInput]:
    """Return ordered object terms for a ground atom."""
    if isinstance(atom, wf.GroundAtom):
        return atom.get_terms()
    if isinstance(
        atom, (af.StaticGroundAtom, af.FluentGroundAtom, af.DerivedGroundAtom)
    ):
        return atom.get_objects()
    raise TypeError(f"Expected ground atom, got {type(atom)!r}")


def literal_atom(literal: GoalLiteralInput) -> GroundAtomInput:
    """Return the atom wrapped by a ground literal."""
    if not isinstance(literal, GOAL_LITERAL_TYPES):
        raise TypeError(f"Expected ground literal, got {type(literal)!r}")
    return literal.get_atom()


def literal_polarity(literal: GoalLiteralInput) -> bool:
    """Return literal polarity (True=positive, False=negative)."""
    if not isinstance(literal, GOAL_LITERAL_TYPES):
        raise TypeError(f"Expected ground literal, got {type(literal)!r}")
    return literal.get_polarity()


def predicate_name(pred: PredicateInput) -> str:
    """Return predicate name."""
    if not isinstance(pred, PREDICATE_TYPES):
        raise TypeError(f"Expected predicate, got {type(pred)!r}")
    return pred.get_name()


def predicate_arity(pred: PredicateInput) -> int:
    """Return predicate arity."""
    if not isinstance(pred, PREDICATE_TYPES):
        raise TypeError(f"Expected predicate, got {type(pred)!r}")
    return pred.get_arity()


def object_name(obj: ObjectInput) -> str:
    """Return object name."""
    if not isinstance(obj, OBJECT_TYPES):
        raise TypeError(f"Expected object, got {type(obj)!r}")
    return obj.get_name()


def action_name(action: GroundActionInput) -> str:
    """Return action schema name for a grounded action."""
    if not isinstance(action, GROUND_ACTION_TYPES):
        raise TypeError(f"Expected ground action, got {type(action)!r}")
    if isinstance(action, wf.GroundAction):
        return action.get_action().get_name()
    return action.get_action().get_name()


def action_arity(action: GroundActionInput) -> int:
    """Return action schema arity for a grounded action."""
    if not isinstance(action, GROUND_ACTION_TYPES):
        raise TypeError(f"Expected ground action, got {type(action)!r}")
    return action.get_action().get_arity()


def action_objects(action: GroundActionInput) -> Iterable[ObjectInput]:
    """Return ordered action argument objects for a grounded action."""
    if not isinstance(action, GROUND_ACTION_TYPES):
        raise TypeError(f"Expected ground action, got {type(action)!r}")
    return action.get_objects()


def atom_signature(atom: GroundAtomInput) -> tuple[str, tuple[str, ...]]:
    """Return canonical atom signature used for equality checks."""
    pred = predicate(atom)
    return (
        predicate_name(pred),
        tuple(object_name(obj) for obj in atom_objects(atom)),
    )


def atoms_equal(lhs: GroundAtomInput, rhs: GroundAtomInput) -> bool:
    """Compare atoms by predicate name and ordered object names."""
    return atom_signature(lhs) == atom_signature(rhs)
