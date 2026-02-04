from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable


@runtime_checkable
class PredicateLike(Protocol):
    def get_name(self) -> str: ...
    def get_arity(self) -> int: ...


@runtime_checkable
class AtomLike(Protocol):
    def get_predicate(self) -> PredicateLike: ...
    def get_terms(self) -> Iterable: ...


@runtime_checkable
class LiteralLike(Protocol):
    def get_atom(self) -> AtomLike: ...
    def get_polarity(self) -> bool: ...


def predicate(atom: AtomLike) -> PredicateLike:
    return atom.get_predicate()


def atom_objects(atom: AtomLike) -> Iterable:
    return atom.get_terms()


def literal_atom(literal: LiteralLike) -> AtomLike:
    return literal.get_atom()


def literal_polarity(literal: LiteralLike) -> bool:
    return literal.get_polarity()


def predicate_name(pred: PredicateLike) -> str:
    return pred.get_name()


def predicate_arity(pred: PredicateLike) -> int:
    return pred.get_arity()


def object_name(obj) -> str:
    return obj.get_name()


def action_name(action) -> str:
    if hasattr(action, "get_name"):
        return action.get_name()
    if hasattr(action, "get_action"):
        return action.get_action().get_name()
    return str(action)


def action_arity(action) -> int:
    if hasattr(action, "get_arity"):
        return action.get_arity()
    if hasattr(action, "get_action"):
        return action.get_action().get_arity()
    return 0


def action_objects(action) -> Iterable:
    if hasattr(action, "get_objects"):
        return action.get_objects()
    if hasattr(action, "get_action"):
        return action.get_action().get_objects()
    return []


def atom_signature(atom: AtomLike) -> tuple[str, tuple[str, ...]]:
    pred = predicate(atom)
    return (
        predicate_name(pred),
        tuple(object_name(obj) for obj in atom_objects(atom)),
    )


def atoms_equal(lhs: AtomLike, rhs: AtomLike) -> bool:
    return atom_signature(lhs) == atom_signature(rhs)
