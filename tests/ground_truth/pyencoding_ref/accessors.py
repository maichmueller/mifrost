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


def predicate(atom: AtomLike) -> PredicateLike:
    return atom.get_predicate()


def atom_objects(atom: AtomLike) -> Iterable:
    return atom.get_terms()


def literal_atom(literal: LiteralLike) -> AtomLike:
    return literal.get_atom()


def literal_polarity(literal) -> bool:
    return literal.get_polarity()


def predicate_name(pred: PredicateLike) -> str:
    try:
        return pred.get_name()
    except AttributeError:
        return pred.name


def predicate_arity(pred: PredicateLike) -> int:
    try:
        return pred.get_arity()
    except AttributeError:
        return pred.arity


def object_name(obj) -> str:
    return obj.get_name()


def action_name(action) -> str:
    try:
        return action.get_name()
    except AttributeError:
        return action.get_action().get_name()


def action_arity(action) -> int:
    try:
        return action.get_arity()
    except AttributeError:
        return action.get_action().get_arity()


def action_objects(action) -> Iterable:
    return action.get_objects()


def atom_signature(atom: AtomLike) -> tuple[str, tuple[str, ...]]:
    pred = predicate(atom)
    return (
        predicate_name(pred),
        tuple(object_name(obj) for obj in atom_objects(atom)),
    )


def atoms_equal(lhs: AtomLike, rhs: AtomLike) -> bool:
    return atom_signature(lhs) == atom_signature(rhs)


def literal_matches_atom(literal: LiteralLike, atom: AtomLike) -> bool:
    return atoms_equal(literal_atom(literal), atom)
