from __future__ import annotations

import pytest

from mifrost.backends import (
    ActionSchemaKey,
    AtomKey,
    DomainSnapshot,
    GroundActionKey,
    LiteralKey,
    PredicateCategory,
    PredicateKey,
    ProblemSnapshot,
)


def test_snapshots_canonicalize_backend_iteration_order() -> None:
    fluent = PredicateKey(PredicateCategory.FLUENT, "at", 2)
    static = PredicateKey(PredicateCategory.STATIC, "object", 1)
    move = ActionSchemaKey("move", 3)

    first = DomainSnapshot.canonical(
        name="transport",
        predicates=[fluent, static],
        actions=[move],
    )
    second = DomainSnapshot.canonical(
        name="transport",
        predicates=[static, fluent],
        actions=[move],
    )

    assert first == second
    assert first.predicates == (fluent, static)


def test_atom_and_action_bindings_preserve_argument_order() -> None:
    at = PredicateKey(PredicateCategory.FLUENT, "at", 2)
    move = ActionSchemaKey("move", 3)

    assert AtomKey(at, ("robot", "room-a")) != AtomKey(at, ("room-a", "robot"))
    assert GroundActionKey(move, ("robot", "room-a", "room-b")) != GroundActionKey(
        move, ("robot", "room-b", "room-a")
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PredicateKey(PredicateCategory.FLUENT, "", 1),
        lambda: PredicateKey(PredicateCategory.FLUENT, "p", -1),
        lambda: ActionSchemaKey("", 0),
        lambda: ActionSchemaKey("move", -1),
        lambda: AtomKey(PredicateKey(PredicateCategory.FLUENT, "at", 2), ("only-one",)),
        lambda: GroundActionKey(ActionSchemaKey("move", 2), ("only-one",)),
    ],
)
def test_semantic_keys_reject_invalid_invariants(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_problem_snapshot_rejects_duplicate_object_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        ProblemSnapshot.canonical(
            name="problem",
            domain_name="domain",
            objects=["a", "a"],
            static_atoms=[],
            goals=[],
        )


def test_problem_snapshot_canonicalizes_goals_without_losing_polarity() -> None:
    predicate = PredicateKey(PredicateCategory.FLUENT, "flag", 0)
    atom = AtomKey(predicate, ())

    snapshot = ProblemSnapshot.canonical(
        name="problem",
        domain_name="domain",
        objects=[],
        static_atoms=[],
        goals=[LiteralKey(atom, True), LiteralKey(atom, False)],
    )

    assert snapshot.goals == (LiteralKey(atom, False), LiteralKey(atom, True))


def test_domain_snapshot_types_default_to_none_and_are_sorted() -> None:
    fluent = PredicateKey(PredicateCategory.FLUENT, "at", 2)
    move = ActionSchemaKey("move", 1)

    untyped = DomainSnapshot.canonical(name="d", predicates=[fluent], actions=[move])
    assert untyped.types is None

    typed = DomainSnapshot.canonical(
        name="d",
        predicates=[fluent],
        actions=[move],
        types=["truck", "object", "location"],
    )
    assert typed.types == ("location", "object", "truck")


def test_problem_snapshot_object_types_stay_aligned_with_sorted_objects() -> None:
    # Objects are re-sorted by name in `canonical()`; object_types must move
    # with them, not just be sorted independently, or a type would silently
    # attach to the wrong object.
    snapshot = ProblemSnapshot.canonical(
        name="problem",
        domain_name="domain",
        objects=["truck1", "airplane1", "city1"],
        object_types=["truck", "airplane", "city"],
        static_atoms=[],
        goals=[],
    )

    assert snapshot.objects == ("airplane1", "city1", "truck1")
    assert snapshot.object_types == ("airplane", "city", "truck")


def test_problem_snapshot_object_types_default_to_none() -> None:
    snapshot = ProblemSnapshot.canonical(
        name="problem",
        domain_name="domain",
        objects=["a", "b"],
        static_atoms=[],
        goals=[],
    )
    assert snapshot.object_types is None


def test_problem_snapshot_rejects_mismatched_object_types_length() -> None:
    with pytest.raises(ValueError, match="one entry per object"):
        ProblemSnapshot.canonical(
            name="problem",
            domain_name="domain",
            objects=["a", "b"],
            object_types=["truck"],
            static_atoms=[],
            goals=[],
        )
