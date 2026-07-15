"""Backend-neutral planning snapshots and adapter contracts.

Backend implementations live in their own modules so importing this package
does not import either optional planning library.
"""

from .semantic import (
    ActionSchemaKey,
    AtomKey,
    DomainSnapshot,
    GroundActionKey,
    LiteralKey,
    PredicateCategory,
    PredicateKey,
    ProblemSnapshot,
    SnapshotReader,
    StateSnapshot,
)

__all__ = [
    "ActionSchemaKey",
    "AtomKey",
    "DomainSnapshot",
    "GroundActionKey",
    "LiteralKey",
    "PredicateCategory",
    "PredicateKey",
    "ProblemSnapshot",
    "SnapshotReader",
    "StateSnapshot",
]
