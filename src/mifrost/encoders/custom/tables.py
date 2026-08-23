"""Insertion-ordered vocabularies, node interning, and edge accumulation.

These tables are the pure-Python building blocks custom encoders use to
assemble one graph before handing it to a `BatchBuilder`. All integer ids are
dense row indices; all arrays are numpy float32/int64 ready for ingestion.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np


class Vocabulary:
    """Insertion-ordered bidirectional string-to-id map."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._names: list[str] = []
        self._frozen = False

    def id_for(self, name: str) -> int:
        """Return the id of ``name``, auto-extending unless frozen.

        After `freeze` the vocabulary rejects unknown names with ValueError;
        known names keep resolving to their existing ids.
        """

        try:
            found = self._ids.get(name)
        except TypeError:
            name = str(name)
            found = self._ids.get(name)
        if found is not None:
            return found
        if self._frozen:
            raise ValueError(f"vocabulary is frozen; cannot add {name!r}")
        index = len(self._names)
        self._ids[name] = index
        self._names.append(name)
        return index

    def name_for(self, id_: int) -> str:
        """Return the name stored at ``id_``."""

        return self._names[int(id_)]

    def __len__(self) -> int:
        return len(self._names)

    def names(self) -> list[str]:
        """All names in insertion order."""

        return list(self._names)

    def freeze(self) -> None:
        """Forbid future extensions; idempotent."""

        self._frozen = True


class NodeTable:
    """Interning table mapping hashable node keys to dense row ids.

    The FIRST call for a key wins: later calls with the same key return the
    existing row id and silently ignore any new role, channels or name. This
    makes repeated lookups (e.g. object nodes touched by many atoms) safe and
    cheap.
    """

    def __init__(self) -> None:
        self._keys: dict[Any, int] = {}
        self._roles: list[str] = []
        self._channels: list[tuple[int, ...]] = []
        self._names: dict[int, str] = {}
        self._roles_vocabulary = Vocabulary()

    def id_for(
        self,
        key: Hashable,
        *,
        role: str,
        channels: Sequence[int] = (),
        name: str | None = None,
    ) -> int:
        """Intern ``key`` and return its dense row id."""

        found = self._keys.get(key)
        if found is not None:
            return found
        row = len(self._roles)
        self._keys[key] = row
        self._roles.append(str(role))
        self._channels.append(tuple(int(value) for value in channels))
        if name is not None:
            self._names[row] = str(name)
        return row

    @property
    def count(self) -> int:
        """Number of interned rows."""

        return len(self._roles)

    @property
    def channel_dim(self) -> int:
        """Maximum per-row channel count seen; shorter rows are zero-padded."""

        return max((len(channels) for channels in self._channels), default=0)

    @property
    def roles(self) -> list[str]:
        """Role string of each row, in row order."""

        return list(self._roles)

    @property
    def roles_vocabulary(self) -> Vocabulary:
        """Insertion-ordered vocabulary over the distinct roles seen."""

        return self._roles_vocabulary

    def role_ids(self) -> list[int]:
        """Vocabulary id of each row's role, in row order."""

        return [self._roles_vocabulary.id_for(role) for role in self._roles]

    def names(self) -> list[str] | None:
        """Row names in row order, or ``None`` when any row is unnamed."""

        if len(self._names) != len(self._roles):
            return None
        return [self._names[row] for row in range(len(self._roles))]

    def to_float_array(self) -> np.ndarray:
        """Row-major float32 array of shape ``[count, channel_dim]``."""

        rows = np.zeros((self.count, self.channel_dim), dtype=np.float32)
        for row, channels in enumerate(self._channels):
            if channels:
                rows[row, : len(channels)] = channels
        return rows


class EdgeSink:
    """Accumulator for typed edges with two integer positions each."""

    def __init__(self) -> None:
        self._sources: list[int] = []
        self._targets: list[int] = []
        self._kinds: list[tuple[int, int, int]] = []
        self._kind_vocabulary = Vocabulary()

    def add(
        self,
        src: int,
        dst: int,
        kind: str,
        pos_a: int = 0,
        pos_b: int = 0,
    ) -> None:
        """Append one directed edge."""

        self._sources.append(int(src))
        self._targets.append(int(dst))
        known = self._kind_vocabulary._ids.get(kind)
        kind_id = known if known is not None else self._kind_vocabulary.id_for(kind)
        self._kinds.append((kind_id, int(pos_a), int(pos_b)))

    def add_both(
        self,
        src: int,
        dst: int,
        kind_fwd: str,
        kind_bwd: str,
        pos_a: int = 0,
        pos_b: int = 0,
    ) -> None:
        """Add ``(src, dst, kind_fwd, pos_a, pos_b)`` plus its reverse edge."""

        self.add(src, dst, kind_fwd, pos_a, pos_b)
        self.add(dst, src, kind_bwd, pos_b, pos_a)

    @property
    def kinds(self) -> Vocabulary:
        """Insertion-ordered vocabulary over the distinct edge kinds."""

        return self._kind_vocabulary

    def __len__(self) -> int:
        return len(self._sources)

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(edge_index int64 [2,E], edge_attr float32 [E,3])``.

        The feature columns are ``(kind_id, pos_a, pos_b)``.
        """

        count = len(self._sources)
        edge_index = np.zeros((2, count), dtype=np.int64)
        if count == 0:
            return edge_index, np.zeros((0, 3), dtype=np.float32)
        edge_index[0] = self._sources
        edge_index[1] = self._targets
        edge_attr = np.array(self._kinds, dtype=np.float32).reshape(count, 3)
        return edge_index, edge_attr


__all__ = ["EdgeSink", "NodeTable", "Vocabulary"]
