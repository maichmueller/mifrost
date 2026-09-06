"""Sparse atom-composition encoder (Stage 1, pure Python).

This module implements the encoder side of the contract consumed by
``relmo.models.SparseAtomCompositionGNN`` (see
``docs/sparse_atom_composition.md`` in the sibling ``relm`` repository) and
described in the architecture note "Sparse Atom Compositions for Relational
STRIPS Message Passing". It stores one persistent vector per *argument
occurrence* of a represented atom rather than one vector per object, and
exposes the sparse pair/witness indices the consumer needs to compose atoms
that share an object pair without ever materializing the dense object-pair
universe.

Everything here is pure Python/PyTorch: no native ``BatchBuilder`` graph is
built and no C++ engine is involved. Two layers are exposed:

- A backend-free core, :func:`encode_sparse_atom_facts` plus
  :func:`batch_sparse_atom_encodings`, that operates on plain
  ``(predicate, args)`` tuples (:class:`~mifrost.encoders.custom.state_view.Atom`)
  and a :class:`SparseAtomPredicateSchema`. This layer never touches pymimir
  or pytyr and is what the unit tests exercise directly for hand-built
  Blocksworld-style and synthetic-graph scenarios.
- :class:`SparseAtomCompositionEncoder`, a thin :class:`~mifrost.encoders.custom.state_view.StateView`
  facade over a real pymimir/pytyr problem that resolves states and goals into
  the core layer's inputs.

Channel convention (frozen, see :data:`CHANNEL_NAMES`)
--------------------------------------------------------
``0=state``, ``1=satisfied``, ``2=unsatisfied``, ``3=auxiliary``. Every
represented atom carries exactly one of these four channel ids, and the
consumer allocates one wide atom MLP per ``(channel, base_predicate)`` pair.

Goal-status encoding
---------------------
Only ``satisfied``/``unsatisfied`` goal-status atoms are emitted, mirroring
the native library's ``GoalDerivation`` split with ``plain`` OFF (the native
default is ``{plain, satisfied}``, which is a *different* encoding: it emits
one atom per goal regardless of truth and never emits an ``unsatisfied``
atom). Emitting ``plain`` here would silently discard the two-tower
Blocksworld separation the architecture relies on, so this encoder always
emits exactly one status atom per supplied goal atom -- satisfied XOR
unsatisfied, never both, never neither, never plain -- and the current atom
for a satisfied goal is represented *separately*, in the state channel.

Nullary normalization
-----------------------
A nullary atom such as ``handempty()`` is encoded as a unary atom on a single
distinguished auxiliary object, :data:`NULLARY_OBJECT_NAME` (the literal
string used by the native hetero family's ``nullary_object_name``/
``add_nullary_predicates`` option -- ported here rather than reusing the flat
family's ``ignore_zero_arity_relations=True``, which would silently drop
``handempty``). The star object is added to the object universe only for
graphs that actually contain a nullary atom, gets its own auxiliary carrier,
and participates in pairs/witnesses like any other object. No arity-0 atom is
ever emitted; the consumer rejects arity 0 outright, naming this
normalization in its error message.

Known gap: object types (R13)
-------------------------------
The architecture's per-occurrence side information ``s_{q,j}`` includes
object types, but neither backend StateView wraps
(:attr:`~mifrost.encoders.custom.state_view.StateView.object_types`) exposes
per-object type data today -- it always returns ``None``. This encoder
therefore emits no ``object_type_ids``/``occurrence_type_ids`` at all (the
carrier's optional fields stay ``None``) rather than faking a constant
placeholder that would silently present as "one type for everything" to a
downstream embedding table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor

from .custom.base import _looks_like_state
from .custom.state_view import Atom, StateView

#: Frozen channel ids. Do not renumber: downstream benchmarks and the
#: consumer model hardcode this order.
CHANNEL_STATE = 0
CHANNEL_SATISFIED = 1
CHANNEL_UNSATISFIED = 2
CHANNEL_AUXILIARY = 3
CHANNEL_NAMES: tuple[str, ...] = ("state", "satisfied", "unsatisfied", "auxiliary")

#: Base predicate used for the R4 auxiliary unary ``object(o)`` carrier.
#:
#: The architecture note writes this predicate ``object`` (as in
#: :math:`\operatorname{object}^{\aux}(o)`), but a plain ``"object"`` collides
#: with the built-in unary ``object`` type predicate most PDDL domain schemas
#: already expose (pymimir and pytyr both surface the root PDDL type as a
#: static unary predicate named ``object``). This bracketed literal follows
#: the same reserved-symbol convention as :data:`NULLARY_OBJECT_NAME` --
#: characters no PDDL identifier can contain -- so it never collides with a
#: real domain predicate.
AUXILIARY_OBJECT_PREDICATE = "![object_carrier]!"

#: Distinguished auxiliary object nullary atoms are normalized onto. This is
#: the exact literal used by the native hetero family's
#: ``nullary_object_name`` default; kept identical so tooling that already
#: recognizes it (e.g. visualization) continues to.
NULLARY_OBJECT_NAME = "![nullary_symbol]!"


# --------------------------------------------------------------------------
# Predicate schema (R1)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SparseAtomPredicateSchema:
    """Fixed base-predicate vocabulary shared by a batch of encoded graphs.

    ``arities`` is the *encoded* arity: a nullary predicate is reported with
    arity 1 here (its normalized unary-on-``star`` form), matching what
    ``SparseAtomCompositionGNN(predicate_arities=...)`` expects and what every
    emitted atom of that predicate actually has. ``logical_arities`` keeps the
    original PDDL arity (0 for nullary predicates) for diagnostics.
    """

    names: tuple[str, ...]
    arities: tuple[int, ...]
    logical_arities: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.arities) or len(self.names) != len(
            self.logical_arities
        ):
            raise ValueError(
                "SparseAtomPredicateSchema names/arities/logical_arities must "
                "have equal length"
            )
        if len(set(self.names)) != len(self.names):
            raise ValueError("SparseAtomPredicateSchema predicate names must be unique")
        if any(arity < 1 for arity in self.arities):
            raise ValueError(
                "SparseAtomPredicateSchema arities must all be positive: nullary "
                "predicates are normalized to encoded arity 1"
            )

    @cached_property
    def name_to_id(self) -> dict[str, int]:
        """Map each base predicate name to its fixed schema index."""
        return {name: index for index, name in enumerate(self.names)}


def build_predicate_schema(
    predicates: Iterable[tuple[str, int]],
    *,
    auxiliary_predicate_name: str = AUXILIARY_OBJECT_PREDICATE,
) -> SparseAtomPredicateSchema:
    """Build a :class:`SparseAtomPredicateSchema` from ``(name, arity)`` pairs.

    Appends the reserved auxiliary carrier predicate
    (``auxiliary_predicate_name``, encoded arity 1) after the supplied
    predicates. Raises ``ValueError`` if a supplied predicate name collides
    with it or duplicates another supplied name.
    """

    names: list[str] = []
    logical: list[int] = []
    encoded: list[int] = []
    seen: set[str] = set()
    for name, arity in predicates:
        name = str(name)
        if name == auxiliary_predicate_name:
            raise ValueError(
                f"predicate name {name!r} collides with the reserved auxiliary "
                "carrier predicate; pass a different auxiliary_predicate_name"
            )
        if name in seen:
            raise ValueError(f"duplicate predicate name in schema: {name!r}")
        seen.add(name)
        names.append(name)
        arity = int(arity)
        logical.append(arity)
        encoded.append(arity if arity >= 1 else 1)
    names.append(auxiliary_predicate_name)
    logical.append(1)
    encoded.append(1)
    return SparseAtomPredicateSchema(
        names=tuple(names), arities=tuple(encoded), logical_arities=tuple(logical)
    )


# --------------------------------------------------------------------------
# Equality-pattern interning (R6)
# --------------------------------------------------------------------------


#: Safety cap on ``EqualityPatternTable.seed_arities``: the number of
#: canonical patterns for arity ``r`` is the Bell number ``B(r)``, which grows
#: super-exponentially. No PDDL predicate arity in practice comes close to
#: this; the cap exists only to fail loudly instead of silently allocating an
#: enormous table if it ever would.
_MAX_SEEDED_ARITY = 9


def _enumerate_equality_patterns(length: int) -> list[tuple[int, ...]]:
    """Enumerate every canonical restricted-growth string of ``length``.

    A restricted-growth string (``a_0 = 0``, ``a_i <= 1 + max(a_0..a_{i-1})``)
    is already the first-occurrence-relabeled canonical form
    :meth:`EqualityPatternTable.canonical` would produce for any concrete
    tuple realizing that partition of positions, and there is exactly one RGS
    per partition of ``{0, ..., length-1}`` -- ``Bell(length)`` of them.
    """
    partial: list[tuple[int, ...]] = [()]
    for _ in range(length):
        next_partial: list[tuple[int, ...]] = []
        for sequence in partial:
            max_label = max(sequence) if sequence else -1
            for label in range(max_label + 2):
                next_partial.append(sequence + (label,))
        partial = next_partial
    return partial


class EqualityPatternTable:
    """Insertion-ordered intern table for the within-atom equality pattern.

    The pattern ``epsilon_q = (1[o_{q,i} = o_{q,j}])_{i,j}`` is stored in its
    canonical compact form: positions are relabeled by first-occurrence order
    of their object, so ``(u, v, u)`` and ``(x, y, x)`` intern to the same id
    (``(0, 1, 0)``) while ``(u, v)`` and ``(u, u)`` intern to different ids
    (``(0, 1)`` vs ``(0, 0)``). This is a bijective encoding of the full
    equality matrix and is exactly what ``s_{q,j}`` needs to carry.

    A consumer model sizes its equality-pattern embedding table
    (``num_equality_patterns``) at *construction* time, before any state is
    encoded, so growing this table lazily as new repeated-argument shapes are
    encountered would leave that cardinality unknown up front. Pass
    ``max_arity`` (or call :meth:`seed_arities` before constructing the
    model) to pre-populate the *complete* pattern vocabulary for every arity
    up to it -- ``sum(Bell(1..max_arity))`` patterns, fixed and known in
    advance. ``id_for`` still accepts new patterns afterwards (e.g. if a
    caller needs to work with an un-seeded table for exploration), but a
    seeded table never needs to grow for any atom within its arity range.
    """

    def __init__(self, *, max_arity: int = 0) -> None:
        self._ids: dict[tuple[int, ...], int] = {}
        self._patterns: list[tuple[int, ...]] = []
        if max_arity:
            self.seed_arities(max_arity)

    def seed_arities(self, max_arity: int) -> None:
        """Pre-intern every equality pattern for arities ``1..max_arity``."""
        if max_arity > _MAX_SEEDED_ARITY:
            raise ValueError(
                f"refusing to seed equality patterns up to arity {max_arity}: "
                f"the pattern count is the Bell number B({max_arity}), which "
                f"exceeds the {_MAX_SEEDED_ARITY} safety cap; pass a smaller "
                "max_arity or seed manually if this is intentional"
            )
        for arity in range(1, int(max_arity) + 1):
            for pattern in _enumerate_equality_patterns(arity):
                if pattern not in self._ids:
                    index = len(self._patterns)
                    self._ids[pattern] = index
                    self._patterns.append(pattern)

    @staticmethod
    def canonical(args: Sequence[int]) -> tuple[int, ...]:
        """Return the first-occurrence relabeling of ``args``."""
        labels: dict[int, int] = {}
        out: list[int] = []
        for value in args:
            label = labels.get(value)
            if label is None:
                label = len(labels)
                labels[value] = label
            out.append(label)
        return tuple(out)

    def id_for(self, args: Sequence[int]) -> int:
        """Intern ``args``'s equality pattern and return its id."""
        key = self.canonical(args)
        found = self._ids.get(key)
        if found is not None:
            return found
        index = len(self._patterns)
        self._ids[key] = index
        self._patterns.append(key)
        return index

    @property
    def patterns(self) -> tuple[tuple[int, ...], ...]:
        """Interned canonical patterns, in id order."""
        return tuple(self._patterns)

    def __len__(self) -> int:
        return len(self._patterns)


# --------------------------------------------------------------------------
# Carrier
# --------------------------------------------------------------------------


@dataclass
class SparseAtomCompositionEncoding:
    """One (or one batched) sparse atom-composition graph.

    Field names and shapes match the ``SparseAtomCompositionBatch`` contract
    in ``relmo.models.sparse_atom_composition`` exactly, so an instance of
    this class can be passed directly to
    ``relmo.models.SparseAtomCompositionGNN(...)``. A single-graph encoding is
    already "batch of one": ``atom_batch``/``object_batch`` are all zero and
    ``goal_available`` has one entry.

    ``object_type_ids``/``occurrence_type_ids`` are always ``None`` -- see the
    module docstring's "Known gap" section (R13).
    """

    atom_args: Tensor
    atom_offsets: Tensor
    atom_predicate_ids: Tensor
    atom_channel_ids: Tensor
    atom_batch: Tensor
    object_batch: Tensor
    pair_objects: Tensor
    pair_support_offsets: Tensor
    pair_support_atom_ids: Tensor
    pair_support_i: Tensor
    pair_support_j: Tensor
    composition_triplets: Tensor
    atom_pair_ids: Tensor
    atom_pair_occurrence_i: Tensor
    atom_pair_occurrence_j: Tensor
    goal_available: Tensor
    equality_pattern_ids: Tensor
    object_carrier_occurrence_ids: Tensor
    counterpart_occurrence_ids: Tensor | None = None
    object_type_ids: Tensor | None = field(default=None, repr=False)
    occurrence_type_ids: Tensor | None = field(default=None, repr=False)

    @property
    def num_atoms(self) -> int:
        return int(self.atom_predicate_ids.numel())

    @property
    def num_objects(self) -> int:
        return int(self.object_batch.numel())

    @property
    def num_occurrences(self) -> int:
        return int(self.atom_args.numel())

    @property
    def num_pairs(self) -> int:
        return int(self.pair_objects.size(0))

    @property
    def num_graphs(self) -> int:
        return int(self.goal_available.numel())


def _empty_long(*shape: int) -> Tensor:
    return torch.empty(shape, dtype=torch.long)


# --------------------------------------------------------------------------
# Triangle enumeration (R8) -- degeneracy ordering, O(alpha * M)
# --------------------------------------------------------------------------


def _list_support_triangles(
    adjacency: dict[int, set[int]],
) -> list[tuple[int, int, int]]:
    """List every triangle of an undirected graph exactly once.

    Standard degeneracy-ordering algorithm (Chiba-Nishizeki): order vertices
    by degree, keep only "forward" edges (to a higher-ranked neighbour), and
    for each forward edge intersect the two forward-neighbour sets. Each
    triangle is discovered exactly once, via its lowest-ranked vertex. This
    is the same construction ``relmo``'s own topology-stats triangle counter
    uses, just listing instead of counting.
    """
    if not adjacency:
        return []
    order = sorted(adjacency, key=lambda vertex: (len(adjacency[vertex]), vertex))
    rank = {vertex: index for index, vertex in enumerate(order)}
    forward = {
        vertex: {other for other in neighbours if rank[other] > rank[vertex]}
        for vertex, neighbours in adjacency.items()
    }
    triangles: list[tuple[int, int, int]] = []
    for vertex, forward_v in forward.items():
        for other in forward_v:
            for witness in forward_v & forward[other]:
                triangles.append((vertex, other, witness))
    return triangles


# 6 ordered (target_u, target_v, witness) selections per undirected triangle
# {a, b, c}: every ordered edge as target, the remaining vertex as witness.
def _triangle_orderings(a: int, b: int, c: int) -> tuple[tuple[int, int, int], ...]:
    return (
        (a, b, c),
        (b, a, c),
        (a, c, b),
        (c, a, b),
        (b, c, a),
        (c, b, a),
    )


# --------------------------------------------------------------------------
# Core encoding (R1-R11), single graph
# --------------------------------------------------------------------------


def encode_sparse_atom_facts(
    objects: Sequence[str],
    current_atoms: Sequence[Atom],
    goal_atoms: Sequence[Atom] | None,
    schema: SparseAtomPredicateSchema,
    equality_table: EqualityPatternTable,
    *,
    exact_tuple_exchange: bool = False,
) -> SparseAtomCompositionEncoding:
    """Encode one state (plus optional goal) into a sparse atom-composition graph.

    ``current_atoms`` is the current-true-atom set S, including static facts
    (R2). ``goal_atoms`` is ``None`` for zeta=0 (goal-free -- no goal atoms,
    no counterpart map, ``goal_available=False``) or a (possibly empty)
    sequence of positive atoms for zeta=1 (``goal_available=True``); an empty
    sequence and ``None`` produce identical atom sets and differ only in
    ``goal_available`` (R10).

    Every goal atom in ``G`` becomes exactly one status atom: ``satisfied`` if
    it is a member of ``current_atoms`` (by predicate + argument-tuple
    equality, computed before nullary/star normalization), else
    ``unsatisfied`` -- never ``plain``, never both (R2).
    """

    zeta = goal_atoms is not None
    goal_list = list(goal_atoms) if zeta else []

    needs_star = any(not atom.args for atom in current_atoms) or any(
        not atom.args for atom in goal_list
    )
    object_names = list(objects)
    object_index = {name: index for index, name in enumerate(object_names)}
    star_id: int | None = None
    if needs_star:
        star_id = len(object_names)
        object_names.append(NULLARY_OBJECT_NAME)
    num_objects = len(object_names)

    def resolve_args(atom: Atom) -> tuple[int, ...]:
        if not atom.args:
            assert star_id is not None
            return (star_id,)
        try:
            return tuple(object_index[name] for name in atom.args)
        except KeyError as exc:
            raise ValueError(
                f"atom {atom.display!r} references an object outside the "
                "supplied object universe"
            ) from exc

    # entries: (base_predicate_id, channel_id, resolved_arg_ids)
    entries: list[tuple[int, int, tuple[int, ...]]] = []

    def predicate_id_for(name: str) -> int:
        found = schema.name_to_id.get(name)
        if found is None:
            raise ValueError(f"unknown predicate {name!r}: not in the supplied schema")
        return found

    state_key_to_index: dict[tuple[str, tuple[str, ...]], int] = {}
    for atom in current_atoms:
        state_key_to_index[(atom.predicate, atom.args)] = len(entries)
        entries.append(
            (predicate_id_for(atom.predicate), CHANNEL_STATE, resolve_args(atom))
        )

    # (goal_entry_index, state_entry_index) pairs for R11 exact-tuple exchange.
    satisfied_links: list[tuple[int, int]] = []
    if zeta:
        for atom in goal_list:
            key = (atom.predicate, atom.args)
            state_index = state_key_to_index.get(key)
            channel = (
                CHANNEL_SATISFIED if state_index is not None else CHANNEL_UNSATISFIED
            )
            entry_index = len(entries)
            entries.append(
                (predicate_id_for(atom.predicate), channel, resolve_args(atom))
            )
            if state_index is not None:
                satisfied_links.append((entry_index, state_index))

    aux_predicate_id = predicate_id_for(AUXILIARY_OBJECT_PREDICATE)
    carrier_entry_index = [0] * num_objects
    for object_id in range(num_objects):
        carrier_entry_index[object_id] = len(entries)
        entries.append((aux_predicate_id, CHANNEL_AUXILIARY, (object_id,)))

    num_atoms = len(entries)
    arities = [len(entry[2]) for entry in entries]
    for (predicate_id, _channel, args), arity in zip(entries, arities):
        if arity == 0:
            raise AssertionError(
                "internal error: an arity-0 atom escaped nullary normalization"
            )
        expected = schema.arities[predicate_id]
        if arity != expected:
            raise ValueError(
                f"atom for predicate {schema.names[predicate_id]!r} has arity "
                f"{arity}, but the schema declares arity {expected}"
            )
        del args

    atom_predicate_ids = torch.tensor([entry[0] for entry in entries], dtype=torch.long)
    atom_channel_ids = torch.tensor([entry[1] for entry in entries], dtype=torch.long)
    atom_offsets = torch.zeros(num_atoms + 1, dtype=torch.long)
    if arities:
        atom_offsets[1:] = torch.cumsum(torch.tensor(arities, dtype=torch.long), dim=0)
    atom_args = torch.tensor(
        [object_id for entry in entries for object_id in entry[2]], dtype=torch.long
    )
    num_occurrences = int(atom_args.numel())

    equality_pattern_ids = torch.tensor(
        [equality_table.id_for(entry[2]) for entry in entries], dtype=torch.long
    )

    object_carrier_occurrence_ids = torch.tensor(
        [int(atom_offsets[carrier_entry_index[o]].item()) for o in range(num_objects)],
        dtype=torch.long,
    )

    # R7 (pair support) + R9 (atom-to-pair maps): a single pass over ordered,
    # non-repeated argument position pairs of every atom.
    pair_id_of: dict[tuple[int, int], int] = {}
    pair_objects_raw: list[tuple[int, int]] = []
    pair_support_raw: list[list[tuple[int, int, int]]] = []
    atom_pair_ids_raw: list[int] = []
    atom_pair_occurrence_i_raw: list[int] = []
    atom_pair_occurrence_j_raw: list[int] = []

    for atom_id, (_predicate_id, _channel, args) in enumerate(entries):
        arity = len(args)
        if arity < 2:
            continue
        start = int(atom_offsets[atom_id].item())
        for i in range(arity):
            u = args[i]
            for j in range(arity):
                if i == j:
                    continue
                v = args[j]
                if u == v:
                    continue
                pair_id = pair_id_of.get((u, v))
                if pair_id is None:
                    pair_id = len(pair_objects_raw)
                    pair_id_of[(u, v)] = pair_id
                    pair_objects_raw.append((u, v))
                    pair_support_raw.append([])
                pair_support_raw[pair_id].append((atom_id, i, j))
                atom_pair_ids_raw.append(pair_id)
                atom_pair_occurrence_i_raw.append(start + i)
                atom_pair_occurrence_j_raw.append(start + j)

    # Canonicalize pair order (sorted by (u, v)) so object renaming under a
    # fixed permutation of ids, and repeated encodes of the same graph,
    # produce a deterministic pair ordering rather than one keyed on
    # traversal order.
    order = sorted(range(len(pair_objects_raw)), key=lambda idx: pair_objects_raw[idx])
    remap = {old: new for new, old in enumerate(order)}
    num_pairs = len(order)
    if num_pairs:
        pair_objects = torch.tensor(
            [pair_objects_raw[old] for old in order], dtype=torch.long
        )
    else:
        pair_objects = _empty_long(0, 2)

    atom_pair_ids = torch.tensor(
        [remap[old] for old in atom_pair_ids_raw], dtype=torch.long
    )
    atom_pair_occurrence_i = torch.tensor(atom_pair_occurrence_i_raw, dtype=torch.long)
    atom_pair_occurrence_j = torch.tensor(atom_pair_occurrence_j_raw, dtype=torch.long)

    pair_support_offsets = torch.zeros(num_pairs + 1, dtype=torch.long)
    support_atom_ids: list[int] = []
    support_i: list[int] = []
    support_j: list[int] = []
    for new_id, old_id in enumerate(order):
        contributions = pair_support_raw[old_id]
        pair_support_offsets[new_id + 1] = pair_support_offsets[new_id] + len(
            contributions
        )
        for atom_id, i, j in contributions:
            support_atom_ids.append(atom_id)
            support_i.append(i)
            support_j.append(j)
    pair_support_atom_ids = torch.tensor(support_atom_ids, dtype=torch.long)
    pair_support_i = torch.tensor(support_i, dtype=torch.long)
    pair_support_j = torch.tensor(support_j, dtype=torch.long)

    # R8: witness triplets via degeneracy-ordered triangle listing on the
    # undirected support graph. Any atom that contributes (u, v) also
    # contributes (v, u) (the double loop above visits both (i, j) and
    # (j, i)), so edge membership can be read off either direction of
    # pair_id_of.
    final_pair_id: dict[tuple[int, int], int] = {
        pair_objects_raw[old]: remap[old] for old in order
    }
    adjacency: dict[int, set[int]] = {}
    for u, v in final_pair_id:
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)

    composition_triplets_raw: list[tuple[int, int, int]] = []
    for a, b, c in _list_support_triangles(adjacency):
        for target_u, target_v, witness in _triangle_orderings(a, b, c):
            target_pair = final_pair_id.get((target_u, target_v))
            left_pair = final_pair_id.get((target_u, witness))
            right_pair = final_pair_id.get((target_v, witness))
            if target_pair is None or left_pair is None or right_pair is None:
                # Defensive only: guaranteed present by the symmetric-support
                # argument above for any real triangle.
                continue
            composition_triplets_raw.append((target_pair, left_pair, right_pair))
    if composition_triplets_raw:
        composition_triplets = torch.tensor(composition_triplets_raw, dtype=torch.long)
    else:
        composition_triplets = _empty_long(0, 3)

    # R11: optional exact-tuple counterpart exchange, current -> satisfied-goal.
    counterpart_occurrence_ids: Tensor | None = None
    if exact_tuple_exchange and zeta:
        counterpart_occurrence_ids = torch.full(
            (num_occurrences,), -1, dtype=torch.long
        )
        for goal_index, state_index in satisfied_links:
            goal_start = int(atom_offsets[goal_index].item())
            state_start = int(atom_offsets[state_index].item())
            arity = arities[goal_index]
            for position in range(arity):
                counterpart_occurrence_ids[state_start + position] = (
                    goal_start + position
                )

    atom_batch = torch.zeros(num_atoms, dtype=torch.long)
    object_batch = torch.zeros(num_objects, dtype=torch.long)
    goal_available = torch.tensor([zeta], dtype=torch.bool)

    return SparseAtomCompositionEncoding(
        atom_args=atom_args,
        atom_offsets=atom_offsets,
        atom_predicate_ids=atom_predicate_ids,
        atom_channel_ids=atom_channel_ids,
        atom_batch=atom_batch,
        object_batch=object_batch,
        pair_objects=pair_objects,
        pair_support_offsets=pair_support_offsets,
        pair_support_atom_ids=pair_support_atom_ids,
        pair_support_i=pair_support_i,
        pair_support_j=pair_support_j,
        composition_triplets=composition_triplets,
        atom_pair_ids=atom_pair_ids,
        atom_pair_occurrence_i=atom_pair_occurrence_i,
        atom_pair_occurrence_j=atom_pair_occurrence_j,
        goal_available=goal_available,
        equality_pattern_ids=equality_pattern_ids,
        object_carrier_occurrence_ids=object_carrier_occurrence_ids,
        counterpart_occurrence_ids=counterpart_occurrence_ids,
    )


# --------------------------------------------------------------------------
# Batching (R12): explicit index rebasing, concatenated index space
# --------------------------------------------------------------------------


def batch_sparse_atom_encodings(
    encodings: Sequence[SparseAtomCompositionEncoding],
) -> SparseAtomCompositionEncoding:
    """Concatenate several encodings, rebasing every index space explicitly.

    No pair or witness triplet ever spans two input graphs: every per-graph
    tensor is offset by that graph's own running totals (objects, atoms,
    occurrences, pairs, pair-support entries, graphs) before concatenation,
    and each input graph's internal structure -- already validated at
    encode time -- is preserved verbatim.
    """

    if not encodings:
        raise ValueError("batch_sparse_atom_encodings requires at least one encoding")
    if len(encodings) == 1:
        return encodings[0]

    any_counterparts = any(
        encoding.counterpart_occurrence_ids is not None for encoding in encodings
    )

    atom_args_parts: list[Tensor] = []
    atom_offset_tail_parts: list[Tensor] = [torch.zeros(1, dtype=torch.long)]
    atom_predicate_id_parts: list[Tensor] = []
    atom_channel_id_parts: list[Tensor] = []
    atom_batch_parts: list[Tensor] = []
    object_batch_parts: list[Tensor] = []
    pair_objects_parts: list[Tensor] = []
    pair_support_offset_tail_parts: list[Tensor] = [torch.zeros(1, dtype=torch.long)]
    pair_support_atom_id_parts: list[Tensor] = []
    pair_support_i_parts: list[Tensor] = []
    pair_support_j_parts: list[Tensor] = []
    composition_triplet_parts: list[Tensor] = []
    atom_pair_id_parts: list[Tensor] = []
    atom_pair_occurrence_i_parts: list[Tensor] = []
    atom_pair_occurrence_j_parts: list[Tensor] = []
    goal_available_parts: list[Tensor] = []
    equality_pattern_id_parts: list[Tensor] = []
    object_carrier_occurrence_id_parts: list[Tensor] = []
    counterpart_parts: list[Tensor] = []

    object_offset = 0
    occurrence_offset = 0
    atom_offset = 0
    pair_offset = 0
    support_offset = 0
    graph_offset = 0

    for encoding in encodings:
        num_atoms = encoding.num_atoms
        num_objects = encoding.num_objects
        num_occurrences = encoding.num_occurrences
        num_pairs = encoding.num_pairs
        num_support = int(encoding.pair_support_atom_ids.numel())
        num_graphs = encoding.num_graphs

        atom_args_parts.append(encoding.atom_args + object_offset)
        atom_offset_tail_parts.append(encoding.atom_offsets[1:] + occurrence_offset)
        atom_predicate_id_parts.append(encoding.atom_predicate_ids)
        atom_channel_id_parts.append(encoding.atom_channel_ids)
        atom_batch_parts.append(encoding.atom_batch + graph_offset)
        object_batch_parts.append(encoding.object_batch + graph_offset)
        pair_objects_parts.append(encoding.pair_objects + object_offset)
        pair_support_offset_tail_parts.append(
            encoding.pair_support_offsets[1:] + support_offset
        )
        pair_support_atom_id_parts.append(encoding.pair_support_atom_ids + atom_offset)
        pair_support_i_parts.append(encoding.pair_support_i)
        pair_support_j_parts.append(encoding.pair_support_j)
        composition_triplet_parts.append(encoding.composition_triplets + pair_offset)
        atom_pair_id_parts.append(encoding.atom_pair_ids + pair_offset)
        atom_pair_occurrence_i_parts.append(
            encoding.atom_pair_occurrence_i + occurrence_offset
        )
        atom_pair_occurrence_j_parts.append(
            encoding.atom_pair_occurrence_j + occurrence_offset
        )
        goal_available_parts.append(encoding.goal_available)
        equality_pattern_id_parts.append(encoding.equality_pattern_ids)
        object_carrier_occurrence_id_parts.append(
            encoding.object_carrier_occurrence_ids + occurrence_offset
        )
        if any_counterparts:
            if encoding.counterpart_occurrence_ids is not None:
                local = encoding.counterpart_occurrence_ids.clone()
                valid = local >= 0
                local = torch.where(valid, local + occurrence_offset, local)
            else:
                local = torch.full((num_occurrences,), -1, dtype=torch.long)
            counterpart_parts.append(local)

        object_offset += num_objects
        occurrence_offset += num_occurrences
        atom_offset += num_atoms
        pair_offset += num_pairs
        support_offset += num_support
        graph_offset += num_graphs

    return SparseAtomCompositionEncoding(
        atom_args=torch.cat(atom_args_parts) if atom_args_parts else _empty_long(0),
        atom_offsets=torch.cat(atom_offset_tail_parts),
        atom_predicate_ids=torch.cat(atom_predicate_id_parts),
        atom_channel_ids=torch.cat(atom_channel_id_parts),
        atom_batch=torch.cat(atom_batch_parts),
        object_batch=torch.cat(object_batch_parts),
        pair_objects=torch.cat(pair_objects_parts, dim=0),
        pair_support_offsets=torch.cat(pair_support_offset_tail_parts),
        pair_support_atom_ids=torch.cat(pair_support_atom_id_parts),
        pair_support_i=torch.cat(pair_support_i_parts),
        pair_support_j=torch.cat(pair_support_j_parts),
        composition_triplets=torch.cat(composition_triplet_parts, dim=0),
        atom_pair_ids=torch.cat(atom_pair_id_parts),
        atom_pair_occurrence_i=torch.cat(atom_pair_occurrence_i_parts),
        atom_pair_occurrence_j=torch.cat(atom_pair_occurrence_j_parts),
        goal_available=torch.cat(goal_available_parts),
        equality_pattern_ids=torch.cat(equality_pattern_id_parts),
        object_carrier_occurrence_ids=torch.cat(object_carrier_occurrence_id_parts),
        counterpart_occurrence_ids=(
            torch.cat(counterpart_parts) if any_counterparts else None
        ),
    )


# --------------------------------------------------------------------------
# StateView-backed encoder facade
# --------------------------------------------------------------------------


class SparseAtomCompositionEncoder:
    """StateView facade producing :class:`SparseAtomCompositionEncoding`.

    Wraps a pymimir ``Problem`` or pytyr ``PlanningTask`` (auto-detected, or
    pass ``backend=``) the same way
    :class:`~mifrost.encoders.custom.base.CustomGraphEncoder` does, and fixes
    the base-predicate vocabulary (:attr:`schema`) from the domain's
    predicates at construction time -- shared across every ``encode``/
    ``encode_batch`` call on this instance, so predicate ids stay stable for
    a model trained against ``encoder.schema.arities``.

    Unlike ``CustomGraphEncoder``'s goal lane (where ``goals=None`` means
    "use the problem's own goal"), ``goals=None`` here means *goal-free*
    (zeta=0): this encoder has a genuine goal-free mode, and conflating it
    with "problem default" would make that mode unreachable. Pass
    ``goals=encoder.view.goal_literals(state)`` explicitly to encode the
    problem's own goal, or ``goals=()`` for a supplied-empty goal (zeta=1,
    G=empty; same atom set as goal-free, but ``goal_available=True``).

    :attr:`equality_patterns` is seeded at construction with the complete
    equality-pattern vocabulary for every arity up to the schema's maximum
    (see :meth:`EqualityPatternTable.seed_arities`), so
    :attr:`num_equality_patterns` is fixed and known before any state is
    encoded -- construct the consumer model with it up front.
    """

    CHANNEL_NAMES = CHANNEL_NAMES
    AUXILIARY_OBJECT_PREDICATE = AUXILIARY_OBJECT_PREDICATE
    NULLARY_OBJECT_NAME = NULLARY_OBJECT_NAME

    def __init__(
        self,
        source: Any,
        *,
        backend: str | None = None,
        exact_tuple_exchange: bool = False,
    ) -> None:
        self.view = StateView(source, backend=backend)
        self.backend = self.view.backend
        self.schema = build_predicate_schema(
            [(info.name, info.arity) for info in self.view.predicates],
            auxiliary_predicate_name=self.AUXILIARY_OBJECT_PREDICATE,
        )
        self.equality_patterns = EqualityPatternTable(
            max_arity=max(self.schema.arities, default=0)
        )
        self.exact_tuple_exchange = bool(exact_tuple_exchange)

    @property
    def predicate_names(self) -> tuple[str, ...]:
        return self.schema.names

    @property
    def predicate_arities(self) -> tuple[int, ...]:
        """Encoded base-predicate arities, ready for ``SparseAtomCompositionGNN``."""
        return self.schema.arities

    @property
    def num_equality_patterns(self) -> int:
        """Size of the (pre-seeded) equality-pattern vocabulary."""
        return len(self.equality_patterns)

    def _current_atoms(self, state: Any) -> list[Atom]:
        return [*self.view.static_facts, *self.view.state_facts(state)]

    def _goal_atoms(self, goals: Any) -> list[Atom] | None:
        if goals is None:
            return None
        literals = self.view.neutral_literals(goals, field="goals")
        atoms: list[Atom] = []
        for literal in literals:
            if not literal.positive:
                raise ValueError(
                    "SparseAtomCompositionEncoder only supports positive "
                    f"conjunctive goals; got a negative literal for "
                    f"{literal.atom.display}"
                )
            atoms.append(literal.atom)
        return atoms

    def encode(self, state: Any, *, goals: Any = None) -> SparseAtomCompositionEncoding:
        """Encode one state (see the class docstring for ``goals`` semantics)."""
        current = self._current_atoms(state)
        goal_atoms = self._goal_atoms(goals)
        return encode_sparse_atom_facts(
            self.view.objects,
            current,
            goal_atoms,
            self.schema,
            self.equality_patterns,
            exact_tuple_exchange=self.exact_tuple_exchange,
        )

    def encode_batch(
        self, states: Any, *, goals: Sequence[Any | None] | None = None
    ) -> SparseAtomCompositionEncoding:
        """Encode several states into one concatenated, cross-graph-safe batch.

        ``states`` is one state or an iterable of states. ``goals`` is either
        ``None`` (goal-free for every graph) or a sequence of per-state goal
        values of the same length, each following :meth:`encode`'s ``goals``
        semantics (``None`` for that graph is goal-free).
        """
        state_list = [states] if _looks_like_state(states) else list(states)
        if goals is None:
            goal_list: list[Any] = [None] * len(state_list)
        else:
            goal_list = list(goals)
            if len(goal_list) != len(state_list):
                raise ValueError("goals length must match states length")
        encodings = [
            self.encode(state, goals=goal) for state, goal in zip(state_list, goal_list)
        ]
        return batch_sparse_atom_encodings(encodings)


# --------------------------------------------------------------------------
# Contract validator
# --------------------------------------------------------------------------


def _as_long(value: Any) -> Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    return tensor.to(dtype=torch.long)


def validate_sparse_atom_composition(
    encoding: Any,
    *,
    predicate_arities: Sequence[int] | None = None,
    num_channels: int = len(CHANNEL_NAMES),
) -> None:
    """Assert every structural invariant ``SparseAtomCompositionGNN.prepare()`` checks.

    Accepts anything duck-typed like :class:`SparseAtomCompositionEncoding`
    (attribute access only, so a plain ``SimpleNamespace`` or PyG ``Data``
    works too). Raises ``ValueError`` naming the first violated invariant;
    returns ``None`` on success. Pass ``predicate_arities`` (the sequence a
    consumer model would be constructed with) to additionally check that
    every atom's arity matches its declared base predicate.

    This function is a standalone reimplementation of the checks in
    ``relmo.models.sparse_atom_composition.SparseAtomCompositionGNN.prepare``,
    kept in sync by the integration test that feeds real encoder output
    through the actual consumer model.
    """

    atom_args = _as_long(encoding.atom_args)
    atom_offsets = _as_long(encoding.atom_offsets)
    atom_predicate_ids = _as_long(encoding.atom_predicate_ids)
    atom_channel_ids = _as_long(encoding.atom_channel_ids)
    atom_batch = _as_long(encoding.atom_batch)
    object_batch = _as_long(encoding.object_batch)
    pair_objects = _as_long(encoding.pair_objects)
    pair_support_offsets = _as_long(encoding.pair_support_offsets)
    pair_support_atom_ids = _as_long(encoding.pair_support_atom_ids)
    pair_support_i = _as_long(encoding.pair_support_i)
    pair_support_j = _as_long(encoding.pair_support_j)
    composition_triplets = (
        _as_long(encoding.composition_triplets).view(-1, 3)
        if (torch.as_tensor(encoding.composition_triplets).numel())
        else _empty_long(0, 3)
    )
    atom_pair_ids = _as_long(encoding.atom_pair_ids)
    atom_pair_occurrence_i = _as_long(encoding.atom_pair_occurrence_i)
    atom_pair_occurrence_j = _as_long(encoding.atom_pair_occurrence_j)
    goal_available = torch.as_tensor(encoding.goal_available)
    if goal_available.dtype != torch.bool:
        goal_available = goal_available != 0
    equality_pattern_ids = getattr(encoding, "equality_pattern_ids", None)
    object_carrier_occurrence_ids = getattr(
        encoding, "object_carrier_occurrence_ids", None
    )
    counterpart_occurrence_ids = getattr(encoding, "counterpart_occurrence_ids", None)

    q = int(atom_predicate_ids.numel())
    num_objects = int(object_batch.numel())
    graph_count = int(goal_available.numel())

    if pair_objects.dim() != 2 or pair_objects.size(1) != 2:
        raise ValueError("pair_objects must have shape [P, 2]")
    if composition_triplets.dim() != 2 or composition_triplets.size(1) != 3:
        raise ValueError("composition_triplets must have shape [K, 3]")
    if int(atom_offsets.numel()) != q + 1:
        raise ValueError("atom_offsets must have one more entry than atom metadata")
    last_offset = int(atom_offsets[-1].item()) if atom_offsets.numel() else 0
    if int(atom_args.numel()) != last_offset:
        raise ValueError("atom_offsets[-1] must equal atom_args length")
    if int(atom_batch.numel()) != q:
        raise ValueError("atom_batch must contain one graph id per atom")
    if int(atom_channel_ids.numel()) != q:
        raise ValueError("atom_channel_ids must contain one entry per atom")
    if num_objects == 0 and int(atom_args.numel()):
        raise ValueError("atom arguments require at least one object")
    if graph_count == 0:
        raise ValueError("goal_available must contain at least one graph flag")

    def check_graph_ids(values: Tensor, name: str) -> None:
        if int(values.numel()) and (
            int(values.min().item()) < 0 or int(values.max().item()) >= graph_count
        ):
            raise ValueError(f"{name} references a missing or negative graph id")

    check_graph_ids(atom_batch, "atom_batch")
    check_graph_ids(object_batch, "object_batch")

    arities = atom_offsets[1:] - atom_offsets[:-1]
    if bool((arities < 0).any()):
        raise ValueError("atom_offsets must be non-decreasing")
    if bool((arities == 0).any()):
        raise ValueError(
            "arity-0 atoms are not permitted: nullary predicates must be "
            "normalized to a unary atom on a distinguished auxiliary object"
        )
    if int(atom_channel_ids.numel()) and (
        int(atom_channel_ids.min().item()) < 0
        or int(atom_channel_ids.max().item()) >= num_channels
    ):
        raise ValueError("atom_channel_ids references a channel outside the schema")

    if predicate_arities is not None:
        arity_lookup = torch.as_tensor(list(predicate_arities), dtype=torch.long)
        if int(atom_predicate_ids.numel()) and (
            int(atom_predicate_ids.min().item()) < 0
            or int(atom_predicate_ids.max().item()) >= arity_lookup.numel()
        ):
            raise ValueError(
                "atom_predicate_ids references a predicate outside predicate_arities"
            )
        if q:
            expected = arity_lookup.index_select(0, atom_predicate_ids)
            mismatch = expected != arities
            if bool(mismatch.any()):
                index = int(torch.nonzero(mismatch)[0].item())
                raise ValueError(
                    "atom arity does not match predicate schema: "
                    f"predicate={int(atom_predicate_ids[index].item())}, "
                    f"got {int(arities[index].item())}, "
                    f"expected {int(expected[index].item())}"
                )

    atom_starts = atom_offsets[:-1]
    occurrence_atom_ids = torch.repeat_interleave(
        torch.arange(q, dtype=torch.long), arities
    )
    total_occurrences = int(atom_args.numel())
    if int(atom_args.numel()) and (
        int(atom_args.min().item()) < 0 or int(atom_args.max().item()) >= num_objects
    ):
        raise ValueError("atom_args references an object outside object_batch")
    occurrence_batch = atom_batch.index_select(0, occurrence_atom_ids)
    if int(occurrence_batch.numel()):
        object_graphs = object_batch.index_select(0, atom_args)
        if bool((occurrence_batch != object_graphs).any()):
            raise ValueError("atom arguments must not connect separate graphs")

    if equality_pattern_ids is not None:
        equality_pattern_ids = _as_long(equality_pattern_ids)
        if int(equality_pattern_ids.numel()) != q:
            raise ValueError("equality_pattern_ids must contain one entry per atom")

    offsets = pair_support_offsets
    if int(offsets.numel()) != int(pair_objects.size(0)) + 1:
        raise ValueError("pair_support_offsets must have one entry per pair plus one")
    if int(offsets.numel()) and (
        int(offsets[0].item()) != 0
        or bool((offsets[1:] < offsets[:-1]).any())
        or bool((offsets < 0).any())
    ):
        raise ValueError(
            "pair_support_offsets must start at zero and be non-decreasing"
        )
    if int(offsets[-1].item()) != int(pair_support_atom_ids.numel()):
        raise ValueError("pair_support_offsets[-1] must equal the support list length")

    if int(pair_objects.numel()):
        if bool((pair_objects[:, 0] == pair_objects[:, 1]).any()):
            raise ValueError("pair_objects must not contain diagonal pairs")
        if (
            int(pair_objects.min().item()) < 0
            or int(pair_objects.max().item()) >= num_objects
        ):
            raise ValueError("pair_objects references an object outside object_batch")
        pair_graphs_left = object_batch.index_select(0, pair_objects[:, 0])
        pair_graphs_right = object_batch.index_select(0, pair_objects[:, 1])
        if bool((pair_graphs_left != pair_graphs_right).any()):
            raise ValueError("pair_objects must not connect separate graphs")
        if int(torch.unique(pair_objects, dim=0).size(0)) != int(pair_objects.size(0)):
            raise ValueError("pair_objects must not contain duplicate ordered pairs")

    support_count = int(pair_support_atom_ids.numel())
    if any(
        int(value.numel()) != support_count
        for value in (pair_support_i, pair_support_j)
    ):
        raise ValueError("pair support atom and position arrays must have equal length")

    support_occurrence_i = support_occurrence_j = None
    if support_count:
        if (
            int(pair_support_atom_ids.min().item()) < 0
            or int(pair_support_atom_ids.max().item()) >= q
        ):
            raise ValueError("pair support references an atom outside atom metadata")
        support_arities = arities.index_select(0, pair_support_atom_ids)
        if bool((pair_support_i < 0).any()) or bool((pair_support_j < 0).any()):
            raise ValueError("pair support positions must be non-negative")
        if bool((pair_support_i >= support_arities).any()) or bool(
            (pair_support_j >= support_arities).any()
        ):
            raise ValueError("pair support positions exceed their atom arity")
        support_pair_ids = torch.repeat_interleave(
            torch.arange(int(pair_objects.size(0)), dtype=torch.long),
            offsets[1:] - offsets[:-1],
        )
        support_starts = atom_starts.index_select(0, pair_support_atom_ids)
        support_occurrence_i = support_starts + pair_support_i
        support_occurrence_j = support_starts + pair_support_j
        mapped_pair_objects = pair_objects.index_select(0, support_pair_ids)
        if bool(
            (
                atom_args.index_select(0, support_occurrence_i)
                != mapped_pair_objects[:, 0]
            ).any()
        ) or bool(
            (
                atom_args.index_select(0, support_occurrence_j)
                != mapped_pair_objects[:, 1]
            ).any()
        ):
            raise ValueError("pair support entries disagree with pair_objects")

    if int(composition_triplets.numel()):
        if int(composition_triplets.min().item()) < 0 or int(
            composition_triplets.max().item()
        ) >= int(pair_objects.size(0)):
            raise ValueError(
                "composition_triplets references a pair outside pair_objects"
            )
        triplet_pairs = pair_objects.index_select(
            0, composition_triplets.reshape(-1)
        ).view(-1, 3, 2)
        target, left, right = (
            triplet_pairs[:, 0],
            triplet_pairs[:, 1],
            triplet_pairs[:, 2],
        )
        if (
            bool((target[:, 0] != left[:, 0]).any())
            or bool((target[:, 1] != right[:, 0]).any())
            or bool((left[:, 1] != right[:, 1]).any())
        ):
            raise ValueError(
                "composition_triplets must align target and side pairs on one witness object"
            )
        if bool((target == right).all(dim=1).any()) or bool(
            (target == left).all(dim=1).any()
        ):
            raise ValueError(
                "composition_triplets must use distinct target and side pairs"
            )

    if int(atom_pair_ids.numel()) != int(atom_pair_occurrence_i.numel()) or int(
        atom_pair_ids.numel()
    ) != int(atom_pair_occurrence_j.numel()):
        raise ValueError("atom pair arrays must have equal length")
    if int(atom_pair_ids.numel()):
        if int(atom_pair_ids.min().item()) < 0 or int(
            atom_pair_ids.max().item()
        ) >= int(pair_objects.size(0)):
            raise ValueError("atom_pair_ids references a pair outside pair_objects")
        for occurrence in (atom_pair_occurrence_i, atom_pair_occurrence_j):
            if (
                int(occurrence.min().item()) < 0
                or int(occurrence.max().item()) >= total_occurrences
            ):
                raise ValueError("atom pair occurrence index is outside atom_args")
        if bool((atom_pair_occurrence_i == atom_pair_occurrence_j).any()):
            raise ValueError("atom pair mappings must not contain diagonal occurrences")
        pair_atom_left = occurrence_atom_ids.index_select(0, atom_pair_occurrence_i)
        pair_atom_right = occurrence_atom_ids.index_select(0, atom_pair_occurrence_j)
        if bool((pair_atom_left != pair_atom_right).any()):
            raise ValueError("atom pair mappings must use positions from one atom")
        pair_left_objects = atom_args.index_select(0, atom_pair_occurrence_i)
        pair_right_objects = atom_args.index_select(0, atom_pair_occurrence_j)
        mapped_pairs = pair_objects.index_select(0, atom_pair_ids)
        if bool((pair_left_objects != mapped_pairs[:, 0]).any()) or bool(
            (pair_right_objects != mapped_pairs[:, 1]).any()
        ):
            raise ValueError("atom pair mappings disagree with pair_objects")

    if counterpart_occurrence_ids is not None:
        counterpart_occurrence_ids = _as_long(counterpart_occurrence_ids)
        if int(counterpart_occurrence_ids.numel()) != total_occurrences:
            raise ValueError(
                "counterpart_occurrence_ids must contain one entry per occurrence"
            )
        valid = counterpart_occurrence_ids >= 0
        if (
            bool(valid.any())
            and int(counterpart_occurrence_ids[valid].max().item()) >= total_occurrences
        ):
            raise ValueError(
                "counterpart_occurrence_ids references an occurrence outside the carrier"
            )
        if bool(valid.any()):
            counterpart_graphs = occurrence_batch.index_select(
                0, counterpart_occurrence_ids[valid]
            )
            if bool((counterpart_graphs != occurrence_batch[valid]).any()):
                raise ValueError(
                    "counterpart occurrence mappings must stay within a graph"
                )
            if bool((~goal_available.index_select(0, counterpart_graphs)).any()):
                raise ValueError(
                    "goal-free graphs must not carry goal counterpart mappings"
                )

    if object_carrier_occurrence_ids is not None:
        object_carrier_occurrence_ids = _as_long(object_carrier_occurrence_ids)
        if int(object_carrier_occurrence_ids.numel()) != num_objects:
            raise ValueError(
                "object_carrier_occurrence_ids must contain one entry per object"
            )
        if int(object_carrier_occurrence_ids.numel()) and (
            int(object_carrier_occurrence_ids.min().item()) < 0
            or int(object_carrier_occurrence_ids.max().item()) >= total_occurrences
        ):
            raise ValueError(
                "object_carrier_occurrence_ids references an occurrence outside the carrier"
            )
        carrier_objects = atom_args.index_select(0, object_carrier_occurrence_ids)
        if not torch.equal(
            carrier_objects, torch.arange(num_objects, dtype=torch.long)
        ):
            raise ValueError(
                "object_carrier_occurrence_ids must be ordered one per object"
            )
        carrier_graphs = occurrence_batch.index_select(0, object_carrier_occurrence_ids)
        if not torch.equal(carrier_graphs, object_batch):
            raise ValueError(
                "object carrier occurrences must stay within their object graphs"
            )


__all__ = [
    "AUXILIARY_OBJECT_PREDICATE",
    "CHANNEL_AUXILIARY",
    "CHANNEL_NAMES",
    "CHANNEL_SATISFIED",
    "CHANNEL_STATE",
    "CHANNEL_UNSATISFIED",
    "NULLARY_OBJECT_NAME",
    "EqualityPatternTable",
    "SparseAtomCompositionEncoder",
    "SparseAtomCompositionEncoding",
    "SparseAtomPredicateSchema",
    "batch_sparse_atom_encodings",
    "build_predicate_schema",
    "encode_sparse_atom_facts",
    "validate_sparse_atom_composition",
]
