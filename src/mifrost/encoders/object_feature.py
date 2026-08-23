"""Objects-only feature-channel encoder built on the custom-encoder toolkit.

`ObjectFeatureEncoder` closes a documented gap of the native objects-only
derived views: those projections drop every fact with fewer than two
arguments, so unary state such as ``(clear a)`` or ``(handempty)`` leaves no
trace in the graph (see ``docs/explanation/derived-encoding-strategies.md``).
This encoder keeps the same compact objects-only graph shape but turns unary
predicates into per-object FEATURE CHANNELS instead of losing them.

Channel layout (single ``node`` table, role ``"object"``, one node per domain
object, in `StateView.objects` order):

===== =========================================================== ======
ch    content                                                     flag
===== =========================================================== ======
0     role id (always 0 = object)                                 always
1..U  unary predicate vocabulary id + 1 if the fact holds else 0  ``include_unary_features``
nextR participation count per non-unary predicate                 ``include_participation_counts``
last  1 if the object appears in any goal literal else 0          ``include_goal_flags``
===== =========================================================== ======

Unary channels cover static unary facts as well as fluent ones; both hold in
every encoded state by definition. Participation counts aggregate state and
static facts. Nullary facts cannot attach to any object: they are skipped,
but reported through `ObjectFeatureEncoder.last_nullary_count`.

Every state/static fact of arity >= 2 becomes directed edges between its
argument objects per ``expansion`` — ``"clique"`` (all ordered pairs),
``"chain"`` (consecutive arguments) or ``"star_first"`` (first argument to
the rest). Edges carry ``(kind_id, pos_a, pos_b)`` attributes exactly like
the native derived family; both directions are emitted explicitly with
distinct ``*_fwd`` / ``*_bwd`` kind names and the schema flag
``include_reverse_edges`` is set so PyG conversion does not mirror them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .custom.base import CustomGraphEncoder, CustomStream
from .custom.state_view import Atom, Literal
from .custom.writer import GraphWriter

EXPANSIONS: tuple[str, ...] = ("clique", "chain", "star_first")

_ROLE_OBJECT = "object"


class ObjectFeatureEncoder(CustomGraphEncoder):
    """Objects-only state graph with unary predicates as feature channels.

    One node per domain object; facts of arity >= 2 wire the object nodes
    together according to ``expansion`` while unary predicates become
    per-object integer-id channels (see module docstring for the layout).

    The input must be a *problem* (a ``pymimir.Problem`` or a pytyr
    ``PlanningTask``); states are passed to `encode` / `encode_batch`.

    ``last_nullary_count`` reports how many nullary facts the most recent
    `encode_state` call skipped ("last encode wins"): it is per-state
    scratch, not an aggregate — reading it after a batch or stream flush
    yields the count of whichever state was encoded last, so capture it
    right after encoding the state you care about.
    """

    def __init__(
        self,
        problem_or_task: Any,
        *,
        expansion: str = "clique",
        include_unary_features: bool = True,
        include_participation_counts: bool = False,
        include_goal_flags: bool = False,
        export_node_names: bool = True,
        backend: str | None = None,
    ) -> None:
        """Create one objects-only feature encoder for one problem/task."""

        if expansion not in EXPANSIONS:
            raise ValueError(
                f"expansion must be one of {list(EXPANSIONS)}, got {expansion!r}"
            )
        super().__init__(
            problem_or_task,
            export_node_names=export_node_names,
            backend=backend,
        )
        self.expansion = expansion
        self.include_unary_features = include_unary_features
        self.include_participation_counts = include_participation_counts
        self.include_goal_flags = include_goal_flags
        self.last_nullary_count = 0
        # Schema-derived invariants of this problem, hoisted out of encode_state.
        self._predicate_names: tuple[str, ...] | None = None
        self._unary_names: tuple[str, ...] = ()
        self._non_unary_names: tuple[str, ...] = ()

    def _hoist_schema(self) -> None:
        """Cache predicate partitions once; ids follow view.predicates order."""

        if self._predicate_names is not None:
            return
        view = self.view
        self._predicate_names = tuple(info.name for info in view.predicates)
        self._unary_names = tuple(
            info.name for info in view.predicates if info.arity == 1
        )
        self._non_unary_names = tuple(
            info.name for info in view.predicates if info.arity != 1
        )

    def stream(self) -> ObjectFeatureEncoderStream:
        """Create a pure-Python streaming variant of this encoder."""

        return ObjectFeatureEncoderStream(self)

    def encode_state(
        self,
        out: GraphWriter,
        state: Any,
        *,
        goals: tuple[Literal, ...] | None = None,
        actions: tuple[Atom, ...] | None = None,
        subgoal_layers: tuple[tuple[Literal, ...], ...] | None = None,
        history_subgoals: tuple[tuple[int, tuple[Literal, ...]], ...] | None = None,
        history_max_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Draw one state's objects-only feature graph into ``out``."""

        del kwargs
        encoder_name = type(self).__name__
        if actions:
            raise ValueError(
                f"{encoder_name} encodes states only; got a non-empty actions lane"
            )
        if subgoal_layers:
            raise ValueError(
                f"{encoder_name} encodes states only; got non-empty subgoal_layers lane"
            )
        if history_subgoals:
            raise ValueError(
                f"{encoder_name} encodes states only; got non-empty "
                "history_subgoals lane"
            )
        if history_max_steps is not None:
            raise ValueError(
                f"{encoder_name} encodes states only; got "
                f"history_max_steps={history_max_steps!r}"
            )

        self._hoist_schema()
        view = out.view
        include_unary = self.include_unary_features
        include_counts = self.include_participation_counts
        include_goals = self.include_goal_flags

        # Seed the writer's vocabulary in view.predicates order exactly as the
        # per-state id_for loop did, so every channel id stays identical.
        predicates = out.vocabulary("predicates")
        for name in self._predicate_names:
            predicates.id_for(name)
        id_of_predicate = predicates._ids

        facts: tuple[Atom, ...] = (*view.static_facts, *view.state_facts(state))
        literals = view.goal_literals(state) if goals is None else tuple(goals)

        unary_holds: dict[str, set[str]] = {}
        counts: Counter[tuple[str, str]] = Counter()
        nullary_count = 0
        for atom in facts:
            args = atom.args
            arity = len(args)
            if arity == 0:
                nullary_count += 1
                continue
            if arity == 1 and include_unary:
                bucket = unary_holds.get(atom.predicate)
                if bucket is None:
                    unary_holds[atom.predicate] = {args[0]}
                else:
                    bucket.add(args[0])
            elif arity > 1:
                if not include_counts:
                    continue
                predicate = atom.predicate
                for arg in args:
                    counts[(predicate, arg)] += 1
        self.last_nullary_count = nullary_count

        goal_objects = {arg for literal in literals for arg in literal.atom.args}

        objects = view.objects
        empty_set: frozenset[str] = frozenset()
        unary_ids = (
            [id_of_predicate[name] + 1 for name in self._unary_names]
            if include_unary
            else ()
        )
        channels: list[list[int]] = []
        add_channel_row = channels.append
        for name in objects:
            row = [0]
            if include_unary:
                for base_id, unary_name in zip(unary_ids, self._unary_names):
                    row.append(
                        base_id if name in unary_holds.get(unary_name, empty_set) else 0
                    )
            if include_counts:
                row.extend(
                    counts.get((info, name), 0) for info in self._non_unary_names
                )
            if include_goals:
                row.append(1 if name in goal_objects else 0)
            add_channel_row(row)

        object_ids = {
            name: out.add_node(
                (_ROLE_OBJECT, name),
                role=_ROLE_OBJECT,
                channels=row,
                name=name,
            )
            for name, row in zip(objects, channels)
        }

        expansion = self.expansion
        add_both = out.add_both
        clique_kinds = ("clique_fwd", "clique_bwd")
        chain_kinds = ("chain_fwd", "chain_bwd")
        star_kinds = ("star_first_fwd", "star_first_bwd")
        for atom in facts:
            args = atom.args
            if len(args) < 2:
                continue
            ids = [object_ids[arg] for arg in args]
            if expansion == "clique":
                kinds = clique_kinds
                count = len(ids)
                for i in range(count):
                    src = ids[i]
                    for j in range(i + 1, count):
                        add_both(src, ids[j], *kinds, pos_a=i, pos_b=j)
            elif expansion == "chain":
                kinds = chain_kinds
                for i in range(len(ids) - 1):
                    add_both(ids[i], ids[i + 1], *kinds, pos_a=i, pos_b=i + 1)
            else:
                kinds = star_kinds
                first = ids[0]
                for j in range(1, len(ids)):
                    add_both(first, ids[j], *kinds, pos_a=0, pos_b=j)

        out.set_vocab_attr("predicates")
        out.set_flag("include_reverse_edges", True)


@dataclass
class ObjectFeatureEncoderStream(CustomStream):
    """Streaming variant of `ObjectFeatureEncoder`.

    Stores cheap ``(state, kwargs)`` recipes until `flush` re-encodes them;
    nothing is computed until then, so appended states must outlive the
    stream. Only the inherited ``encoder`` field's typing is narrowed.
    """

    encoder: ObjectFeatureEncoder


__all__ = ["ObjectFeatureEncoder", "ObjectFeatureEncoderStream"]
