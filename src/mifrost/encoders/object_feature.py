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

        view = out.view
        predicates = out.vocabulary("predicates")
        for info in view.predicates:
            predicates.id_for(info.name)

        facts: tuple[Atom, ...] = (*view.static_facts, *view.state_facts(state))
        literals = view.goal_literals(state) if goals is None else tuple(goals)

        unary = [info for info in view.predicates if info.arity == 1]
        non_unary = [info for info in view.predicates if info.arity != 1]
        holds = {
            (atom.predicate, atom.args[0]) for atom in facts if len(atom.args) == 1
        }
        counts: dict[tuple[str, str], int] = {}
        nullary_count = 0
        for atom in facts:
            arity = len(atom.args)
            if arity == 0:
                nullary_count += 1
                continue
            if arity == 1:
                continue
            for arg in atom.args:
                key = (atom.predicate, arg)
                counts[key] = counts.get(key, 0) + 1
        self.last_nullary_count = nullary_count

        goal_objects = {arg for literal in literals for arg in literal.atom.args}

        channels: dict[str, list[int]] = {}
        for name in view.objects:
            row = [0]
            if self.include_unary_features:
                row.extend(
                    predicates.id_for(info.name) + 1
                    if (info.name, name) in holds
                    else 0
                    for info in unary
                )
            if self.include_participation_counts:
                row.extend(counts.get((info.name, name), 0) for info in non_unary)
            if self.include_goal_flags:
                row.append(1 if name in goal_objects else 0)
            channels[name] = row

        object_ids = {
            name: out.add_node(
                (_ROLE_OBJECT, name),
                role=_ROLE_OBJECT,
                channels=channels[name],
                name=name,
            )
            for name in view.objects
        }

        for atom in facts:
            args = atom.args
            if len(args) < 2:
                continue
            ids = [object_ids[arg] for arg in args]
            if self.expansion == "clique":
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        out.add_both(
                            ids[i],
                            ids[j],
                            "clique_fwd",
                            "clique_bwd",
                            pos_a=i,
                            pos_b=j,
                        )
            elif self.expansion == "chain":
                for i in range(len(ids) - 1):
                    out.add_both(
                        ids[i],
                        ids[i + 1],
                        "chain_fwd",
                        "chain_bwd",
                        pos_a=i,
                        pos_b=i + 1,
                    )
            else:
                for j in range(1, len(ids)):
                    out.add_both(
                        ids[0],
                        ids[j],
                        "star_first_fwd",
                        "star_first_bwd",
                        pos_a=0,
                        pos_b=j,
                    )

        for kind_name in out.edges.kinds.names():
            out.vocabulary("edge_kinds").id_for(kind_name)
        out.set_vocab_attr("predicates")
        out.set_vocab_attr("edge_kinds")
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
