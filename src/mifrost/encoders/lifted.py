"""Lifted planning-task graph encoder built on the custom-encoder toolkit.

`LiftedTaskEncoder` encodes the planning problem's own structure — objects,
predicate schemas, lifted action schemas with their parameters, preconditions
and (conditional) effects, and the goal — as one GNN-consumable homogeneous
graph. This is the task-structure encoding family discussed in
``docs/explanation/derived-encoding-strategies.md``: an ASG-style lifted task
graph in the spirit of the IPC graph dataset (Ferber et al. 2019), suited to
planner-selection-style task-level predictions.

The task structure is STATE-INDEPENDENT: predicates, actions, parameters and
lifted literals are schema-level facts of the problem, so `encode` returns a
byte-identical graph for every state of one problem unless
``include_state_facts=True`` (the only option that consults the state, adding
grounded ``fact`` edges for static plus fluent/derived atoms).

Channel layout (single ``node`` table; role ids 0=object, 1=predicate,
2=action, 3=parameter; ``x_ids`` width 5):

===== ==================== ============================================= ======
ch    content              per role                                      notes
===== ==================== ============================================= ======
0     role id              object=0, predicate=1, action=2, parameter=3
1     schema id + 1        predicate/action vocabulary position + 1;     0 for
                           parameters carry their owning action's id     objects
2     aux_position         predicate arity / action parameter count /
                           parameter index within its action
3     aux_sign             always 0 (polarity lives on edges)
4     aux_category         predicate category: static=0, fluent=1,       0 else
                           derived=2
===== ==================== ============================================= ======

Node names are exported by default: object/predicate/action nodes carry their
PDDL names, parameter nodes are named ``"<action>?a<i>"``.

Edge kinds (``edge_attr = [kind_id, pos_a, pos_b]``):

============= ======================= ===================== ================
kind          direction               pos_a                 pos_b
============= ======================= ===================== ================
``param_of``  parameter -> action     parameter index       0
``pre``       predicate -> action     0                     0
``eff_add``   predicate -> action     effect group index    0
``eff_del``   predicate -> action     effect group index    0
``eff_cond``  predicate -> action     condition group idx   0
``pre_arg``   parameter -> predicate  argument position     parameter index
``eff_add_arg`` parameter -> predicate argument position    parameter index
``eff_del_arg`` parameter -> predicate argument position    parameter index
``goal_pos``  predicate -> object     argument position     0
``goal_neg``  predicate -> object     argument position     0
``fact``      predicate -> object     argument position     0
============= ==================== ========================== ================

Group indices number conditional effects canonically: 0 is the unconditional
effect, 1.. enumerate conditional effect groups; a group's ``eff_cond`` edges
(condition literals) share the group index of its ``eff_add``/``eff_del``
edges. ``pre``/``eff_*`` literal edges are always emitted even when
``include_parameters=False``, so the action-predicate bipartite structure
survives; only ``param_of`` and the ``*_arg`` edges disappear with the
parameter nodes. Constant (non-parameter) literal arguments contribute no arg
edge — their literal keeps its main edge.

Downstream consumption:
- node channels give role, schema identity, arity/parameter counts and
  predicate categories directly;
- edge kinds, argument positions and parameter indices are recoverable from
  ``edge_attr`` columns;
- ``vocab_predicates`` / ``vocab_actions`` graph attributes list schema names
  in vocabulary order for embedding tables; ``vocab_edge_kinds`` is exported
  automatically; ``vocab_roles`` / ``vocab_categories`` and the encoder flags
  are plain graph attributes;
- for task-level predictions (planner selection), pool globally (mean over
  all nodes) into one graph embedding;
- batching: multiple problems OF ONE DOMAIN batch correctly because
  vocabulary ids are domain-scoped and identical across those problems.
  Cross-domain batching mixes id spaces and is unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .custom.base import CustomGraphEncoder, CustomStream
from .custom.state_view import ActionStructure, Atom, Literal
from .custom.writer import GraphWriter

ROLE_NAMES: tuple[str, ...] = ("object", "predicate", "action", "parameter")
CATEGORY_NAMES: tuple[str, ...] = ("static", "fluent", "derived")

_CATEGORY_IDS = {name: index for index, name in enumerate(CATEGORY_NAMES)}


class LiftedTaskEncoder(CustomGraphEncoder):
    """Schema-level task graph over objects, predicates and lifted actions.

    One node per object, predicate schema, action schema and action
    parameter, wired by lifted precondition/effect/goal edges (see module
    docstring for channel and edge tables). The input must be a *problem*
    (a ``pymimir.Problem`` or pytyr ``PlanningTask``); states are passed to
    `encode` / `encode_batch` and ignored unless
    ``include_state_facts=True``.
    """

    def __init__(
        self,
        problem_or_task: Any,
        *,
        include_parameters: bool = True,
        include_goal: bool = True,
        include_state_facts: bool = False,
        export_node_names: bool = True,
        backend: str | None = None,
    ) -> None:
        """Create one lifted task-structure encoder for one problem/task."""

        super().__init__(
            problem_or_task,
            export_node_names=export_node_names,
            backend=backend,
        )
        self.include_parameters = bool(include_parameters)
        self.include_goal = bool(include_goal)
        self.include_state_facts = bool(include_state_facts)

    def stream(self) -> LiftedTaskEncoderStream:
        """Create a pure-Python streaming variant of this encoder."""

        return LiftedTaskEncoderStream(self)

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
        """Draw the problem's lifted task graph into ``out``."""

        del kwargs
        encoder_name = type(self).__name__
        if goals:
            raise ValueError(
                f"{encoder_name} derives goals from the problem itself via "
                "StateView.goal_literals; got a non-empty goals lane"
            )
        if actions:
            raise ValueError(
                f"{encoder_name} encodes lifted task structure only; got a "
                "non-empty actions lane"
            )
        if subgoal_layers:
            raise ValueError(
                f"{encoder_name} encodes lifted task structure only; got "
                "non-empty subgoal_layers lane"
            )
        if history_subgoals:
            raise ValueError(
                f"{encoder_name} encodes lifted task structure only; got "
                "non-empty history_subgoals lane"
            )

        view = out.view
        predicates = out.vocabulary("predicates")
        for info in view.predicates:
            predicates.id_for(info.name)
        action_vocab = out.vocabulary("actions")
        for info in view.action_schemas:
            action_vocab.id_for(info.name)

        object_ids = {
            name: out.add_node(
                ("object", name),
                role="object",
                channels=(0, 0, 0, 0, 0),
                name=name,
            )
            for name in view.objects
        }

        predicate_ids: dict[str, int] = {}
        for info in view.predicates:
            predicate_ids[info.name] = out.add_node(
                ("predicate", info.name),
                role="predicate",
                channels=(
                    1,
                    predicates.id_for(info.name) + 1,
                    info.arity,
                    0,
                    _CATEGORY_IDS[info.category],
                ),
                name=info.name,
            )

        structures: dict[str, ActionStructure] = {
            structure.name: structure for structure in view.action_structures()
        }
        action_ids: dict[str, int] = {}
        for info in view.action_schemas:
            structure = structures[info.name]
            action_ids[info.name] = out.add_node(
                ("action", info.name),
                role="action",
                channels=(
                    2,
                    action_vocab.id_for(info.name) + 1,
                    len(structure.parameters),
                    0,
                    0,
                ),
                name=info.name,
            )

        add_edge = out.add_edge
        for info in view.action_schemas:
            structure = structures[info.name]
            action_id = action_ids[info.name]
            parameter_ids: list[int] | None = None
            if self.include_parameters:
                parameter_ids = []
                schema_id = action_vocab.id_for(info.name) + 1
                for index, parameter_name in enumerate(structure.parameters):
                    parameter_ids.append(
                        out.add_node(
                            ("parameter", info.name, index),
                            role="parameter",
                            channels=(3, schema_id, index, 0, 0),
                            name=f"{info.name}{parameter_name}",
                        )
                    )
                    add_edge(parameter_ids[-1], action_id, "param_of", pos_a=index)
            for literal in structure.precondition:
                predicate_id = predicate_ids[literal.atom.predicate]
                add_edge(predicate_id, action_id, "pre")
                if parameter_ids is not None:
                    _add_argument_edges(
                        add_edge,
                        parameter_ids,
                        predicate_id,
                        literal.atom.args,
                        "pre_arg",
                    )
            next_conditional_group = 1
            for effect in structure.effects:
                if effect.condition:
                    group_index = next_conditional_group
                    next_conditional_group += 1
                    for literal in effect.condition:
                        add_edge(
                            predicate_ids[literal.atom.predicate],
                            action_id,
                            "eff_cond",
                            pos_a=group_index,
                        )
                else:
                    group_index = 0
                for literal in effect.literals:
                    kind = "eff_add" if literal.positive else "eff_del"
                    predicate_id = predicate_ids[literal.atom.predicate]
                    add_edge(predicate_id, action_id, kind, pos_a=group_index)
                    if parameter_ids is not None:
                        _add_argument_edges(
                            add_edge,
                            parameter_ids,
                            predicate_id,
                            literal.atom.args,
                            f"{kind}_arg",
                        )

        if self.include_goal:
            for literal in view.goal_literals(state):
                kind = "goal_pos" if literal.positive else "goal_neg"
                predicate_id = predicate_ids[literal.atom.predicate]
                for position, argument in enumerate(literal.atom.args):
                    add_edge(predicate_id, object_ids[argument], kind, pos_a=position)

        if self.include_state_facts:
            for atom in (*view.static_facts, *view.state_facts(state)):
                predicate_id = predicate_ids[atom.predicate]
                for position, argument in enumerate(atom.args):
                    add_edge(predicate_id, object_ids[argument], "fact", pos_a=position)

        out.set_attr("vocab_roles", list(ROLE_NAMES))
        out.set_attr("vocab_categories", list(CATEGORY_NAMES))
        out.set_attr("include_parameters", int(self.include_parameters))
        out.set_attr("include_goal", int(self.include_goal))
        out.set_attr("include_state_facts", int(self.include_state_facts))
        out.set_vocab_attr("predicates")
        out.set_vocab_attr("actions")
        out.set_flag("include_reverse_edges", True)


def _add_argument_edges(
    add_edge: Any,
    parameter_ids: list[int],
    predicate_id: int,
    arguments: tuple[str, ...],
    kind: str,
) -> None:
    """Wire parameter nodes to one literal's predicate for parameter args."""

    for position, argument in enumerate(arguments):
        if not argument.startswith("?"):
            continue
        parameter_index = int(argument[2:])
        add_edge(
            parameter_ids[parameter_index],
            predicate_id,
            kind,
            pos_a=position,
            pos_b=parameter_index,
        )


@dataclass
class LiftedTaskEncoderStream(CustomStream):
    """Streaming variant of `LiftedTaskEncoder`.

    Stores cheap ``(state, kwargs)`` recipes until `flush` re-encodes them;
    nothing is computed until then, so appended states must outlive the
    stream. Only the inherited ``encoder`` field's typing is narrowed.
    """

    encoder: LiftedTaskEncoder


__all__ = ["LiftedTaskEncoder", "LiftedTaskEncoderStream"]
