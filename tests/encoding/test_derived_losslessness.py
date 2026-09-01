"""Losslessness invariants for the derived-graph encoder family.

Every encoder in this family must carry the full information load of the
STRIPS state it encodes. The encoded object is the *literal-instance
multiset* of one state::

    instance := (role, relation, arguments, sign, goal_level, history_dt, category)

An encoder is lossless iff that multiset is exactly recoverable from the
tensors it emits plus the graph-attribute vocabularies. These tests assert
that recovery directly -- they rebuild the multiset from each view's tensors
and compare it against the reified star view's reconstruction -- and then
pin the individual mechanisms that used to lose information:

* the pairwise ``objects_only`` projection labelled every edge with only
  ``(kind, pos_a, pos_b)``, so a goal literal was byte-identical to a state
  fact and arity-0/1 literals vanished entirely;
* action-schema ids collided with predicate ids in one unlabelled id space;
* an arity-0 instance produced a zero-member hyperedge, which desynced
  ``HypergraphConv``'s inferred ``num_edges``;
* a repeated instance duplicated graph edges while interning one node;
* ``line_share`` ignored ``include_reverse_edges``;
* the documented embedding recipe crashed on the signed history channel;
* ``node_names`` collided for subgoal levels >= 3;
* ``x`` shipped as an all-zero placeholder that silently produced a constant
  graph in any stock layer.

Both fixtures are exercised with goals, a negated goal literal, subgoal
layers, history and grounded actions all supplied. ``tri/p1`` is the only
bundled problem with an arity >= 3 predicate (``between ?x ?y ?z``) plus a
binary ``link`` and a nullary ``flag``, which is what separates a genuinely
lossless projection from one that only looks lossless on binary domains.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import pytest
import torch
from torch_geometric.data import Batch

from mifrost.encoders.derived import (
    AtomLineGraphEncoder,
    HypergraphIncidenceEncoder,
    ObjectGraphEncoder,
    StarGraphEncoder,
    TransformerBiasEncoder,
    TupleTensorEncoder,
)
from mifrost.encoders.derived_graph_data import (
    normalize_derived_graph_batch_metadata,
)

try:
    from tests.conftest import load_problem
except ImportError:  # pragma: no cover - wheel test layouts
    from conftest import load_problem  # type: ignore[no-redef]


# --------------------------------------------------------------------------
# channel layout (mirrors the frozen contract; hard-coded on purpose so a
# silent reordering of the columns is caught here rather than downstream)
# --------------------------------------------------------------------------

ROLE, RELATION, SIGN, GOAL_LEVEL, HISTORY_DT, CATEGORY = range(6)
EDGE_KIND, EDGE_POS_A, EDGE_POS_B = 0, 1, 2
#: ``edge_attr`` columns 3..8 repeat the six ``x_ids`` channels of the
#: instance the edge was derived from, in ``x_ids`` order but with the
#: relation first (``rel_id_plus_one, role, sign, goal_level, history_dt,
#: category``).
EDGE_LABEL_SLICE = slice(3, 9)

ROLE_OBJECT, ROLE_FACT, ROLE_GOAL = 0, 1, 2
ROLE_SUBGOAL, ROLE_HISTORY, ROLE_ACTION, ROLE_ANCHOR = 3, 4, 5, 6

ANCHOR_NODE_NAME = "<nullary>"

EXPECTED_CHANNEL_NAMES = [
    "role",
    "relation_id_plus_one",
    "sign",
    "goal_level",
    "history_dt",
    "category",
]
EXPECTED_EDGE_CHANNEL_NAMES = [
    "kind",
    "pos_a",
    "pos_b",
    "rel_id_plus_one",
    "role",
    "sign",
    "goal_level",
    "history_dt",
    "category",
]
EXPECTED_ROLE_NAMES = [
    "object",
    "fact",
    "goal",
    "subgoal",
    "history",
    "action",
    "anchor",
]
EXPECTED_EDGE_KIND_NAMES = [
    "arg_fwd",
    "arg_bwd",
    "clique_fwd",
    "clique_bwd",
    "chain_fwd",
    "chain_bwd",
    "star_first_fwd",
    "star_first_bwd",
    "nullary_self",
    "action_fwd",
    "action_bwd",
    "line_share",
    "unary_self",
]
EXPECTED_CATEGORY_NAMES = ["static", "fluent", "derived", "action"]


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

#: ``(predicate, arguments, positive)`` literal specs per domain, chosen so
#: every lane and every id channel is populated: two goals of which one is
#: negated, one subgoal layer, one history step, plus grounded actions.
_LANE_SPECS: dict[str, dict[str, Any]] = {
    "blocks": {
        "goals": [("on", ("c", "b"), True), ("clear", ("c",), False)],
        "subgoal_layers": [[("on", ("b", "a"), True)]],
        "history_subgoals": [(-1, [("clear", ("b",), True)])],
        "nullary": ("handempty", ()),
        "binary": ("on", ("a", "b")),
    },
    "tri": {
        "goals": [("between", ("d", "b", "a"), True), ("link", ("c", "d"), False)],
        "subgoal_layers": [[("between", ("a", "b", "c"), True)]],
        "history_subgoals": [(-1, [("flag", (), True)])],
        "nullary": ("flag", ()),
        "binary": ("link", ("c", "d")),
    },
}

FIXTURE_CASES = (("blocks", "smedium"), ("tri", "p1"))


@dataclass(frozen=True)
class Case:
    """One problem plus the lane payloads used to encode it."""

    domain: str
    problem_name: str
    problem: Any
    state: Any

    def literal(self, predicate: str, args: Iterable[str], positive: bool = True):
        """Build one ground literal from predicate/object *names*."""
        predicates = {
            p.get_name(): p for p in self.problem.get_domain().get_predicates()
        }
        objects = {o.get_name(): o for o in self.problem.get_objects()}
        atom = self.problem.new_ground_atom(
            predicates[predicate], [objects[name] for name in args]
        )
        return self.problem.new_ground_literal(atom, positive)

    @property
    def spec(self) -> dict[str, Any]:
        return _LANE_SPECS[self.domain]

    def lanes(self, **overrides: Any) -> dict[str, Any]:
        """Goals (one negated), a subgoal layer, history and one action."""
        spec = self.spec
        lanes = {
            "goals": [self.literal(*entry) for entry in spec["goals"]],
            "subgoal_layers": [
                [self.literal(*entry) for entry in layer]
                for layer in spec["subgoal_layers"]
            ],
            "history_subgoals": [
                (dt, [self.literal(*entry) for entry in layer])
                for dt, layer in spec["history_subgoals"]
            ],
            "actions": list(self.state.generate_applicable_actions()[:1]),
        }
        lanes.update(overrides)
        return lanes


@pytest.fixture(
    scope="module",
    params=FIXTURE_CASES,
    ids=[f"{domain}-{problem}" for domain, problem in FIXTURE_CASES],
)
def case(request) -> Case:
    domain, problem_name = request.param
    _domain, problem, state, _dpath, _ppath = load_problem(domain, problem_name)
    return Case(domain, problem_name, problem, state)


#: The six public facades, plus the two remaining ``objects_only`` expansions
#: (they share ``ObjectGraphEncoder`` but have different topology rules).
FACADES: dict[str, Callable[[Any], Any]] = {
    "star": lambda problem: StarGraphEncoder(problem),
    "object_clique": lambda problem: ObjectGraphEncoder(
        problem, atom_expansion="clique"
    ),
    "object_chain": lambda problem: ObjectGraphEncoder(problem, atom_expansion="chain"),
    "object_star_first": lambda problem: ObjectGraphEncoder(
        problem, atom_expansion="star_first"
    ),
    "line": lambda problem: AtomLineGraphEncoder(problem),
    "hyper": lambda problem: HypergraphIncidenceEncoder(problem),
    "tuple": lambda problem: TupleTensorEncoder(problem),
    "spd": lambda problem: TransformerBiasEncoder(problem),
}

#: One entry per public facade class -- what "for every facade" means for the
#: batching and carrier-shape invariants.
SIX_FACADES = ("star", "object_clique", "line", "hyper", "tuple", "spd")

#: Views that materialize a node per literal instance.
REIFIED_FACADES = ("star", "line", "hyper", "tuple")

#: Views that only materialize object nodes (plus actions and the anchor).
OBJECTS_ONLY_FACADES = ("object_clique", "object_chain", "object_star_first", "spd")

#: Views that carry the tuple instance table.
TUPLE_TABLE_FACADES = (
    "object_clique",
    "object_chain",
    "object_star_first",
    "tuple",
    "spd",
)


# --------------------------------------------------------------------------
# reconstruction helpers -- each rebuilds the instance multiset from one
# view's tensors and the graph-attribute vocabularies, nothing else
# --------------------------------------------------------------------------


def _object_names(data: Any) -> list[str]:
    """Object node index -> object name.

    Objects are always the leading node block in both universes, so this list
    doubles as the index->name map for every argument reference.
    """
    return [str(name) for name in data.object_names]


def _instances_from_reified_nodes(data: Any) -> collections.Counter:
    """Rebuild the multiset from the node table plus the argument edges."""
    kinds = list(data.vocab_edge_kinds)
    arg_fwd = kinds.index("arg_fwd")
    action_fwd = kinds.index("action_fwd")
    names = _object_names(data)

    x_ids = data.x_ids.long()
    edge_index = data.edge_index
    edge_attr = data.edge_attr.long()

    arguments: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for column in range(edge_index.size(1)):
        kind = int(edge_attr[column, EDGE_KIND])
        if kind in (arg_fwd, action_fwd):
            arguments[int(edge_index[0, column])].append(
                (int(edge_attr[column, EDGE_POS_A]), int(edge_index[1, column]))
            )

    found: collections.Counter = collections.Counter()
    for node in range(x_ids.size(0)):
        role = int(x_ids[node, ROLE])
        if role in (ROLE_OBJECT, ROLE_ANCHOR):
            continue
        args = tuple(names[target] for _pos, target in sorted(arguments[node]))
        found[
            (
                role,
                int(x_ids[node, RELATION]) - 1,
                args,
                int(x_ids[node, SIGN]),
                int(x_ids[node, GOAL_LEVEL]),
                int(x_ids[node, HISTORY_DT]),
                int(x_ids[node, CATEGORY]),
            )
        ] += 1
    return found


def _instances_from_tuple_table(data: Any) -> collections.Counter:
    """Rebuild the multiset from the tuple instance table."""
    names = _object_names(data)
    args = data.tuple_args.long().view(-1)
    ptr = data.tuple_ptr.long().view(-1)
    attrs = data.tuple_attr_ids.long()

    found: collections.Counter = collections.Counter()
    for row in range(attrs.size(0)):
        arg_names = tuple(
            names[index] for index in args[int(ptr[row]) : int(ptr[row + 1])].tolist()
        )
        found[
            (
                int(attrs[row, ROLE]),
                int(attrs[row, RELATION]) - 1,
                arg_names,
                int(attrs[row, SIGN]),
                int(attrs[row, GOAL_LEVEL]),
                int(attrs[row, HISTORY_DT]),
                int(attrs[row, CATEGORY]),
            )
        ] += 1
    return found


def _instances_from_hyperedges(data: Any) -> collections.Counter:
    """Rebuild the multiset from hyperedge membership plus the label rows."""
    names = _object_names(data)
    anchor = data.anchor_node_index
    attrs = data.hyperedge_attr_ids.long()
    membership = data.hyperedge_index.long()

    members: dict[int, list[int]] = collections.defaultdict(list)
    for column in range(membership.size(1)):
        members[int(membership[1, column])].append(int(membership[0, column]))

    found: collections.Counter = collections.Counter()
    for row in range(attrs.size(0)):
        arg_names = tuple(names[index] for index in members[row] if index != anchor)
        found[
            (
                int(attrs[row, ROLE]),
                int(attrs[row, RELATION]) - 1,
                arg_names,
                int(attrs[row, SIGN]),
                int(attrs[row, GOAL_LEVEL]),
                int(attrs[row, HISTORY_DT]),
                int(attrs[row, CATEGORY]),
            )
        ] += 1
    return found


def _reconstruct(facade_name: str, data: Any) -> collections.Counter:
    """Rebuild the instance multiset from whatever lane this view exposes."""
    if facade_name == "hyper":
        return _instances_from_hyperedges(data)
    if facade_name in TUPLE_TABLE_FACADES:
        return _instances_from_tuple_table(data)
    return _instances_from_reified_nodes(data)


_RELATION_HEAD = re.compile(r"\(([^\s()]+)")


def _relation_head(node_name: str) -> str:
    """Return the relation named by a rendered node name.

    ``(on a b)`` -> ``on``; ``(not (clear c))[g]`` -> ``clear``;
    ``@(unstack a b)`` -> ``unstack``; ``(handempty)`` -> ``handempty``.
    """
    return _RELATION_HEAD.findall(node_name)[-1]


def _tuple_label_rows(data: Any) -> set[tuple[int, ...]]:
    """The distinct ``edge_attr`` label suffixes the instance table implies."""
    attrs = data.tuple_attr_ids.long()
    return {
        (
            int(row[RELATION]),
            int(row[ROLE]),
            int(row[SIGN]),
            int(row[GOAL_LEVEL]),
            int(row[HISTORY_DT]),
            int(row[CATEGORY]),
        )
        for row in attrs
    }


def _edge_label_rows(data: Any) -> set[tuple[int, ...]]:
    """The distinct label suffixes actually present in ``edge_attr``."""
    return {tuple(row) for row in data.edge_attr[:, EDGE_LABEL_SLICE].long().tolist()}


# --------------------------------------------------------------------------
# §9.1 -- round trip
# --------------------------------------------------------------------------


def test_reference_reconstruction_covers_every_lane(case: Case) -> None:
    """The star reference itself must contain the distinguishing instances.

    Without this, a round trip between two equally-degenerate views would
    pass vacuously.
    """
    data = FACADES["star"](case.problem).encode_pyg(case.state, **case.lanes())
    found = _reconstruct("star", data)
    relations = list(data.vocab_relations)
    roles = {role for role, *_rest in found}

    assert roles == {ROLE_FACT, ROLE_GOAL, ROLE_SUBGOAL, ROLE_HISTORY, ROLE_ACTION}
    # a negated goal literal
    assert any(role == ROLE_GOAL and sign == 1 for role, _r, _a, sign, *_ in found)
    # a subgoal on layer 1
    assert any(
        role == ROLE_SUBGOAL and level == 1 for role, _r, _a, _s, level, *_ in found
    )
    # a history literal with a negative dt
    assert any(
        role == ROLE_HISTORY and dt < 0 for role, _r, _a, _s, _l, dt, _c in found
    )
    # an action row, in the action category, over an action-schema relation
    action_rows = [entry for entry in found if entry[0] == ROLE_ACTION]
    assert action_rows
    for role, relation, _args, _sign, _level, _dt, category in action_rows:
        assert category == EXPECTED_CATEGORY_NAMES.index("action")
        assert relations[relation] in list(data.vocab_actions)
    # an arity-0 literal and at least one arity >= 2 literal
    assert any(len(args) == 0 for _r, _rel, args, *_rest in found)
    assert any(len(args) >= 2 for _r, _rel, args, *_rest in found)


@pytest.mark.parametrize("facade_name", sorted(FACADES))
def test_every_view_round_trips_the_instance_multiset(
    facade_name: str, case: Case
) -> None:
    """§9.1: the literal-instance multiset survives every view intact.

    Each view is reconstructed from its own tensors alone (plus the exported
    vocabularies) and compared against the reified star view. The projections
    used to lose the relation, the role, the sign, the goal level, the history
    age and the category on every edge, and to drop arity-0/1 literals
    outright, so this comparison failed for all four ``objects_only``
    variants and for the hypergraph.
    """
    lanes = case.lanes()
    reference = _reconstruct(
        "star", FACADES["star"](case.problem).encode_pyg(case.state, **lanes)
    )
    assert sum(reference.values()) > 0

    data = FACADES[facade_name](case.problem).encode_pyg(case.state, **lanes)
    assert _reconstruct(facade_name, data) == reference


# --------------------------------------------------------------------------
# §9.2 -- the unified relation id space
# --------------------------------------------------------------------------


@pytest.mark.parametrize("facade_name", sorted(FACADES))
def test_relation_channel_decodes_to_the_right_predicate_or_action(
    facade_name: str, case: Case
) -> None:
    """§9.2: ``x_ids[:, 1]`` names the right relation, never the wrong one.

    Predicate ids and action-schema ids used to share one id space, so an
    action node claiming schema 3 decoded to ``vocab_predicates[3]`` -- a
    predicate name. Schemas are now shifted past ``num_predicates`` and
    ``vocab_relations`` covers both halves.
    """
    data = FACADES[facade_name](case.problem).encode_pyg(case.state, **case.lanes())
    relations = list(data.vocab_relations)
    predicates = list(data.vocab_predicates)
    actions = list(data.vocab_actions)
    num_predicates = int(data.num_predicates)

    assert relations == predicates + actions
    assert num_predicates == len(predicates)

    x_ids = data.x_ids.long()
    names = list(data.node_names)
    seen_roles = set()
    for node in range(x_ids.size(0)):
        role = int(x_ids[node, ROLE])
        if role in (ROLE_OBJECT, ROLE_ANCHOR):
            assert int(x_ids[node, RELATION]) == 0
            continue
        seen_roles.add(role)
        relation = int(x_ids[node, RELATION]) - 1
        assert 0 <= relation < len(relations)
        if role == ROLE_ACTION:
            # The defect: an action row decoding into the predicate range.
            assert relation >= num_predicates
            assert relations[relation] == actions[relation - num_predicates]
        else:
            assert relation < num_predicates
            assert relations[relation] == predicates[relation]
        assert relations[relation] == _relation_head(names[node])

    if facade_name in OBJECTS_ONLY_FACADES:
        # Only actions are reified there, but they must still decode.
        assert seen_roles <= {ROLE_ACTION}
    else:
        assert ROLE_ACTION in seen_roles

    if facade_name in TUPLE_TABLE_FACADES:
        roles = data.tuple_role_ids.long()
        rel_ids = data.tuple_rel_ids.long()
        for row in range(rel_ids.numel()):
            relation = int(rel_ids[row])
            assert 0 <= relation < len(relations)
            if int(roles[row]) == ROLE_ACTION:
                assert relation >= num_predicates
            else:
                assert relation < num_predicates


# --------------------------------------------------------------------------
# §9.3 / §9.4 -- the objects-only projection carries the whole label
# --------------------------------------------------------------------------


@pytest.mark.parametrize("facade_name", OBJECTS_ONLY_FACADES)
def test_objects_only_edge_labels_separate_every_distinct_instance(
    facade_name: str, case: Case
) -> None:
    """§9.3: distinct instances leave distinct ``edge_attr`` rows.

    The projection used to label every edge ``(kind, pos_a, pos_b)`` only, so
    a state fact and a goal literal over the same objects produced identical
    rows. Columns 3..8 now repeat the originating instance's six channels, and
    the set of label suffixes present in ``edge_attr`` must be exactly the set
    of distinct instance labels -- no collapsing, no invented rows.
    """
    data = FACADES[facade_name](case.problem).encode_pyg(case.state, **case.lanes())
    assert data.edge_attr.size(1) == len(EXPECTED_EDGE_CHANNEL_NAMES)
    assert _edge_label_rows(data) == _tuple_label_rows(data)


@pytest.mark.parametrize("facade_name", OBJECTS_ONLY_FACADES)
def test_objects_only_labels_distinguish_instances_over_identical_arguments(
    facade_name: str, case: Case
) -> None:
    """§9.3, pointed: one relation, one argument pair, seven instances.

    Every channel that can vary while relation and arguments stay fixed is
    varied here: role (fact / goal / subgoal / history), sign, goal level and
    history age. Each must reach the projection as its own labelled edge.
    """
    predicate, (first, second) = case.spec["binary"]

    def literal(positive: bool = True):
        return case.literal(predicate, (first, second), positive)

    data = FACADES[facade_name](case.problem).encode_pyg(
        case.state,
        goals=[literal(), literal(False)],
        subgoal_layers=[[literal()], [literal()]],
        history_subgoals=[(-1, [literal()]), (-2, [literal()])],
    )

    names = _object_names(data)
    source, target = names.index(first), names.index(second)
    relation_id = list(data.vocab_relations).index(predicate)

    edge_index, edge_attr = data.edge_index, data.edge_attr.long()
    rows = [
        tuple(edge_attr[column].tolist())
        for column in range(edge_index.size(1))
        if int(edge_index[0, column]) == source
        and int(edge_index[1, column]) == target
        and int(edge_attr[column, 3]) == relation_id + 1
    ]
    labels = {row[EDGE_LABEL_SLICE] for row in rows}

    # One instance per (role, sign, goal_level, history_dt) combination, and
    # nothing else reaches this argument pair through this relation.
    expected = {
        (ROLE_FACT, 0, 0, 0),
        (ROLE_GOAL, 0, 0, 0),
        (ROLE_GOAL, 1, 0, 0),
        (ROLE_SUBGOAL, 0, 1, 0),
        (ROLE_SUBGOAL, 0, 2, 0),
        (ROLE_HISTORY, 0, 0, -1),
        (ROLE_HISTORY, 0, 0, -2),
    }
    got = {(role, sign, level, dt) for _rel, role, sign, level, dt, _cat in labels}
    assert got == expected
    assert len(labels) == len(expected)
    assert len(rows) == len(expected)


@pytest.mark.parametrize("facade_name", OBJECTS_ONLY_FACADES)
def test_objects_only_every_nullary_and_unary_instance_leaves_an_edge(
    facade_name: str, case: Case
) -> None:
    """§9.4: arity-0 and arity-1 literals leave a labelled trace.

    A pairwise projection has no pair to draw for them, so they used to
    disappear without a trace. An arity-1 literal now emits ``unary_self`` on
    its argument and an arity-0 literal ``nullary_self`` on the anchor.
    """
    data = FACADES[facade_name](case.problem).encode_pyg(case.state, **case.lanes())
    kinds = list(data.vocab_edge_kinds)
    unary_self, nullary_self = kinds.index("unary_self"), kinds.index("nullary_self")
    anchor = data.anchor_node_index
    assert anchor >= 0

    edges = {
        (int(data.edge_index[0, column]), int(data.edge_index[1, column]))
        + tuple(data.edge_attr[column].long().tolist())
        for column in range(data.edge_index.size(1))
    }

    sizes = data.tuple_sizes.long().view(-1)
    args = data.tuple_args.long().view(-1)
    ptr = data.tuple_ptr.long().view(-1)
    attrs = data.tuple_attr_ids.long()
    roles = data.tuple_role_ids.long()

    checked = 0
    for row in range(attrs.size(0)):
        arity = int(sizes[row])
        if arity > 1 or int(roles[row]) == ROLE_ACTION:
            # Actions are reified in every universe and keep their own
            # action_fwd/action_bwd edges rather than a self loop.
            continue
        label = (
            int(attrs[row, RELATION]),
            int(attrs[row, ROLE]),
            int(attrs[row, SIGN]),
            int(attrs[row, GOAL_LEVEL]),
            int(attrs[row, HISTORY_DT]),
            int(attrs[row, CATEGORY]),
        )
        if arity == 1:
            endpoint = int(args[int(ptr[row])])
            expected = (endpoint, endpoint, unary_self, 0, 0) + label
        else:
            expected = (anchor, anchor, nullary_self, 0, 0) + label
        assert expected in edges, f"instance row {row} left no edge: {expected}"
        checked += 1

    assert checked, "fixture must contain at least one arity-0/1 instance"


def test_objects_only_forces_the_tuple_instance_table(case: Case) -> None:
    """Pairwise edges cannot group an arity >= 3 literal, so the table is on.

    ``tri`` makes the point: ``(between a b c)`` and ``(between a b d)`` both
    contribute an ``a -> b`` pair edge with identical labels, and nothing in
    the topology says which of ``a -> c`` / ``a -> d`` belongs to which
    literal. The tuple table is the authoritative instance list, which is why
    ``objects_only`` always carries it.
    """
    for facade_name in OBJECTS_ONLY_FACADES:
        encoder = FACADES[facade_name](case.problem)
        assert encoder.config.include_tuple_tensors
        data = encoder.encode_pyg(case.state, **case.lanes())
        assert data.tuple_attr_ids is not None
        assert data.tuple_ptr.numel() - 1 == data.tuple_rel_ids.numel()

    if case.domain != "tri":
        return
    data = FACADES["object_clique"](case.problem).encode_pyg(case.state)
    sizes = data.tuple_sizes.long().view(-1)
    assert int(sizes.max()) >= 3, "tri must contribute an arity >= 3 instance"


# --------------------------------------------------------------------------
# §9.5 -- hyperedges are never empty
# --------------------------------------------------------------------------


def _nullary_goal_last_lanes(case: Case) -> dict[str, Any]:
    """Goals holding only the domain's nullary literal.

    Instances are emitted in lane order, so supplying nothing but a nullary
    goal puts the zero-arity instance *last*. That ordering is what made the
    old zero-member hyperedge visible: ``hyperedge_index[1].max() + 1``
    undercounted by exactly one while ``hyperedge_attr_ids`` did not.
    """
    predicate, args = case.spec["nullary"]
    return {"goals": [case.literal(predicate, args)]}


@pytest.mark.parametrize(
    "lane_builder",
    (Case.lanes, _nullary_goal_last_lanes),
    ids=("all-lanes", "nullary-goal-last"),
)
def test_hyperedge_count_matches_membership(lane_builder, case: Case) -> None:
    """§9.5: ``hyperedge_index[1].max() + 1 == hyperedge_attr_ids.size(0)``.

    An arity-0 instance used to produce a hyperedge with no members at all,
    which is invisible in the membership tensor. That is exactly the count
    ``HypergraphConv`` infers when ``num_edges`` is not passed, so the layer
    silently disagreed with the attribute table.
    """
    data = FACADES["hyper"](case.problem).encode_pyg(case.state, **lane_builder(case))

    rows = int(data.hyperedge_attr_ids.size(0))
    inferred = int(data.hyperedge_index[1].max()) + 1
    assert inferred == rows
    assert int(data.num_hyperedges.sum()) == rows
    assert data.hyperedge_attr_ids.size(1) == len(EXPECTED_CHANNEL_NAMES)

    # Every hyperedge really has a member, and every instance has a hyperedge.
    owners = data.hyperedge_index[1].long()
    assert set(owners.tolist()) == set(range(rows))
    # The zero-arity instance takes the anchor as its single member.
    assert data.anchor_node_index >= 0
    assert data.anchor_node_index in data.hyperedge_index[0].tolist()


@pytest.mark.parametrize(
    "lane_builder",
    (Case.lanes, _nullary_goal_last_lanes),
    ids=("all-lanes", "nullary-goal-last"),
)
def test_hypergraph_conv_runs_with_inferred_num_edges(lane_builder, case: Case) -> None:
    """§9.5: a stock attention hypergraph layer accepts the carrier as-is.

    ``num_edges`` is deliberately *not* passed, so the layer infers it from
    the membership tensor. A member-less hyperedge contributes nothing to the
    aggregation, so the layer does **not** raise when the inference
    undercounts -- it quietly trains an attention row that never reaches any
    node. That silence is why the count agreement is asserted here explicitly
    rather than being left to the forward pass to discover.
    """
    pyg_nn = pytest.importorskip("torch_geometric.nn")
    hypergraph_conv = getattr(pyg_nn, "HypergraphConv", None)
    if hypergraph_conv is None:  # pragma: no cover - old torch_geometric
        pytest.skip("torch_geometric lacks HypergraphConv")

    data = FACADES["hyper"](case.problem).encode_pyg(case.state, **lane_builder(case))
    rows = int(data.hyperedge_attr_ids.size(0))
    assert int(data.hyperedge_index[1].max()) + 1 == rows

    features = data.x_ids.float()
    torch.manual_seed(0)
    layer = hypergraph_conv(features.size(1), 8, use_attention=True, heads=1)
    attributes = data.hyperedge_attr_ids.float()
    out = layer(features, data.hyperedge_index, hyperedge_attr=attributes)
    assert out.shape == (data.num_nodes, 8)

    # Inferring the count and declaring it must be the same computation.
    explicit = layer(
        features, data.hyperedge_index, hyperedge_attr=attributes, num_edges=rows
    )
    assert torch.allclose(out, explicit)

    out.sum().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in layer.parameters()
    )


# --------------------------------------------------------------------------
# §9.6 -- graph view is a set, instance table is a multiset
# --------------------------------------------------------------------------


@pytest.mark.parametrize("facade_name", sorted(FACADES))
def test_duplicate_instances_keep_the_graph_but_grow_the_table(
    facade_name: str, case: Case
) -> None:
    """§9.6: a repeated literal adds an instance row, not extra edges.

    Node interning was idempotent but edge emission was not, so a duplicated
    goal literal re-emitted its argument edges (and, on the line view, its
    shortcut edges) against a single interned node.
    """
    goal = case.literal(*case.spec["goals"][0])
    encoder = FACADES[facade_name](case.problem)
    once = encoder.encode_pyg(case.state, goals=[goal])
    twice = encoder.encode_pyg(case.state, goals=[goal, goal])

    assert torch.equal(twice.x_ids, once.x_ids)
    assert torch.equal(twice.edge_index, once.edge_index)
    assert torch.equal(twice.edge_attr, once.edge_attr)
    assert list(twice.node_names) == list(once.node_names)

    if facade_name in TUPLE_TABLE_FACADES:
        assert twice.tuple_rel_ids.numel() == once.tuple_rel_ids.numel() + 1
        assert twice.tuple_attr_ids.size(0) == once.tuple_attr_ids.size(0) + 1
        assert torch.equal(twice.tuple_attr_ids[-1], once.tuple_attr_ids[-1])
        assert twice.tuple_ptr.numel() - 1 == twice.tuple_rel_ids.numel()
    if facade_name == "hyper":
        assert twice.hyperedge_attr_ids.size(0) == once.hyperedge_attr_ids.size(0) + 1
        assert torch.equal(twice.hyperedge_attr_ids[-1], once.hyperedge_attr_ids[-1])
        assert int(twice.num_hyperedges.sum()) == int(once.num_hyperedges.sum()) + 1


# --------------------------------------------------------------------------
# §9.7 -- batching
# --------------------------------------------------------------------------


def _assert_carriers_equal(left: Any, right: Any, *, label: str) -> None:
    left_fields = {k: v for k, v in left.to_dict().items() if not k.startswith("_")}
    right_fields = {k: v for k, v in right.to_dict().items() if not k.startswith("_")}
    assert set(left_fields) == set(right_fields), label

    for key in sorted(left_fields):
        left_value, right_value = left_fields[key], right_fields[key]
        if torch.is_tensor(left_value) or torch.is_tensor(right_value):
            assert torch.is_tensor(left_value) and torch.is_tensor(right_value), (
                f"{label}: {key}"
            )
            assert left_value.dtype == right_value.dtype, f"{label}: {key}"
            assert left_value.shape == right_value.shape, f"{label}: {key}"
            assert torch.equal(left_value, right_value), f"{label}: {key}"
        else:
            assert left_value == right_value, f"{label}: {key}"


@pytest.mark.parametrize("facade_name", SIX_FACADES)
def test_batching_agrees_across_native_manual_and_stream(
    facade_name: str, case: Case
) -> None:
    """§9.7: the three batching paths produce the same carrier.

    ``encode_batch_pyg`` (native collation), ``Batch.from_data_list`` (PyG's
    increment rules on ``DerivedGraphData``) and ``stream().flush_pyg()`` must
    agree on every field, including the widened instance tables and the
    per-graph ``anchor_index`` / ``history_dt_offset`` scalars.
    """
    encoder = FACADES[facade_name](case.problem)
    lanes = case.lanes()
    successors = case.state.generate_applicable_actions()
    second_state = successors[0].apply(case.state) if successors else case.state

    states = [case.state, second_state]
    native = encoder.encode_batch_pyg(
        states,
        goals=[lanes["goals"], lanes["goals"]],
        subgoal_layers=[lanes["subgoal_layers"], lanes["subgoal_layers"]],
        history_subgoals=[lanes["history_subgoals"], lanes["history_subgoals"]],
    )
    singles = [
        encoder.encode_pyg(
            state,
            goals=lanes["goals"],
            subgoal_layers=lanes["subgoal_layers"],
            history_subgoals=lanes["history_subgoals"],
        )
        for state in states
    ]
    manual = normalize_derived_graph_batch_metadata(Batch.from_data_list(singles))

    stream = encoder.stream()
    for state in states:
        stream.append(
            state,
            goals=lanes["goals"],
            subgoal_layers=lanes["subgoal_layers"],
            history_subgoals=lanes["history_subgoals"],
        )
    streamed = stream.flush_pyg()

    assert native.num_graphs == manual.num_graphs == streamed.num_graphs == 2
    _assert_carriers_equal(native, manual, label=f"{facade_name}: native vs manual")
    _assert_carriers_equal(native, streamed, label=f"{facade_name}: native vs stream")


@pytest.mark.parametrize("facade_name", TUPLE_TABLE_FACADES)
def test_tuple_csr_stays_aligned_on_every_batching_path(
    facade_name: str, case: Case
) -> None:
    """§9.7 / the CSR desync: ``tuple_ptr`` must index ``tuple_rel_ids``.

    A per-graph CSR of length ``T_i + 1`` cannot be concatenated into a global
    CSR: ``Batch.from_data_list`` produced ``T + B`` entries against ``T`` id
    rows, so ``padded_tuple_matrix()`` returned one row too many per graph
    junction and every row after the junction was shifted.
    """
    encoder = FACADES[facade_name](case.problem)
    lanes = case.lanes()
    successors = case.state.generate_applicable_actions()
    second_state = successors[0].apply(case.state) if successors else case.state
    states = [case.state, second_state]

    native = encoder.encode_batch_pyg(states, goals=[lanes["goals"], lanes["goals"]])
    singles = [encoder.encode_pyg(state, goals=lanes["goals"]) for state in states]
    manual = normalize_derived_graph_batch_metadata(Batch.from_data_list(singles))
    stream = encoder.stream()
    for state in states:
        stream.append(state, goals=lanes["goals"])
    streamed = stream.flush_pyg()

    for label, data in (
        ("single", singles[0]),
        ("native", native),
        ("manual", manual),
        ("stream", streamed),
    ):
        assert data.tuple_ptr.numel() - 1 == data.tuple_rel_ids.numel(), label
        assert int(data.tuple_ptr[0]) == 0, label
        assert int(data.tuple_ptr[-1]) == int(data.tuple_args.numel()), label
        assert bool((data.tuple_ptr.diff() >= 0).all()), label
        rows, mask = data.padded_tuple_matrix()
        assert rows.size(0) == data.tuple_rel_ids.numel(), label
        assert mask.shape == rows.shape, label


# --------------------------------------------------------------------------
# §9.8 -- line_share honours include_reverse_edges
# --------------------------------------------------------------------------


def test_line_share_honours_include_reverse_edges(case: Case) -> None:
    """§9.8: the shortcut edges follow the same directionality switch.

    ``line_share`` used to emit both directions unconditionally, so
    ``include_reverse_edges=False`` still produced a symmetric shortcut graph
    next to one-way argument edges.
    """
    lanes = case.lanes()
    both = AtomLineGraphEncoder(case.problem, include_reverse_edges=True).encode_pyg(
        case.state, **lanes
    )
    forward = AtomLineGraphEncoder(
        case.problem, include_reverse_edges=False
    ).encode_pyg(case.state, **lanes)

    line_share = list(both.vocab_edge_kinds).index("line_share")

    def shares(data):
        selector = data.edge_attr[:, EDGE_KIND].long() == line_share
        return data.edge_index[:, selector], data.edge_attr[selector]

    both_index, both_attr = shares(both)
    forward_index, forward_attr = shares(forward)

    assert both_index.size(1) > 0
    assert forward_index.size(1) * 2 == both_index.size(1)

    both_pairs = set(zip(both_index[0].tolist(), both_index[1].tolist()))
    forward_pairs = set(zip(forward_index[0].tolist(), forward_index[1].tolist()))
    assert all((dst, src) in both_pairs for src, dst in both_pairs)
    assert not any((dst, src) in forward_pairs for src, dst in forward_pairs)
    assert forward_pairs <= both_pairs

    # line_share is the one documented "no originating instance" edge: both
    # endpoints are reified fact nodes that already carry their own labels.
    assert bool((both_attr[:, EDGE_LABEL_SLICE] == 0).all())
    assert bool((forward_attr[:, EDGE_LABEL_SLICE] == 0).all())

    # Every other edge kind keeps its label, so the zeros are not a blanket.
    other = both.edge_attr[both.edge_attr[:, EDGE_KIND].long() != line_share]
    assert bool((other[:, 3] > 0).all())


# --------------------------------------------------------------------------
# §9.9 -- the documented embedding recipe with history supplied
# --------------------------------------------------------------------------


@pytest.mark.parametrize("facade_name", SIX_FACADES)
def test_documented_embedding_recipe_works_with_history(
    facade_name: str, case: Case
) -> None:
    """§9.9: the how-to's recipe must survive a negative history age.

    ``x_ids[:, 4]`` is the one signed channel, so the naive
    ``Embedding(int(ids.max()) + 2)`` recipe indexed an embedding table with
    negative ids. The documented fix is to shift by ``history_dt_offset``.
    """
    import torch.nn as tnn

    lanes = case.lanes()
    lanes["history_subgoals"] = [
        (-1, [case.literal(*case.spec["subgoal_layers"][0][0])]),
        (-2, [case.literal(*case.spec["goals"][0])]),
    ]
    data = FACADES[facade_name](case.problem).encode_pyg(case.state, **lanes)

    offset = int(data.history_dt_offset.reshape(-1)[0])
    assert offset > 0, "history lane must be populated"

    # The documented per-channel recipe, verbatim, on the node id table.
    torch.manual_seed(0)
    columns = []
    for channel in range(data.x_ids.size(1)):
        ids = data.x_ids[:, channel].long()
        if channel == HISTORY_DT:
            ids = ids + data.history_dt_offset  # shift the signed channel
        assert int(ids.min()) >= 0
        embedding = tnn.Embedding(int(ids.max().item()) + 2, 8)
        columns.append(embedding(ids))
    feats = torch.cat(columns, dim=-1)
    assert feats.shape == (data.num_nodes, 8 * data.x_ids.size(1))

    # ``objects_only`` reifies no history node, so its signed history age
    # rides in the edge labels and the instance table instead. Whichever lane
    # a facade carries it in, the same shift has to make it embeddable.
    signed_lanes: list[tuple[str, torch.Tensor]] = [
        ("x_ids[:, 4]", data.x_ids[:, HISTORY_DT].long()),
        ("edge_attr[:, 7]", data.edge_attr[:, 3 + HISTORY_DT].long()),
    ]
    if facade_name in TUPLE_TABLE_FACADES:
        signed_lanes.append(("tuple_dt_ids", data.tuple_dt_ids.long()))
    if facade_name == "hyper":
        signed_lanes.append(
            ("hyperedge_attr_ids[:, 4]", data.hyperedge_attr_ids[:, HISTORY_DT].long())
        )

    assert any(int(raw.min()) < 0 for _label, raw in signed_lanes), (
        "no lane carries the negative history age"
    )
    for label, raw in signed_lanes:
        shifted = raw + offset
        assert int(shifted.min()) >= 0, label
        # The shift is invertible: nothing is traded for embeddability.
        assert torch.equal(shifted - offset, raw), label
        tnn.Embedding(int(shifted.max().item()) + 2, 8)(shifted)
        if int(raw.min()) < 0:
            # Without the shift the very same recipe still raises, so the
            # offset is load-bearing rather than decorative.
            naive = tnn.Embedding(int(raw.max().item()) + 2, 8)
            with pytest.raises(IndexError):
                naive(raw)

    # Every other channel is a plain non-negative id.
    for channel in range(data.x_ids.size(1)):
        if channel == HISTORY_DT:
            continue
        assert int(data.x_ids[:, channel].min()) >= 0


def test_documented_embedding_recipe_works_on_a_batch(case: Case) -> None:
    """§9.9, batched: the shift is per graph via ``history_dt_offset[batch]``."""
    import torch.nn as tnn

    encoder = FACADES["star"](case.problem)
    successors = case.state.generate_applicable_actions()
    second_state = successors[0].apply(case.state) if successors else case.state
    history = [(-1, [case.literal(*case.spec["subgoal_layers"][0][0])])]
    batch = encoder.encode_batch_pyg(
        [case.state, second_state], history_subgoals=[history, []]
    )

    assert batch.history_dt_offset.numel() == 2
    ids = batch.x_ids[:, HISTORY_DT].long() + batch.history_dt_offset[batch.batch]
    assert int(ids.min()) >= 0
    torch.manual_seed(0)
    embedding = tnn.Embedding(int(ids.max().item()) + 2, 8)
    assert embedding(ids).shape == (batch.num_nodes, 8)


# --------------------------------------------------------------------------
# §9.10 -- node names stay injective across subgoal levels
# --------------------------------------------------------------------------


@pytest.mark.parametrize("facade_name", REIFIED_FACADES)
def test_node_names_are_injective_across_deep_subgoal_levels(
    facade_name: str, case: Case
) -> None:
    """§9.10: the same atom at levels 3, 4 and 5 gets three distinct names.

    The goal-level suffix table was clamped, so every level >= 3 rendered
    ``[sssg]`` and three genuinely different nodes carried one name.
    """
    literal = case.literal(*case.spec["goals"][0])
    data = FACADES[facade_name](case.problem).encode_pyg(
        case.state, subgoal_layers=[[literal] for _ in range(6)]
    )

    names = list(data.node_names)
    assert len(set(names)) == len(names)

    levels = data.x_ids[:, GOAL_LEVEL].long()
    roles = data.x_ids[:, ROLE].long()
    subgoal_names = [
        names[node]
        for node in range(len(names))
        if int(roles[node]) == ROLE_SUBGOAL and int(levels[node]) >= 3
    ]
    assert len(subgoal_names) >= 3
    assert len(set(subgoal_names)) == len(subgoal_names)
    assert sorted(
        int(levels[node])
        for node in range(len(names))
        if int(roles[node]) == ROLE_SUBGOAL
    ) == [1, 2, 3, 4, 5, 6]


# --------------------------------------------------------------------------
# §9.11 -- no all-zero placeholder node features
# --------------------------------------------------------------------------


@pytest.mark.parametrize("facade_name", SIX_FACADES)
def test_no_all_zero_placeholder_node_feature_lane(
    facade_name: str, case: Case
) -> None:
    """``x`` must not exist as an all-zero stand-in for ``x_ids``.

    The native node-feature lane is shared with families that put real
    features in ``x``. This family does not, so the lane arrived as an
    all-zero ``[N, 6]`` tensor: ``GCNConv(data.x, data.edge_index)`` returned
    identical rows for every node with no error anywhere. Silent total loss is
    exactly what this contract forbids, so the lane is dropped -- reaching for
    ``data.x`` must fail loudly instead.
    """
    encoder = FACADES[facade_name](case.problem)
    lanes = case.lanes()
    successors = case.state.generate_applicable_actions()
    second_state = successors[0].apply(case.state) if successors else case.state

    single = encoder.encode_pyg(case.state, **lanes)
    native = encoder.encode_batch_pyg(
        [case.state, second_state],
        goals=[lanes["goals"], lanes["goals"]],
    )
    manual = normalize_derived_graph_batch_metadata(
        Batch.from_data_list(
            [
                encoder.encode_pyg(state, goals=lanes["goals"])
                for state in (case.state, second_state)
            ]
        )
    )
    stream = encoder.stream()
    for state in (case.state, second_state):
        stream.append(state, goals=lanes["goals"])
    streamed = stream.flush_pyg()

    for label, data in (
        ("single", single),
        ("native batch", native),
        ("manual batch", manual),
        ("stream", streamed),
    ):
        stored = data.to_dict().get("x", None)
        assert stored is None, (
            f"{facade_name} ({label}) still ships an 'x' lane of shape "
            f"{getattr(stored, 'shape', None)}; if it is all-zero this is the "
            "silent-loss placeholder, and if it is not the contract changed"
        )
        assert getattr(data, "x", None) is None, f"{facade_name} ({label})"
        assert "x" not in set(map(str, data.keys())), f"{facade_name} ({label})"
        # num_nodes must not have depended on the dropped lane.
        assert data.num_nodes == data.x_ids.size(0), f"{facade_name} ({label})"


# --------------------------------------------------------------------------
# the exported vocabularies are part of the contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("facade_name", sorted(FACADES))
def test_exported_vocabularies_match_the_contract(facade_name: str, case: Case) -> None:
    """Ids are only meaningful against these lists, so pin them exactly."""
    data = FACADES[facade_name](case.problem).encode_pyg(case.state, **case.lanes())

    assert list(data.channel_names) == EXPECTED_CHANNEL_NAMES
    assert list(data.edge_channel_names) == EXPECTED_EDGE_CHANNEL_NAMES
    assert list(data.vocab_roles) == EXPECTED_ROLE_NAMES
    assert list(data.vocab_edge_kinds) == EXPECTED_EDGE_KIND_NAMES
    assert list(data.vocab_categories) == EXPECTED_CATEGORY_NAMES
    assert data.x_ids.size(1) == len(EXPECTED_CHANNEL_NAMES)
    assert data.edge_attr.size(1) == len(EXPECTED_EDGE_CHANNEL_NAMES)

    # The anchor is auxiliary: role 6, last node, never an object.
    if bool(data.has_anchor):
        anchor = data.anchor_node_index
        assert anchor == data.num_nodes - 1
        assert int(data.x_ids[anchor, ROLE]) == ROLE_ANCHOR
        assert bool((data.x_ids[anchor, 1:] == 0).all())
        assert list(data.node_names)[anchor] == ANCHOR_NODE_NAME
        assert ANCHOR_NODE_NAME not in list(data.object_names)
        assert anchor >= len(data.object_names)
    else:
        assert data.anchor_node_index == -1
        assert ROLE_ANCHOR not in set(data.x_ids[:, ROLE].long().tolist())

    if facade_name == "spd":
        # The anchor is not an object, so it never enters the SPD lanes.
        endpoints = set(data.spd_src.tolist()) | set(data.spd_dst.tolist())
        assert all(endpoint < len(data.object_names) for endpoint in endpoints)
