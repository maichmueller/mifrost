import networkx as nx
import pytest

from mifrost import RelationFormatter


def _state_atoms(state):
    if hasattr(state, "get_atoms"):
        try:
            return list(
                state.get_atoms(
                    ignore_static=False, ignore_fluent=False, ignore_derived=False
                )
            )
        except TypeError:
            return list(state.get_atoms())
    adv = getattr(state, "_advanced_state", state)
    if hasattr(adv, "get_atoms"):
        return list(adv.get_atoms())
    return []


def _atom_objects(atom):
    if hasattr(atom, "get_objects"):
        objs = atom.get_objects()
    elif hasattr(atom, "get_terms"):
        objs = atom.get_terms()
    else:
        return []
    return [getattr(obj, "_advanced_object", obj) for obj in objs]


def _literal_atom(literal):
    if hasattr(literal, "get_atom"):
        return literal.get_atom()
    return getattr(literal, "atom", literal)


def _advanced_literal(literal):
    return getattr(literal, "_advanced_ground_literal", literal)


def _advanced_atom(atom):
    return getattr(atom, "_advanced_ground_atom", atom)


@pytest.mark.parametrize(
    "color_encoded_state",
    [
        [
            "blocks",
            "smedium",
            "initial",
            {"enable_global_predicate_nodes": False, "edge_features": False},
        ],
        [
            "blocks",
            "smedium",
            "initial",
            {"enable_global_predicate_nodes": True, "edge_features": False},
        ],
        [
            "blocks",
            "smedium",
            "initial",
            {"enable_global_predicate_nodes": False, "edge_features": True},
        ],
        [
            "blocks",
            "smedium",
            "initial",
            {"enable_global_predicate_nodes": True, "edge_features": True},
        ],
    ],
    indirect=True,
)
def test_color_encoding_initial(color_encoded_state):
    data, encoder, state = color_encoded_state
    graph = encoder.to_networkx(data)

    problem = state.get_problem()
    goals = list(problem.get_goal_condition().get_literals())
    facts = _state_atoms(state)

    # All objects referenced in facts/goals should be present as object nodes.
    object_nodes = set()
    for atom in facts:
        for obj in _atom_objects(atom):
            object_nodes.add(RelationFormatter.format_object(obj))
    for literal in goals:
        atom = _literal_atom(literal)
        for obj in _atom_objects(atom):
            object_nodes.add(RelationFormatter.format_object(obj))

    # All fact atoms should be present.
    for atom in facts:
        name = RelationFormatter.format_atom(_advanced_atom(atom))
        if encoder.edge_features:
            assert name in graph.nodes
        else:
            for pos, _obj in enumerate(_atom_objects(atom)):
                assert f"{name}:{pos}" in graph.nodes

    # All goal literals should be present with goal level 0.
    for literal in goals:
        base = RelationFormatter.format_literal(
            _advanced_literal(literal), 0, None, literal.get_polarity(), ""
        )
        if encoder.edge_features:
            assert base in graph.nodes
        else:
            for pos, _obj in enumerate(_atom_objects(_literal_atom(literal))):
                assert f"{base}:{pos}" in graph.nodes

    for obj_name in object_nodes:
        assert obj_name in graph.nodes

    if not encoder.edge_features:
        assert all("type" in attr for _, attr in graph.nodes.data())


@pytest.mark.parametrize(
    "color_encoded_state",
    [
        [
            "blocks",
            "smedium",
            "goal",
            {"enable_global_predicate_nodes": False, "edge_features": False},
        ]
    ],
    indirect=True,
)
def test_color_encoding_goal(color_encoded_state):
    data, encoder, state = color_encoded_state
    graph = encoder.to_networkx(data)
    problem = state.get_problem()
    goals = list(problem.get_goal_condition().get_literals())
    assert goals

    # All goal literals should have nodes.
    for literal in goals:
        base = RelationFormatter.format_literal(
            _advanced_literal(literal), 0, None, literal.get_polarity(), ""
        )
        if encoder.edge_features:
            assert base in graph.nodes
        else:
            for pos, _obj in enumerate(_atom_objects(_literal_atom(literal))):
                assert f"{base}:{pos}" in graph.nodes

    assert all("type" in attr for _, attr in graph.nodes.data())


@pytest.mark.parametrize(
    "color_encoded_state",
    [
        [
            "blocks",
            "smedium",
            "initial",
            {"enable_global_predicate_nodes": True, "edge_features": False},
        ],
        [
            "blocks",
            "smedium",
            "goal",
            {"enable_global_predicate_nodes": False, "edge_features": True},
        ],
    ],
    indirect=True,
)
def test_color_encoding_dict_to_pyg(color_encoded_state):
    data, encoder, _state = color_encoded_state
    graph = encoder.to_networkx(data)
    assert data.num_nodes == graph.number_of_nodes()
    # undirected graph, but pyg uses directed edges always,
    # -1 because a self-loop (`handempty`) is not doubled when made directed
    nr_self_loops = len(list(nx.selfloop_edges(graph)))
    assert data.num_edges == (2 * graph.number_of_edges() - nr_self_loops)
    assert data.edge_index is not None
    if encoder.edge_features:
        assert data.edge_attr is not None
        assert data.edge_attr.size(0) == data.edge_index.size(1)
        assert data.edge_attr.size(0) == data.num_edges
    else:
        assert data.x is not None
        assert data.edge_attr is None
        assert data.x.size(0) == data.num_nodes
        assert data.x.size(1) == 1  # feature is a single integer encoded as float


@pytest.mark.parametrize(
    "color_encoded_state",
    [
        [
            "blocks",
            "smedium",
            "initial",
            {"enable_global_predicate_nodes": False, "edge_features": False},
        ]
    ],
    indirect=True,
)
def test_color_encoder_draw_runs(color_encoded_state):
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        pytest.skip("matplotlib not available in test environment")

    matplotlib.use("Agg")

    data, encoder, _state = color_encoded_state
    ax = encoder.draw(data, with_labels=False, edge_labels=False)
    plt.close(ax.figure)
