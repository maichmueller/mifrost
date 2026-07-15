from __future__ import annotations

from collections import defaultdict
from typing import Any

import pytest

import mifrost
from mifrost.backends.flat import FlatSemanticAdapter

from tests.backends.test_semantic_parity import _backend_pair


def _flat_node_names(data: Any) -> list[str]:
    names = list(data.node_names)
    if names and isinstance(names[0], list):
        return [str(name) for graph_names in names for name in graph_names]
    return [str(name) for name in names]


def _named_edges(data: Any) -> tuple[tuple[str, str], ...]:
    names = _flat_node_names(data)
    return tuple(
        sorted(
            (names[int(source)], names[int(target)])
            for source, target in data.edge_index.t().tolist()
        )
    )


def _node_color_partition(data: Any) -> set[frozenset[str]]:
    groups: defaultdict[float, set[str]] = defaultdict(set)
    for name, color in zip(
        _flat_node_names(data), data.x.flatten().tolist(), strict=True
    ):
        groups[float(color)].add(str(name))
    return {frozenset(group) for group in groups.values()}


def _edge_color_partition(data: Any) -> set[frozenset[tuple[str, str]]]:
    names = _flat_node_names(data)
    groups: defaultdict[float, set[tuple[str, str]]] = defaultdict(set)
    for (source, target), color in zip(
        data.edge_index.t().tolist(), data.edge_attr.flatten().tolist(), strict=True
    ):
        groups[float(color)].add((names[int(source)], names[int(target)]))
    return {frozenset(group) for group in groups.values()}


@pytest.mark.parametrize("edge_features", [False, True])
@pytest.mark.parametrize("predicate_nodes", [False, True])
def test_semantic_color_matches_native_pymimir_structure(
    edge_features: bool, predicate_nodes: bool
) -> None:
    pymimir_reader, problem, _pytyr_reader, _successor_generator = _backend_pair()
    state = problem.get_initial_state()
    adapter = FlatSemanticAdapter(pymimir_reader)

    native_config = mifrost.ColorEncoderConfig(
        edge_features=edge_features,
        enable_global_predicate_nodes=predicate_nodes,
    )
    semantic_config = mifrost._neutral_core.SemanticColorEncoderConfig(
        edge_features=edge_features,
        enable_global_predicate_nodes=predicate_nodes,
    )
    native = mifrost.ColorEncoderEngine(
        problem.get_domain()._advanced_domain, native_config
    ).encode(state._advanced_state)
    semantic = mifrost._neutral_core.SemanticColorEncoderEngine(
        adapter.engine.predicates, semantic_config
    ).encode(adapter.make_input(state))

    native_data = native.as_pyg()
    semantic_data = semantic.as_pyg()
    assert set(native_data.node_names) == set(semantic_data.node_names)
    assert _named_edges(native_data) == _named_edges(semantic_data)
    assert native_data.to_dict().keys() == semantic_data.to_dict().keys()
    if edge_features:
        assert _edge_color_partition(native_data) == _edge_color_partition(
            semantic_data
        )
    else:
        assert _node_color_partition(native_data) == _node_color_partition(
            semantic_data
        )


def test_semantic_color_batches_and_composes_with_owned_builder() -> None:
    pymimir_reader, problem, _pytyr_reader, _successor_generator = _backend_pair()
    state = problem.get_initial_state()
    adapter = FlatSemanticAdapter(pymimir_reader)
    input_value = adapter.make_input(state)
    engine = mifrost._neutral_core.SemanticColorEncoderEngine(adapter.engine.predicates)

    batch = engine.encode_batch([input_value, input_value]).as_pyg()
    assert batch.num_graphs == 2
    assert batch.ptr.numel() == 3

    builder = mifrost.BatchBuilder()
    engine.encode(input_value, builder)
    builder.next_graph()
    engine.encode(input_value, builder)
    builder.next_graph()
    composed = builder.build().as_pyg()
    assert composed.num_graphs == 2
    assert _named_edges(composed) == _named_edges(batch)


def test_semantic_color_rejects_unsupported_lanes() -> None:
    pymimir_reader, problem, _pytyr_reader, _successor_generator = _backend_pair()
    state = problem.get_initial_state()
    adapter = FlatSemanticAdapter(pymimir_reader)
    engine = mifrost._neutral_core.SemanticColorEncoderEngine(adapter.engine.predicates)
    input_value = adapter.make_input(state)
    input_value.actions = [mifrost.SemanticGroundAction(0, [])]
    with pytest.raises(ValueError, match="does not support actions"):
        engine.encode(input_value)

    input_value.actions = []
    input_value.subgoal_layers = [[], [], [], []]
    with pytest.raises(ValueError, match="at most three subgoal layers"):
        engine.encode(input_value)
