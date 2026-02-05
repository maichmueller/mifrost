import dataclasses
import os
import pathlib
from typing import Any, Dict, Sequence, Set, Tuple

import networkx as nx
import pymimir.wrapper_datasets
import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.data import BaseData
from torch_geometric.utils import to_networkx
from tests.conftest import load_problem
from mifrost.encoders import _parts_to_pyg, _split_goals
from mifrost import DEFAULT_LGAN_NN_EDGE_POS


def adv_state(state):
    return getattr(state, "_advanced_state", state)


def adv_domain(domain):
    return getattr(domain, "_advanced_domain", domain)


def adv_action(action):
    return getattr(action, "_advanced_ground_action", action)


def state_atoms(state, *, with_statics: bool = True):
    if hasattr(state, "get_atoms"):
        if with_statics:
            return list(state.get_atoms())
        return list(state.get_atoms(ignore_static=True))

    adv = adv_state(state)
    if hasattr(adv, "get_atoms"):
        if with_statics:
            return list(adv.get_atoms())
        return list(adv.get_atoms(ignore_static=True))
    if hasattr(adv, "get_fluent_atoms"):
        atoms = list(adv.get_fluent_atoms())
        if hasattr(adv, "get_derived_atoms"):
            atoms.extend(list(adv.get_derived_atoms()))
        return atoms
    return list(state.atoms(with_statics=with_statics))


def object_names(atom):
    return [obj.get_name() for obj in atom.get_terms()]


def predicate_name(atom) -> str:
    pred = predicate(atom)
    if not hasattr(pred, "get_name"):
        raise AttributeError("Predicate object does not expose get_name()")
    return pred.get_name()


def predicate(atom):
    if hasattr(atom, "get_predicate"):
        pred = atom.get_predicate()
        return getattr(pred, "_advanced_predicate", pred)
    adv = getattr(atom, "_advanced_ground_atom", atom)
    pred = adv.get_predicate()
    return getattr(pred, "_advanced_predicate", pred)


def predicate_arity(atom) -> int:
    pred = predicate(atom)
    if not hasattr(pred, "get_arity"):
        raise AttributeError("Predicate object does not expose get_arity()")
    return pred.get_arity()


def format_atom_with_suffix(atom, suffix: str = "") -> str:
    name = predicate_name(atom) + suffix
    objs = object_names(atom)
    if objs:
        return f"({name} {' '.join(objs)})"
    return f"({name})"


def format_literal_with_suffix(atom, polarity: bool, suffix: str = "") -> str:
    atom_str = format_atom_with_suffix(atom, suffix)
    prefix = "[+]" if polarity else "[-]"
    return f"{prefix}{atom_str}"


def hetero_data_equal(data: HeteroData, expected: HeteroData):
    assert isinstance(data, HeteroData) and isinstance(expected, HeteroData)
    assert set(data.node_types) == set(expected.node_types)
    assert set(data.edge_types) == set(expected.edge_types)
    for key in data.node_types:
        dx = data[key].x
        ex = expected[key].x
        if dx.numel() == ex.numel() == 0:
            continue
        if dx.shape != ex.shape:
            return False
        if not torch.allclose(dx, ex):
            return False
    for edge_type in data.edge_types:
        if not torch.equal(data[edge_type].edge_index, expected[edge_type].edge_index):
            return False
    return True


def keywise_equal(sample_normal, sample_streaming):
    assert sorted(sample_normal.keys()) == sorted(sample_streaming.keys())
    for key in sample_normal.keys():
        if isinstance(sample_normal[key], torch.Tensor) and isinstance(
            sample_streaming[key], torch.Tensor
        ):
            assert torch.equal(sample_normal[key], sample_streaming[key])
        else:
            if key == "targets" and any(
                isinstance(value, Sequence) and isinstance(value[0], BaseData)
                for value in (sample_streaming[key], sample_normal[key])
            ):
                v1s, v2s = sample_normal[key], sample_streaming[key]
                for v1, v2 in zip(v1s, v2s):
                    assert isinstance(v1, BaseData) and isinstance(v2, BaseData)
                    keywise_equal(v1, v2)
            else:
                try:
                    normal = sorted(sample_normal[key])
                    streaming = sorted(sample_streaming[key])
                except (ValueError, TypeError):
                    normal = sample_normal[key]
                    streaming = sample_streaming[key]
                assert normal == streaming, (
                    f"Mismatch in key '{key}': {normal} != {streaming}"
                )


def parts_to_pyg(parts: dict) -> HeteroData:
    return _parts_to_pyg(parts, as_batch=None)


def goal_inputs_from_problem(problem, *, goals=None, subgoal_layers=None):
    if goals is None:
        goals = list(problem.get_goal_condition().get_literals())
    inputs, _ = _split_goals(goals, subgoal_layers)
    return inputs


def to_named_networkx(
    data: HeteroData,
    *,
    drop_lgan: bool = False,
    lgan_rel: str = DEFAULT_LGAN_NN_EDGE_POS,
) -> nx.MultiDiGraph:
    graph = to_networkx(data, node_attrs=["node_names"], edge_attrs=[], to_multi=True)
    name_counts: Dict[str, int] = {}
    mapping: Dict[Any, str] = {}

    for node, attrs in graph.nodes(data=True):
        node_type = attrs.get("node_type")
        if node_type is None and isinstance(node, tuple) and len(node) == 2:
            node_type = node[0]
            attrs["node_type"] = node_type
        attrs.setdefault("type", node_type)
        name = attrs.get("node_names")
        if name is None:
            if isinstance(node, tuple) and len(node) == 2:
                name = f"{node_type}:{node[1]}"
            else:
                name = str(node)

        if name in name_counts:
            name_counts[name] += 1
            name = f"{name}#{name_counts[name]}"
        else:
            name_counts[name] = 0
        mapping[node] = name

    graph = nx.relabel_nodes(graph, mapping, copy=True)

    if graph.is_multigraph():
        edges = graph.edges(keys=True, data=True)
        for _, _, _, edge_attrs in edges:
            edge_type = edge_attrs.get("edge_type") or edge_attrs.get("type")
            if edge_type is None or len(edge_type) < 2:
                continue
            rel = edge_type[1]
            if isinstance(rel, str) and rel.isdigit():
                edge_attrs["position"] = int(rel)
    else:
        for _, _, edge_attrs in graph.edges(data=True):
            edge_type = edge_attrs.get("edge_type") or edge_attrs.get("type")
            if edge_type is None or len(edge_type) < 2:
                continue
            rel = edge_type[1]
            if isinstance(rel, str) and rel.isdigit():
                edge_attrs["position"] = int(rel)

    if drop_lgan:
        if graph.is_multigraph():
            to_remove = [
                (u, v, k)
                for u, v, k, edge_attrs in graph.edges(keys=True, data=True)
                if (
                    edge_attrs.get("edge_type")
                    or edge_attrs.get("type")
                    or (None, None)
                )[1]
                == lgan_rel
            ]
            graph.remove_edges_from(to_remove)
        else:
            to_remove = [
                (u, v)
                for u, v, edge_attrs in graph.edges(data=True)
                if (
                    edge_attrs.get("edge_type")
                    or edge_attrs.get("type")
                    or (None, None)
                )[1]
                == lgan_rel
            ]
            graph.remove_edges_from(to_remove)

    return graph
