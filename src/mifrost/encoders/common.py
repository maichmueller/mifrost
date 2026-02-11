from __future__ import annotations

import itertools
from typing import Any, Iterable, Mapping

import pymimir.advanced.formalism as af
import torch
from torch_geometric.data import Batch, HeteroData

from .types import (
    EncodingDict,
    DomainInput,
    GoalLiteralInput,
    HistorySubgoalInput,
    GroundActionInput,
    NativeEncodingInput,
    StateInput,
    to_advanced_action,
    to_advanced_domain,
    to_advanced_literal,
    to_advanced_state,
)

from .._core import GoalInputs


def _to_tensor(value: Any) -> torch.Tensor:
    """Normalize array-like values to torch tensors."""
    return torch.as_tensor(value)


# The C++ boundary is strict/typed; wrappers are unwrapped explicitly here.
def _advanced_domain(domain: DomainInput):
    """Return the advanced pymimir domain object when wrapped."""
    return to_advanced_domain(domain)


def _advanced_state(state: StateInput):
    """Return the advanced pymimir state object when wrapped."""
    return to_advanced_state(state)


def _advanced_literal(literal: GoalLiteralInput):
    """Return the advanced pymimir literal object when wrapped."""
    return to_advanced_literal(literal)


def _advanced_action(action: GroundActionInput):
    """Return the advanced pymimir action object when wrapped."""
    return to_advanced_action(action)


def _split_goals(
    goals: Iterable[GoalLiteralInput],
    subgoal_layers: Iterable[Iterable[GoalLiteralInput]] | None,
) -> GoalInputs:
    """
    Build ``GoalInputs`` from goals and optional layered subgoals.

    Returns ``(goal_inputs, layer_count)`` where ``layer_count`` includes the
    primary goal layer plus optional subgoal layers.
    """
    # Keep Python-side unwrapping/normalization (registered conversion logic) but
    # delegate type splitting and level bookkeeping to the C++ GoalInputs container.
    goals = list(goals)

    if subgoal_layers is None:
        return GoalInputs(map(_advanced_literal, goals), 0)

    inputs = GoalInputs([])
    for depth, layer in enumerate(itertools.chain([goals], subgoal_layers)):
        if not layer:
            continue
        inputs.extend(map(_advanced_literal, layer), int(depth))

    return inputs


def _prepare_actions(
    actions: Iterable[GroundActionInput] | None,
) -> list[af.GroundAction]:
    """Convert action wrappers to advanced pymimir ``GroundAction`` objects."""
    if actions is None:
        return []
    return [_advanced_action(action) for action in actions]


def _prepare_history_subgoals(
    history_subgoals: HistorySubgoalInput | None,
) -> list[
    tuple[
        int,
        list[af.StaticGroundLiteral | af.FluentGroundLiteral | af.DerivedGroundLiteral],
    ]
]:
    """Normalize history subgoals into advanced literal lists."""
    if history_subgoals is None:
        return []
    out: list[
        tuple[
            int,
            list[
                af.StaticGroundLiteral
                | af.FluentGroundLiteral
                | af.DerivedGroundLiteral
            ],
        ]
    ] = []
    for dt, literals in history_subgoals:
        adv_literals = [_advanced_literal(literal) for literal in literals]
        out.append((int(dt), adv_literals))
    return out


def _coerce_encoding_dict(encoding: NativeEncodingInput | Any) -> EncodingDict:
    if isinstance(encoding, Mapping):
        return encoding
    if hasattr(encoding, "as_dict"):
        coerced = encoding.as_dict()
        if isinstance(coerced, Mapping):
            return coerced
    raise TypeError(
        f"Expected encoding dictionary or BatchEncoding-like object, got {type(encoding)}"
    )


def _encoding_dict_to_pyg(
    encoding: NativeEncodingInput | Any,
    *,
    as_batch: bool | None = None,
    include_metadata: bool = True,
) -> HeteroData:
    """
    Convert normalized encoder dictionaries into PyG ``HeteroData``/``Batch``.

    Expected dictionary contract:
    - ``encoding["schema"]``: schema descriptor
    - ``encoding["tensors"]``: flat tensor payload keyed by schema keys
    - optional metadata: ``node_names``, ``object_names``, ``graph_attrs``
    """
    encoding_dict = _coerce_encoding_dict(encoding)
    raw_tensors: Mapping[str, Any] = encoding_dict.get("tensors", {})
    schema_obj = encoding_dict.get("schema")
    if schema_obj is None:
        raise ValueError(
            "Encoding schema missing; rebuild the extension to emit schema"
        )
    if hasattr(schema_obj, "to_dict"):
        schema_obj = schema_obj.to_dict()
    if not isinstance(schema_obj, Mapping):
        raise TypeError(f"Encoding schema must be a mapping, got {type(schema_obj)}")
    schema: Mapping[str, Any] = schema_obj
    node_names: Mapping[str, list[str]] = encoding_dict.get("node_names", {})
    node_feature_dims: Mapping[str, int] = encoding_dict.get("node_feature_dims", {})
    object_names_raw = encoding_dict.get("object_names", [])
    object_names: list[str] = (
        object_names_raw
        if isinstance(object_names_raw, list)
        else list(object_names_raw)
    )
    graph_attrs = encoding_dict.get("graph_attrs", {})
    num_graphs = int(encoding_dict.get("num_graphs", 0))

    if as_batch is None:
        as_batch = num_graphs > 1

    data: HeteroData
    if as_batch:
        data = Batch(_base_cls=HeteroData)
    else:
        data = HeteroData()

    edge_components: dict[tuple[str, str, str], dict[str, torch.Tensor]] = {}
    tensors: dict[str, Any] = {}
    tensors_torch: dict[str, torch.Tensor] = {}
    consumed_keys: set[str] = set()

    for key_obj, value in raw_tensors.items():
        key = key_obj if isinstance(key_obj, str) else str(key_obj)
        tensors[key] = value

    def get_tensor(key: str, value: Any | None = None) -> torch.Tensor:
        cached = tensors_torch.get(key)
        if cached is not None:
            return cached
        if value is None:
            value = tensors[key]
        tensor = _to_tensor(value)
        tensors_torch[key] = tensor
        return tensor

    graph_kind = schema.get("graph_kind")
    if graph_kind not in ("hetero", "homo"):
        raise ValueError(f"Unsupported graph_kind in schema: {graph_kind!r}")
    if "flags" not in schema:
        raise ValueError("Schema missing required 'flags' entry")
    if "extensions" not in schema:
        raise ValueError("Schema missing required 'extensions' entry")

    edge_type_list: list[tuple[str, str, str]] = []
    for entry in schema.get("edge_types", []):
        edge_type_list.append((entry["src"], entry["rel"], entry["dst"]))

    for entry in schema.get("node_tensors", []):
        key = entry["key"]
        if key not in tensors:
            raise KeyError(f"Schema references missing tensor key: {key}")
        node_type = entry["node_type"]
        attr = entry["attr"]
        data[node_type][attr] = get_tensor(key, tensors[key])
        consumed_keys.add(key)

    for entry in schema.get("edge_tensors", []):
        key = entry["key"]
        if key not in tensors:
            raise KeyError(f"Schema references missing tensor key: {key}")
        edge_type_idx = entry["edge_type"]
        if edge_type_idx >= len(edge_type_list):
            raise IndexError(f"Schema edge_type index {edge_type_idx} out of range")
        edge_type = edge_type_list[edge_type_idx]
        attr = entry["attr"]
        if attr == "edge_index":
            component = str(entry.get("part", ""))
            if component == "":
                raise ValueError(f"Missing edge_index component for key: {key}")
            edge_entry = edge_components.setdefault(edge_type, {})
            edge_entry[component] = get_tensor(key, tensors[key])
        else:
            data[edge_type][attr] = get_tensor(key, tensors[key])
        consumed_keys.add(key)

    for entry in schema.get("graph_tensors", []):
        key = entry["key"]
        if key not in tensors:
            raise KeyError(f"Schema references missing graph tensor key: {key}")
        attr = str(entry["attr"])
        setattr(data, attr, get_tensor(key, tensors[key]))
        consumed_keys.add(key)

        ptr_key = str(entry.get("ptr_key", ""))
        if ptr_key:
            if ptr_key not in tensors:
                raise KeyError(
                    f"Schema references missing graph tensor ptr key: {ptr_key}"
                )
            setattr(data, f"{attr}_ptr", get_tensor(ptr_key, tensors[ptr_key]))
            consumed_keys.add(ptr_key)

    for (src, rel, dst), component_map in edge_components.items():
        if "0" not in component_map or "1" not in component_map:
            raise ValueError(f"Incomplete edge_index components for {src}|{rel}|{dst}")
        edge_index = torch.stack((component_map["0"], component_map["1"]), dim=0)
        data[(src, rel, dst)].edge_index = edge_index

    node_names_lists: dict[str, list[str]] = {}
    if include_metadata:
        for node_type, names in node_names.items():
            store = data[node_type]
            names_list = names if isinstance(names, list) else list(names)
            node_names_lists[node_type] = names_list
            if as_batch:
                ptr_key = f"{node_type}/ptr"
                if ptr_key in tensors:
                    ptr_tensor = get_tensor(ptr_key).tolist()
                    store.node_names = [
                        names_list[ptr_tensor[i] : ptr_tensor[i + 1]]
                        for i in range(len(ptr_tensor) - 1)
                    ]
                else:
                    store.node_names = [names_list]
            else:
                store.node_names = names_list
            store.num_nodes = len(names_list)
        if object_names:
            if as_batch:
                symbol_type = None
                for node_type, names in node_names_lists.items():
                    if names == object_names:
                        symbol_type = node_type
                        break
                if symbol_type:
                    ptr_key = f"{symbol_type}/ptr"
                    if ptr_key in tensors:
                        ptr_tensor = get_tensor(ptr_key).tolist()
                        data.object_names = [
                            object_names[ptr_tensor[i] : ptr_tensor[i + 1]]
                            for i in range(len(ptr_tensor) - 1)
                        ]
                    else:
                        data.object_names = [object_names]
                else:
                    data.object_names = [object_names]
            else:
                data.object_names = object_names

    if include_metadata and graph_attrs:
        for key, value in graph_attrs.items():
            setattr(data, str(key), value)

    for node_type, dim in node_feature_dims.items():
        store = data[node_type]
        if "x" in store:
            continue
        ptr_key = f"{node_type}/ptr"
        if ptr_key in tensors:
            ptr_tensor = get_tensor(ptr_key)
            count = int(ptr_tensor[-1].item()) if ptr_tensor.numel() > 0 else 0
        elif include_metadata and node_type in node_names_lists:
            count = len(node_names_lists[node_type])
        else:
            count = 0
        store.x = torch.zeros((count, int(dim)), dtype=torch.float32)
        store.num_nodes = count

    if as_batch and num_graphs > 0:
        data._num_graphs = num_graphs

    if not as_batch:
        for node_type in data.node_types:
            store = data[node_type]
            if "ptr" in store:
                del store["ptr"]
            if "batch" in store:
                del store["batch"]

    return data


def encoding_to_tensors(
    encoding: Mapping[str, Any] | Any,
) -> Mapping[str, torch.Tensor]:
    """Return ``encoding['tensors']`` as a plain ``str -> torch.Tensor`` mapping."""
    encoding_dict = _coerce_encoding_dict(encoding)
    tensors: Mapping[str, Any] = encoding_dict.get("tensors", {})
    return {str(key): _to_tensor(value) for key, value in tensors.items()}
