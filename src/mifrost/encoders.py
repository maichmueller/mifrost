from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Sequence, Tuple

import torch
from torch_geometric.data import Batch, HeteroData

from ._core import BatchBuilder, GoalInputs, HGraphEncoderConfig, HGraphEncoderEngine


def _to_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value)


# The C++ boundary is strict/typed; wrappers are unwrapped explicitly here.
def _advanced_domain(domain: Any) -> Any:
    return getattr(domain, "_advanced_domain", domain)


def _advanced_state(state: Any) -> Any:
    return getattr(state, "_advanced_state", state)


def _advanced_literal(literal: Any) -> Any:
    return getattr(literal, "_advanced_ground_literal", literal)


def _advanced_action(action: Any) -> Any:
    return getattr(action, "_advanced_ground_action", action)


def _split_goals(
    goals: Iterable[Any],
    subgoal_layers: Iterable[Iterable[Any]] | None,
) -> Tuple[GoalInputs, int]:
    import pymimir.advanced.formalism as af

    goals = list(goals)
    if subgoal_layers is None:
        return GoalInputs([_advanced_literal(goal) for goal in goals], 0), 1

    # Split tagged literals into the universal GoalInputs container.
    inputs = GoalInputs([])
    static_goals: list[Any] = []
    fluent_goals: list[Any] = []
    derived_goals: list[Any] = []
    static_levels: dict[Any, int] = {}
    fluent_levels: dict[Any, int] = {}
    derived_levels: dict[Any, int] = {}

    layers = [goals]
    if subgoal_layers is not None:
        layers.extend(list(layer) for layer in subgoal_layers)

    for depth, layer in enumerate(layers):
        for literal in layer:
            adv = _advanced_literal(literal)
            if isinstance(adv, af.StaticGroundLiteral):
                static_goals.append(adv)
                static_levels[adv] = depth
            elif isinstance(adv, af.FluentGroundLiteral):
                fluent_goals.append(adv)
                fluent_levels[adv] = depth
            elif isinstance(adv, af.DerivedGroundLiteral):
                derived_goals.append(adv)
                derived_levels[adv] = depth
            else:
                raise TypeError(f"Unsupported goal literal type: {type(literal)}")

    inputs.static_goals = static_goals
    inputs.fluent_goals = fluent_goals
    inputs.derived_goals = derived_goals
    if hasattr(inputs, "static_goal_levels"):
        inputs.static_goal_levels = static_levels
        inputs.fluent_goal_levels = fluent_levels
        inputs.derived_goal_levels = derived_levels
    return inputs, len(layers)


def _prepare_actions(actions: Iterable[Any] | None) -> list[Any]:
    if actions is None:
        return []
    import pymimir.advanced.formalism as af

    out: list[Any] = []
    for action in actions:
        adv = _advanced_action(action)
        if not isinstance(adv, af.GroundAction):
            raise TypeError(f"Unsupported action type: {type(action)}")
        out.append(adv)
    return out


def _parts_to_pyg(
    parts: Mapping[str, Any],
    *,
    as_batch: bool | None = None,
    include_metadata: bool = True,
) -> HeteroData:
    # Assemble engine "parts" into PyG objects on the Python side only.
    raw_tensors: Mapping[str, Any] = parts.get("tensors", {})
    node_names: Mapping[str, list[str]] = parts.get("node_names", {})
    node_feature_dims: Mapping[str, int] = parts.get("node_feature_dims", {})
    object_names_raw = parts.get("object_names", [])
    object_names: list[str] = (
        object_names_raw
        if isinstance(object_names_raw, list)
        else list(object_names_raw)
    )
    num_graphs = int(parts.get("num_graphs", 0))

    if as_batch is None:
        as_batch = num_graphs > 1

    data: HeteroData
    if as_batch:
        data = Batch(_base_cls=HeteroData)
    else:
        data = HeteroData()

    edge_parts: MutableMapping[tuple[str, str, str], dict[str, torch.Tensor]] = {}
    tensors: dict[str, Any] = {}
    tensors_torch: dict[str, torch.Tensor] = {}

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

    for key, value in tensors.items():
        edge_pos = key.rfind("/edge_index_")
        if edge_pos != -1:
            base = key[:edge_pos]
            suffix = key[edge_pos + len("/edge_index_") :]
            src, rel, dst = base.split("|", 2)
            entry = edge_parts.setdefault((src, rel, dst), {})
            entry[suffix] = get_tensor(key, value)
            continue

        if "/" not in key:
            continue
        node_type, attr = key.split("/", 1)
        data[node_type][attr] = get_tensor(key, value)

    for (src, rel, dst), parts_map in edge_parts.items():
        if "0" not in parts_map or "1" not in parts_map:
            raise ValueError(f"Incomplete edge_index parts for {src}|{rel}|{dst}")
        edge_index = torch.stack((parts_map["0"], parts_map["1"]), dim=0)
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
                if symbol_type is not None:
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
    if as_batch and num_graphs > 0:
        data._num_graphs = num_graphs

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

    if not as_batch:
        for node_type in data.node_types:
            store = data[node_type]
            if "ptr" in store:
                del store["ptr"]
            if "batch" in store:
                del store["batch"]

    return data


def parts_to_tensors(parts: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    tensors: Mapping[str, Any] = parts.get("tensors", {})
    return {str(key): _to_tensor(value) for key, value in tensors.items()}


@dataclass
class HGraphEncoderStream:
    _engine: HGraphEncoderEngine

    def __post_init__(self) -> None:
        self._builder = BatchBuilder()

    def append(
        self,
        state: Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
    ) -> None:
        adv_state = _advanced_state(state)
        action_list = _prepare_actions(actions)
        if goals is None and subgoal_layers is None and not action_list:
            # Fast path: let the engine derive goals from the state/problem.
            self._engine.encode(adv_state, self._builder)
        else:
            if goals is None:
                if hasattr(state, "get_problem"):
                    goals = list(
                        state.get_problem().get_goal_condition().get_literals()
                    )
                else:
                    raise ValueError(
                        "goals must be provided when passing an advanced state"
                    )
            inputs, _ = _split_goals(goals, subgoal_layers)
            self._engine.encode(
                adv_state,
                inputs,
                action_list,
                self._builder,
            )
        if hasattr(self._builder, "next_graph"):
            self._builder.next_graph()

    def flush(
        self, *, as_batch: bool = True, include_metadata: bool = True
    ) -> HeteroData:
        parts = self.flush_parts()
        return _parts_to_pyg(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )

    def flush_parts(self) -> Mapping[str, Any]:
        parts = self._builder.build_parts()
        self._builder = BatchBuilder()
        return parts


class HGraphEncoder:
    def __init__(
        self,
        domain: Any,
        *,
        symbol_type_id: str = "_symbol_",
        ignore_actions: bool = True,
        add_nullary_predicates: bool = False,
        include_lgan_edges: bool = False,
        include_static: bool = True,
        include_empty_edge_types: bool = True,
        max_goal_level: int = 0,
        support_literals: bool = False,
        nullary_object_name: str = "![nullary_symbol]!",
        lgan_nn_edge_pos: str = "lgan_nn",
    ) -> None:
        config = HGraphEncoderConfig()
        config.symbol_type_id = symbol_type_id
        config.ignore_actions = ignore_actions
        config.add_nullary_predicates = add_nullary_predicates
        config.include_lgan_edges = include_lgan_edges
        config.include_static = include_static
        config.include_empty_edge_types = include_empty_edge_types
        config.max_goal_level = max_goal_level
        config.support_literals = support_literals
        config.nullary_object_name = nullary_object_name
        config.lgan_nn_edge_pos = lgan_nn_edge_pos
        self._engine = HGraphEncoderEngine(_advanced_domain(domain), config)

    @property
    def engine(self) -> HGraphEncoderEngine:
        return self._engine

    def encode_parts(
        self,
        state: Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
    ) -> Mapping[str, Any]:
        adv_state = _advanced_state(state)
        action_list = _prepare_actions(actions)
        if goals is None and subgoal_layers is None and not action_list:
            return self._engine.encode(adv_state)

        if goals is None:
            if hasattr(state, "get_problem"):
                goals = list(state.get_problem().get_goal_condition().get_literals())
            else:
                raise ValueError(
                    "goals must be provided when passing an advanced state"
                )
        inputs, _ = _split_goals(goals, subgoal_layers)
        # Explicitly pass goals/actions across the strict C++ boundary.
        return self._engine.encode(adv_state, inputs, action_list)

    def encode(
        self,
        state: Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
        include_metadata: bool = True,
    ) -> HeteroData:
        parts = self.encode_parts(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
        )
        return _parts_to_pyg(parts, as_batch=False, include_metadata=include_metadata)

    def _encode_batch_parts(
        self,
        states: Iterable[Any] | Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
    ) -> Mapping[str, Any]:
        is_state_like = hasattr(states, "get_problem") or hasattr(
            states, "_advanced_state"
        )
        if is_state_like:
            state_list = [states]
        else:
            if isinstance(states, (str, bytes)):
                raise TypeError("encode_batch expects a state or an iterable of states")
            state_list = list(states)

        shared_actions: list[Any] | None = None
        per_state_actions: list[Any] | None = None
        if actions is not None:
            try:
                shared_actions = _prepare_actions(actions)
            except TypeError:
                if isinstance(actions, Sequence):
                    if len(actions) != len(state_list):
                        raise ValueError(
                            "actions length must match states when providing per-state actions"
                        ) from None
                    per_state_actions = list(actions)
                else:
                    raise

        shared_inputs: GoalInputs | None = None
        if goals is not None:
            shared_inputs, _ = _split_goals(goals, subgoal_layers)

        builder = BatchBuilder()
        for idx, state in enumerate(state_list):
            adv_state = _advanced_state(state)
            if per_state_actions is not None:
                actions_for_state = per_state_actions[idx]
                action_list = (
                    _prepare_actions(actions_for_state)
                    if actions_for_state is not None
                    else []
                )
            else:
                action_list = shared_actions or []

            if goals is None and subgoal_layers is None and not action_list:
                # Fast path: let the engine derive goals from the state/problem.
                self._engine.encode(adv_state, builder)
            else:
                if goals is None:
                    if hasattr(state, "get_problem"):
                        goals_for_state = list(
                            state.get_problem().get_goal_condition().get_literals()
                        )
                        inputs, _ = _split_goals(goals_for_state, subgoal_layers)
                    else:
                        raise ValueError(
                            "goals must be provided when passing an advanced state"
                        )
                else:
                    inputs = shared_inputs
                self._engine.encode(adv_state, inputs, action_list, builder)
            if hasattr(builder, "next_graph"):
                builder.next_graph()

        return builder.build_parts()

    def encode_batch(
        self,
        states: Iterable[Any] | Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
        include_metadata: bool = True,
    ) -> HeteroData:
        """
        Encode multiple states into a single PyG Batch using the C++ BatchBuilder.

        If a single state object is provided, it is treated as a batch of size 1.
        """
        parts = self._encode_batch_parts(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
        )
        return _parts_to_pyg(parts, as_batch=True, include_metadata=include_metadata)

    def encode_batch_parts(
        self,
        states: Iterable[Any] | Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
    ) -> Mapping[str, Any]:
        """
        Encode multiple states and return engine parts without PyG assembly.
        """
        return self._encode_batch_parts(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
        )

    def stream(self) -> HGraphEncoderStream:
        return HGraphEncoderStream(self._engine)
