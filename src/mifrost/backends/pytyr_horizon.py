"""PyTyr runtime for the public heterogeneous Horizon encoder."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Literal, cast

from pytyr.formalism.planning import GroundAction

from .pytyr import SemanticPlanningTaskAdapter
from .pytyr_flat import (
    _batch_param,
    _is_state,
    _lane_values,
    _sequence,
    _subgoal_values,
)
from .. import _neutral_core
from ..encoders._flat_validation import validate_subgoal_layers_state_payload
from ..encoders._rustworkx_dag import _is_rustworkx_digraph


def _mapping_or_attr(value: object, key: str) -> tuple[object, bool]:
    if isinstance(value, Mapping) and key in value:
        return value[key], True
    if hasattr(value, key):
        return getattr(value, key), True
    return None, False


def _semantic_atom_key(value: Any) -> tuple[int, tuple[int, ...]]:
    return int(value.predicate), tuple(int(argument) for argument in value.arguments)


def _semantic_input_key(value: Any) -> tuple[Any, ...]:
    return (
        tuple(value.objects),
        tuple(_semantic_atom_key(atom) for atom in value.state_facts),
    )


def _semantic_action_key(value: Any | None) -> tuple[int, tuple[int, ...]] | None:
    if value is None:
        return None
    return int(value.action), tuple(int(argument) for argument in value.arguments)


class _PyTyrHorizonStream:
    def __init__(self, runtime: "PyTyrHorizonRuntime") -> None:
        self._runtime = runtime
        self._reuse_removed = False
        self.reset()

    def append(self, root: object, dag: object = None, **kwargs: Any) -> int:
        item = self._runtime._dag(root, dag, **kwargs)
        if self._reuse_removed and self._removed:
            stream_id = min(self._removed)
            self._removed.remove(stream_id)
            self._items[stream_id] = item
            return stream_id
        stream_id = self._next_id
        self._next_id += 1
        self._items[stream_id] = item
        return stream_id

    def update(
        self,
        stream_id: int,
        root: object,
        dag: object = None,
        **kwargs: Any,
    ) -> None:
        if stream_id not in self._items:
            raise KeyError(stream_id)
        self._items[stream_id] = self._runtime._dag(root, dag, **kwargs)

    def remove(self, stream_id: int) -> None:
        if stream_id not in self._items:
            raise KeyError(stream_id)
        del self._items[stream_id]
        self._removed.add(stream_id)

    def set_reuse_removed(self, value: bool) -> None:
        self._reuse_removed = bool(value)

    def flush(self) -> Any:
        return self._runtime.engine.encode_batch(
            [self._items[key] for key in sorted(self._items)]
        )

    def reset(self) -> None:
        self._items: dict[int, Any] = {}
        self._removed: set[int] = set()
        self._next_id = 0


class PyTyrHorizonRuntime:
    """Convert PyTyr state DAGs into owned planner-neutral Horizon inputs."""

    backend_name: Literal["pytyr"] = "pytyr"

    def __init__(self, planning_task: object, config: Any) -> None:
        self._adapter = SemanticPlanningTaskAdapter(planning_task)
        self.engine = self._adapter.make_horizon_hgraph_engine(config)
        self._relation_dict = MappingProxyType(dict(self.engine.relation_arities))

    @property
    def relation_dict(self) -> Any:
        return self._relation_dict

    def _validate_root(self, root: object) -> None:
        if not _is_state(root):
            raise TypeError(
                "a PyTyr HorizonEncoder expects a lifted or ground PyTyr root "
                f"state, got {type(root)!r}"
            )

    def _literal(self, value: object) -> Any:
        predicate, arguments, positive = self._adapter._compact_literal(value)
        return _neutral_core.SemanticLiteral(
            _neutral_core.SemanticAtom(predicate, arguments), positive
        )

    def _state_input(
        self,
        state: object,
        *,
        goals: object = None,
        subgoal_layers: object = None,
    ) -> Any:
        if not _is_state(state):
            raise TypeError(
                "a PyTyr Horizon DAG can contain only PyTyr states, "
                f"got {type(state)!r}"
            )
        layers = (
            []
            if subgoal_layers is None
            else list(cast(Iterable[Iterable[object]], subgoal_layers))
        )
        validate_subgoal_layers_state_payload(
            layers,
            state_index=0,
            max_goal_level=int(self.engine.config.max_goal_level),
        )
        return self._adapter.make_input(
            state,
            goals=cast(Iterable[object] | None, goals),
            subgoal_layers=layers,
        )

    def _incoming_action(self, state: object, action: object) -> Any:
        if not isinstance(action, GroundAction):
            raise TypeError(
                "a PyTyr Horizon DAG edge must contain a PyTyr ground action "
                f"or None, got {type(action)!r}"
            )
        semantic_input = self._adapter.make_input(state, [action])
        return semantic_input.actions[0]

    def _semantic_dag(
        self,
        root: object,
        dag: Any,
        *,
        goals: object = None,
        subgoal_layers: object = None,
    ) -> Any:
        root_input = self._state_input(root, goals=goals, subgoal_layers=subgoal_layers)
        if _semantic_input_key(root_input) != _semantic_input_key(dag.root.state):
            raise ValueError("dag root must match root state")
        if goals is None and subgoal_layers is None:
            return dag

        explicit_goals = list(root_input.goals) if goals is not None else None
        explicit_subgoals = (
            list(root_input.subgoal_layers) if subgoal_layers is not None else None
        )
        nodes = []
        for node in dag.nodes:
            state_input = node.state
            if explicit_goals is not None:
                state_input.goals = explicit_goals
            if explicit_subgoals is not None:
                state_input.subgoal_layers = explicit_subgoals
            nodes.append(
                _neutral_core.SemanticTransitionNode(
                    state_input,
                    node.index,
                    node.depth,
                    node.incoming_action,
                    node.candidate_id,
                    node.delta_literals,
                    node.display_name,
                )
            )
        return _neutral_core.SemanticTransitionDAG(
            dag.predicates, dag.actions, nodes, dag.edges
        )

    def _rustworkx_dag(
        self,
        root: object,
        graph: Any,
        *,
        goals: object = None,
        subgoal_layers: object = None,
    ) -> Any:
        node_indices = [int(value) for value in graph.node_indices()]
        if not node_indices:
            raise ValueError("rustworkx PyDiGraph must not be empty")
        roots = [index for index in node_indices if int(graph.in_degree(index)) == 0]
        if len(roots) != 1:
            raise ValueError("rustworkx PyDiGraph must contain exactly one root node")
        graph_root = roots[0]

        states: dict[int, object] = {}
        candidate_ids: dict[int, int | None] = {}
        deltas: dict[int, list[Any] | None] = {}
        displays: dict[int, str | None] = {}
        for index in node_indices:
            data = graph.get_node_data(index)
            state_value, has_state = _mapping_or_attr(data, "state")
            state = state_value if has_state else data
            if not _is_state(state):
                raise TypeError(
                    f"rustworkx PyDiGraph node data at index {index} must be a "
                    f"PyTyr state input, got {type(state)!r}"
                )
            states[index] = state

            raw_candidate, has_candidate = _mapping_or_attr(data, "candidate_id")
            if not has_candidate or raw_candidate is None:
                candidate_ids[index] = None
            elif isinstance(raw_candidate, bool) or not isinstance(raw_candidate, int):
                raise TypeError(
                    "rustworkx PyDiGraph node data 'candidate_id' at index "
                    f"{index} must be an int or None"
                )
            else:
                candidate_ids[index] = int(raw_candidate)

            raw_delta, has_delta = _mapping_or_attr(data, "delta_literals")
            if has_delta and raw_delta is not None:
                if isinstance(raw_delta, (str, bytes, bytearray)) or not isinstance(
                    raw_delta, Iterable
                ):
                    raise TypeError(
                        "rustworkx PyDiGraph node data 'delta_literals' at index "
                        f"{index} must be an iterable or None"
                    )
                deltas[index] = [self._literal(value) for value in raw_delta]
            else:
                deltas[index] = None

            raw_display, has_display = _mapping_or_attr(data, "display_name")
            if has_display and raw_display is not None:
                if not isinstance(raw_display, str) or not raw_display:
                    raise TypeError(
                        "rustworkx PyDiGraph node data 'display_name' at index "
                        f"{index} must be a nonempty str or None"
                    )
                displays[index] = raw_display
            else:
                displays[index] = str(state)

        root_input = self._state_input(root, goals=goals, subgoal_layers=subgoal_layers)
        graph_root_input = self._state_input(
            states[graph_root], goals=goals, subgoal_layers=subgoal_layers
        )
        if _semantic_input_key(root_input) != _semantic_input_key(graph_root_input):
            raise ValueError("dag root must match root state")

        weighted_edges = [
            (int(source), int(target), action)
            for source, target, action in graph.weighted_edge_list()
        ]
        adjacency: dict[int, list[int]] = {index: [] for index in node_indices}
        incoming_edges: dict[int, list[tuple[int, object]]] = {
            index: [] for index in node_indices
        }
        seen_edges: set[tuple[int, int]] = set()
        for source, target, action in weighted_edges:
            if source not in adjacency or target not in adjacency:
                raise ValueError("rustworkx PyDiGraph edge endpoint is invalid")
            if source == target:
                raise ValueError("rustworkx PyDiGraph self edges are not supported")
            if (source, target) in seen_edges:
                # The historical Pymimir importer accepts multigraph input. The
                # neutral DAG stores topology as unique endpoint pairs, so retain
                # the additional edge only for incoming-action consistency.
                incoming_edges[target].append((source, action))
                continue
            seen_edges.add((source, target))
            adjacency[source].append(target)
            incoming_edges[target].append((source, action))

        depths = {graph_root: 0}
        queue: deque[int] = deque([graph_root])
        while queue:
            source = queue.popleft()
            for target in adjacency[source]:
                candidate_depth = depths[source] + 1
                if target not in depths or candidate_depth < depths[target]:
                    depths[target] = candidate_depth
                    queue.append(target)
        if len(depths) != len(node_indices):
            raise ValueError(
                "rustworkx PyDiGraph nodes must be reachable from its root"
            )

        ordered = [
            graph_root,
            *sorted(
                (index for index in node_indices if index != graph_root),
                key=lambda index: (depths[index], index),
            ),
        ]
        semantic_index = {native: index for index, native in enumerate(ordered)}
        semantic_nodes = []
        for native_index in ordered:
            incoming_action = None
            has_incoming_action = False
            for _parent, action in incoming_edges[native_index]:
                if action is None:
                    candidate = None
                else:
                    candidate = self._incoming_action(states[native_index], action)
                if not has_incoming_action:
                    incoming_action = candidate
                    has_incoming_action = True
                elif _semantic_action_key(incoming_action) != _semantic_action_key(
                    candidate
                ):
                    raise ValueError(
                        "all incoming rustworkx edges for one Horizon node must "
                        "carry the same action"
                    )
            semantic_nodes.append(
                _neutral_core.SemanticTransitionNode(
                    self._state_input(
                        states[native_index],
                        goals=goals,
                        subgoal_layers=subgoal_layers,
                    ),
                    semantic_index[native_index],
                    depths[native_index],
                    incoming_action,
                    candidate_ids[native_index],
                    deltas[native_index],
                    displays[native_index],
                )
            )
        semantic_edges = sorted(
            (semantic_index[source], semantic_index[target])
            for source, target in seen_edges
        )
        return _neutral_core.SemanticTransitionDAG(
            self.engine.predicates,
            self.engine.actions,
            semantic_nodes,
            semantic_edges,
        )

    def _dag(
        self,
        root: object,
        dag: object = None,
        *,
        goals: object = None,
        subgoal_layers: object = None,
    ) -> Any:
        self._validate_root(root)
        if isinstance(dag, _neutral_core.SemanticTransitionDAG):
            return self._semantic_dag(
                root, dag, goals=goals, subgoal_layers=subgoal_layers
            )
        if dag is None:
            root_input = self._state_input(
                root, goals=goals, subgoal_layers=subgoal_layers
            )
            return _neutral_core.SemanticTransitionDAG(
                self.engine.predicates,
                self.engine.actions,
                [
                    _neutral_core.SemanticTransitionNode(
                        root_input, 0, 0, display_name=str(root)
                    )
                ],
                [],
            )
        if _is_rustworkx_digraph(dag):
            return self._rustworkx_dag(
                root, dag, goals=goals, subgoal_layers=subgoal_layers
            )
        raise TypeError(
            "a PyTyr HorizonEncoder dag must be a rustworkx.PyDiGraph, "
            "SemanticTransitionDAG, or None; "
            f"got {type(dag)!r}"
        )

    def append_into_builder(
        self,
        root: object,
        builder: Any,
        *,
        dag: object = None,
        **kwargs: Any,
    ) -> None:
        self.engine.encode(self._dag(root, dag, **kwargs), builder)

    def encode_one(self, root: object, dag: object = None, **kwargs: Any) -> Any:
        return self.engine.encode(self._dag(root, dag, **kwargs))

    @staticmethod
    def _dag_values(value: object, *, root_count: int) -> list[Any]:
        kind, payload = _batch_param(value)
        if kind == "none" or payload is None:
            return [None] * root_count
        if (
            kind == "shared"
            or isinstance(payload, _neutral_core.SemanticTransitionDAG)
            or _is_rustworkx_digraph(payload)
        ):
            return [payload] * root_count
        values = _sequence(payload, field="dags")
        if len(values) != root_count:
            raise ValueError("dags length must match roots length")
        return values

    def encode_batch(
        self,
        roots: object,
        *,
        dags: object = None,
        goals: object = None,
        subgoal_layers: object = None,
    ) -> Any:
        root_values = [roots] if _is_state(roots) else _sequence(roots, field="roots")
        if not all(_is_state(root) for root in root_values):
            raise TypeError(
                "a PyTyr HorizonEncoder batch can contain only PyTyr root states"
            )
        count = len(root_values)

        def is_literal(value: object) -> bool:
            try:
                self._adapter._literal_key(value)
            except (TypeError, ValueError):
                return False
            return True

        goal_values = _lane_values(
            goals, state_count=count, field="goals", leaf=is_literal
        )
        subgoal_values = _subgoal_values(subgoal_layers, state_count=count)
        dag_values = self._dag_values(dags, root_count=count)
        semantic_dags = [
            self._dag(
                root,
                dag,
                goals=goal_values[index],
                subgoal_layers=subgoal_values[index],
            )
            for index, (root, dag) in enumerate(
                zip(root_values, dag_values, strict=True)
            )
        ]
        return self.engine.encode_batch(semantic_dags)

    def update_relations(self, relation_dict: Any) -> None:
        del relation_dict
        raise NotImplementedError(
            "update_relations is not implemented for the PyTyr Horizon backend"
        )

    def make_stream(self) -> Any:
        return _PyTyrHorizonStream(self)


__all__ = ["PyTyrHorizonRuntime"]
