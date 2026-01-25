from __future__ import annotations

import itertools
from collections import defaultdict, deque
from itertools import chain
from typing import Dict, Iterable, Iterator, List, Literal, Sequence

import networkx as nx
import torch
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from plangolin.utils.misc import forward_kwargs, tolist
from xmimir import (
    XAction,
    XAtom,
    XDomain,
    XLiteral,
    XObject,
    XProblem,
    XState,
    XSuccessorGenerator,
    XTransition,
)

from .base_encoder import check_encoded_by_this
from .hetero_encoder import AdHocRelation, HGraphEncoder, PredicateEdgeType
from .pyg_batch_builder import HorizonBatchBuilder
from .pyg_builder import PygBuilderBase
from .relation_dict import RelationDict
from .transition_hetero_encoder import (
    _compute_change_literals,
    _compute_change_satisfaction,
)

TargetT = XState | str


class TransitionDAG:
    """
    Thin DAG wrapper around a networkx.DiGraph.
    Nodes are targets (XState or synthetic labels in actions-only mode) with attrs:
      - dag_index: insertion order index (stable iteration)
      - depth: shortest path length from root (computed)
      - action: final action leading to this target (or None if not provided)
    Edges encode parent→child relations.
    """

    def __init__(
        self,
        root: XState,
        transitions: Sequence[Sequence[XState]]
        | Sequence[XTransition | Sequence[XTransition]]
        | Sequence[XAction | Sequence[XAction]]
        | nx.DiGraph
        | nx.MultiDiGraph
        | None,
        ignore_actions: bool,
        actions_only: bool = False,
    ) -> None:
        self.G = nx.DiGraph()
        self.root: TargetT = root
        self.ignore_actions = ignore_actions
        self.actions_only = actions_only
        if self.actions_only and self.ignore_actions:
            raise ValueError(
                "TransitionDAG with actions_only=True requires ignore_actions=False."
            )
        self._next_idx = itertools.count()
        # bidirectional index mapping for O(1) lookups by index or node
        self._idx_to_target: Dict[int, TargetT] = {}
        # action-step node label cache for actions-only mode
        self._action_prefix_nodes: Dict[
            tuple[str | tuple[str, ...] | None, ...], TargetT
        ] = {}
        self._action_label_counter: itertools.count | None = (
            itertools.count(1) if actions_only else None
        )
        self._add_node(root, action=None)
        self.root_index: int = self.index(self.root)
        self._process_transitions(transitions)
        self._finalize_depths()

    def _add_node(self, target: TargetT, action: XAction | None = None) -> None:
        if target in self.G:
            # set action only once if not set
            if action is not None and self.G.nodes[target].get("action") is None:
                self.G.nodes[target]["action"] = action
            return
        idx = next(self._next_idx)
        self.G.add_node(target, dag_index=idx, depth=None, action=action)
        self._idx_to_target[idx] = target

    def __contains__(self, target: TargetT) -> bool:
        return target in self.G

    def __iter__(self) -> Iterator[tuple[TargetT, int]]:
        return iter(sorted(self.G.nodes(data="dag_index"), key=lambda pair: pair[1]))

    def successor_iter(self) -> Iterator[tuple[TargetT, XAction | None, int]]:
        for target, idx in sorted(self.G.nodes(data="dag_index"), key=lambda p: p[1]):
            if target == self.root:
                continue
            yield target, self.action(target), idx

    @property
    def transitions(self) -> set[tuple[TargetT, TargetT]]:
        return set(self.G.edges())

    def register_transition(
        self,
        parent: TargetT,
        child: TargetT,
        action: XAction | Sequence[XAction] | None = None,
    ) -> tuple[int, int]:
        # ensure nodes exist
        self._add_node(parent)
        # final action (store only XAction)
        act = None
        if not self.ignore_actions and action is not None:
            act = (
                action[-1]
                if isinstance(action, Sequence) and not isinstance(action, (str, bytes))
                else action
            )
            if not isinstance(act, XAction):
                act = None
        self._add_node(child, action=act)
        self.G.add_edge(parent, child)
        return self.index(parent), self.index(child)

    def index(self, target: TargetT) -> int:
        return self.G.nodes[target]["dag_index"]

    def depth(self, key: int | TargetT) -> int:
        node = key if not isinstance(key, int) else self._idx_to_target[key]
        return int(self.G.nodes[node]["depth"])

    def action(self, target: TargetT | int) -> XAction | None:
        node = target if not isinstance(target, int) else self._idx_to_target[target]
        return self.G.nodes[node].get("action")

    def _finalize_depths(self) -> None:
        # compute minimal depths from root
        depths = nx.single_source_shortest_path_length(self.G, self.root)
        nx.set_node_attributes(self.G, depths, "depth")

    def _process_transitions(
        self,
        transitions: Sequence[Sequence[XState]]
        | Sequence[XTransition | Sequence[XTransition]]
        | Sequence[XAction | Sequence[XAction]]
        | nx.DiGraph
        | nx.MultiDiGraph
        | None,
    ):
        if not transitions:
            return
        if self.actions_only:
            self._process_action_sequences(transitions)
            return
        if isinstance(transitions, (nx.DiGraph, nx.MultiDiGraph)):
            if self.root not in transitions:
                raise ValueError(
                    "The provided lookahead transition graph must contain the root state."
                )
            self._register_graph_transitions(transitions)
            return

        assert isinstance(transitions, Sequence), (
            "transitions must be a sequence if not Graph."
        )

        for path in transitions:
            if not path:
                continue
            if isinstance(path, XTransition):
                expanded_path = self._expand_transition(path)
            else:
                assert isinstance(path, Sequence)
                expanded_path = self._expand_path(path)
            parent = self.root
            for successor_state, action in expanded_path:
                self.register_transition(parent, successor_state, action=action)
                parent = successor_state

    def _process_action_sequences(
        self,
        transitions: Sequence[
            XAction
            | XTransition
            | Sequence[XAction]
            | Sequence[Sequence[XAction]]
            | Sequence[XTransition]
        ],
    ) -> None:
        for path in transitions:
            if not path:
                continue
            parent = self.root
            prefix: tuple[str | tuple[str, ...] | None, ...] = ()
            for action in self._iter_actions(path):
                if action is None:
                    continue
                prefix = (*prefix, self._action_signature(action))
                child = self._ensure_action_node(prefix)
                self.register_transition(parent, child, action=action)
                parent = child

    def _iter_actions(
        self,
        candidate: XAction | XTransition | Sequence[XAction] | Sequence,
    ) -> Iterator[XAction]:
        if isinstance(candidate, XAction):
            yield candidate
            return
        if isinstance(candidate, XTransition):
            action = candidate.action
            if action is None:
                return
            if isinstance(action, Sequence) and not isinstance(action, (str, bytes)):
                for act in action:
                    yield from self._iter_actions(act)
            else:
                yield action
            return
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            for element in candidate:
                yield from self._iter_actions(element)
            return
        raise TypeError(
            "Unsupported element in actions-only transition sequence: "
            f"{type(candidate)!r}"
        )

    def _action_signature(
        self, action: XAction | Sequence[XAction] | None
    ) -> str | tuple[str, ...] | None:
        if action is None:
            return None
        if isinstance(action, Sequence) and not isinstance(action, (str, bytes)):
            signatures = [
                self._action_signature(act) for act in action if act is not None
            ]
            return tuple(sig for sig in signatures if sig is not None)
        return getattr(action, "name", str(action))

    def _ensure_action_node(
        self, signature: tuple[str | tuple[str, ...] | None, ...]
    ) -> str:
        node = self._action_prefix_nodes.get(signature)
        if node is None:
            if self._action_label_counter is None:
                raise RuntimeError(
                    "Action label counter not initialised for actions-only DAG"
                )
            target_id = next(self._action_label_counter)
            node = f"target:{target_id}"
            self._action_prefix_nodes[signature] = node
        return node

    def _register_graph_transitions(
        self,
        graph: nx.DiGraph | nx.MultiDiGraph,
    ) -> None:
        queue: deque[TargetT] = deque([self.root])
        seen: set[TargetT] = {self.root}

        def _edge_action(data: dict | None):
            if not data:
                return None
            if "action" in data:
                return data["action"]
            if "transition" in data and isinstance(data["transition"], XTransition):
                return data["transition"].action
            if "actions" in data:
                return data["actions"]
            return None

        while queue:
            parent = queue.popleft()
            if self.ignore_actions:
                child_iter = ((child, None) for child in graph.successors(parent))
            else:
                if isinstance(graph, nx.MultiDiGraph):
                    edges = graph.out_edges(parent, data=True, keys=True)
                    child_iter = (
                        (child, _edge_action(data)) for _, child, _, data in edges
                    )
                else:
                    edges = graph.out_edges(parent, data=True)
                    child_iter = (
                        (child, _edge_action(data)) for _, child, data in edges
                    )
            for child, action in child_iter:
                self.register_transition(parent, child, action=action)
                if child not in seen:
                    seen.add(child)
                    queue.append(child)

    def _expand_transition(
        self,
        transition: XTransition,
    ) -> list[tuple[XState, XAction | Sequence[XAction] | None]]:
        action = transition.action
        if isinstance(action, Sequence):
            successor_gen = XSuccessorGenerator(transition.source.problem)
            target = transition.source
            steps: list[tuple[XState, XAction | Sequence[XAction] | None]] = []
            for act in action:
                target, _ = successor_gen.successor(target, act)
                steps.append((target, None if self.ignore_actions else act))
            return steps
        target_state = transition.target
        stored_action: XAction | Sequence[XAction] | None
        if self.ignore_actions:
            stored_action = None
        else:
            stored_action = action
        return [(target_state, stored_action)]

    def _expand_path(
        self,
        path: Sequence[XState] | Sequence[XTransition],
    ) -> list[tuple[XState, XAction | Sequence[XAction] | None]]:
        expanded: list[tuple[XState, XAction | Sequence[XAction] | None]] = []
        for item in path:
            if isinstance(item, XTransition):
                expanded.extend(self._expand_transition(item))
            else:
                expanded.append((item, None))
        return expanded


class HorizonHGraphEncoder(HGraphEncoder):
    """
    Encode a current state together with a tree/DAG of lookahead targets (e.g., generated
    by IW search) into a heterogeneous graph. Each target receives a dedicated **target
    symbol** node that connects to all relations (atoms/literals) that hold in that
    target, while temporal `primitive_transition` relation nodes connect parent-child
    target symbols, preserving the topology.

    Two successor encodings are supported:

    * ``"full"``  – successor target symbols connect to every atom that holds in the
      successor (excluding static facts).
    * ``"delta"`` – successor target symbols connect only to literals of added
      or deleted atoms with respect to the root target.

    Parameters
    ----------
    successor_mode : {"full", "delta", "action"}, default "full"
        Controls how successor targets are represented.
    parent_relation : str, default "parent"
        Relation node type used to connect parent-child target symbols.
    exclude_root_candidate : bool, default True
        If True, the root target (DAG index 0) is not marked as a candidate.
    """

    SUPPORTED_SUCCESSOR_MODES = {"full", "delta", "action"}

    def __init__(
        self,
        domain: XDomain,
        *,
        relation_dict: RelationDict | None = None,
        successor_mode: str
        | Literal["full"]
        | Literal["delta"]
        | Literal["action"] = "full",
        target_symbol_node_prefix: str = "target:",
        parent_relation: str = "parent",
        sibling_relation: str = "sibling",
        cousin_relation: str = "cousin",
        enable_parent_relation: bool = False,
        enable_sibling_relation: bool = False,
        enable_cousin_relation: bool = False,
        exclude_root_candidate: bool = True,
        add_nullary_predicates: bool = False,
        include_lgan_edges: bool = False,
        lgan_nn_edge_pos: str = "lgan_nn",
        ignore_actions: bool = True,
        **kwargs,
    ) -> None:
        mode = successor_mode.lower()
        if mode not in self.SUPPORTED_SUCCESSOR_MODES:
            raise ValueError(
                f"Unsupported successor_mode '{successor_mode}'. "
                f"Expected one of {sorted(self.SUPPORTED_SUCCESSOR_MODES)}."
            )
        # slice kwargs down to what RelationDict needs in __init__
        relation_dict_kwargs = forward_kwargs(RelationDict.__init__, kwargs)
        init_kwargs = {k: v for k, v in kwargs.items() if k not in relation_dict_kwargs}
        if mode == "delta":
            if relation_dict is not None:
                if not relation_dict.support_literals:
                    raise ValueError(
                        f"{self.__class__.__name__}(delta) requires a RelationDict "
                        "with support_literals=True."
                    )
            else:
                relation_dict_kwargs["support_literals"] = True
        self.transition_mode = mode
        self._encode_effects = mode != "action"
        self.target_symbol_node_prefix = target_symbol_node_prefix
        self.lgan_nn_edge_pos = lgan_nn_edge_pos
        self.parent_relation = parent_relation
        self.sibling_relation = sibling_relation
        self.cousin_relation = cousin_relation
        self.enable_parent_relation = enable_parent_relation
        self.enable_sibling_relation = enable_sibling_relation
        self.enable_cousin_relation = enable_cousin_relation
        self.exclude_root_candidate = bool(exclude_root_candidate)
        relation_dict = relation_dict or RelationDict(
            tuple(AdHocRelation(p.name, p.arity + 1) for p in domain.predicates()),
            tuple(AdHocRelation(p.name, p.arity + 1) for p in domain.actions)
            if not ignore_actions
            else (),
            **relation_dict_kwargs,
        )
        super().__init__(
            domain,
            relation_dict=relation_dict,
            ignore_actions=ignore_actions,
            add_nullary_predicates=add_nullary_predicates,
            include_lgan_edges=include_lgan_edges,
            lgan_nn_edge_pos=lgan_nn_edge_pos,
            **init_kwargs,
        )
        # Always include parent relation; conditionally include family relations
        for rel, enabled in (
            (self.parent_relation, self.enable_parent_relation),
            (self.sibling_relation, self.enable_sibling_relation),
            (self.cousin_relation, self.enable_cousin_relation),
        ):
            if not enabled:
                continue
            self.relation_dict.update({rel: 2})
            for pos in range(2):
                forward = PredicateEdgeType(self.symbol_type_id, str(pos), rel)
                reverse = PredicateEdgeType(rel, str(pos), self.symbol_type_id)
                self.all_edge_types.append(forward)
                self.all_edge_types.append(reverse)
        self._ensure_config_hash()

    def _config_dict(self) -> dict:
        base = super()._config_dict()
        base.update(
            {
                "transition_mode": self.transition_mode,
                "target_symbol_prefix": self.target_symbol_node_prefix,
                "parent_relation": self.parent_relation,
                "enable_sibling_relations": self.enable_sibling_relation,
                "enable_cousin_relations": self.enable_cousin_relation,
                "exclude_root_candidate": self.exclude_root_candidate,
            }
        )
        return base

    def encode(
        self,
        state: XState | Iterable[XAtom],
        goals: Iterable[XLiteral] | None = None,
        actions: Iterable[XAction] | None = None,
        *,
        transitions: Sequence[XTransition]
        | Sequence[Sequence[XState]]
        | nx.Graph
        | None = None,
        problem: XProblem | None = None,
        **kwargs,
    ) -> HeteroData:
        if not isinstance(state, XState):
            raise TypeError(
                f"{self.__class__.__name__} expects the current state to be an XState."
            )
        assert transitions is not None, "Transitions are required."
        dag = TransitionDAG(
            state,
            transitions,
            ignore_actions=self.ignore_actions and self.transition_mode != "action",
            actions_only=self.transition_mode == "action",
        )
        return super().encode(state, goals, actions, dag=dag, **kwargs)

    def _encode(
        self,
        builder: PygBuilderBase,
        facts: Sequence[XAtom],
        goals: Sequence[XLiteral],
        actions: Sequence[XAction],
        objects: Sequence[XObject] | None = None,
        dag: TransitionDAG | None = None,
        **kwargs,
    ) -> None:
        assert dag is not None, "TransitionDAG is required for HorizonHGraphEncoder."
        target_nodes: Dict[int, str] = {}
        target_positions: list[int] = []
        for target_obj, idx in dag:
            target_node = self._target_node_label(idx)
            target_nodes[idx] = target_node
            target_index = (
                getattr(target_obj, "index", None)
                if not isinstance(target_obj, str)
                else None
            )
            target_pos = builder.add_node(
                target_node,
                self.symbol_type_id,
                name=str(target_obj),
                depth=dag.depth(idx),
                target_index=target_index,
            )
            target_positions.append(target_pos)

        builder.set_graph_attr(
            "target_positions", torch.tensor(target_positions, dtype=torch.long)
        )
        objects = objects or self._contained_objects(
            chain(facts, goals, actions if not self.ignore_actions else ())
        )
        # encode the root state first
        self._encode_objects(objects, builder)
        root_node = target_nodes[0]
        to_prefix = lambda node: f"{node}|"
        root_prefix = to_prefix(root_node)
        self._encode_facts(
            facts,
            builder,
            node_prefix=root_prefix,
            extra_objects=[root_node],
        )
        # goal literals have no reason to be associated with any target node!
        self._encode_literals(
            goals,
            builder,
            node_prefix=root_prefix,
            extra_objects=[root_node],
        )
        if self.relation_dict.goal_satisfaction_derivations and goals:
            self._encode_goal_satisfaction(
                self._goal_satisfaction_map(facts, goals),
                builder,
                node_prefix=root_prefix,
                extra_objects=[root_node],
            )
        # now encode all the transitions in the dag
        root_facts = self._collect_target_atoms(dag.root)
        match self.transition_mode:
            case "full":
                for successor_state, succ_action, idx in dag.successor_iter():
                    successor_atoms = self._collect_target_atoms(successor_state)
                    successor_node = target_nodes[idx]
                    self._encode_facts(
                        successor_atoms,
                        builder,
                        node_prefix=to_prefix(successor_node),
                        extra_objects=[successor_node],
                    )
                    if succ_action is not None:
                        self._encode_action(
                            succ_action, builder, extra_objects=[successor_node]
                        )
                    if self.relation_dict.goal_satisfaction_derivations and goals:
                        sat_goals = self._goal_satisfaction_map(successor_atoms, goals)
                        self._encode_goal_satisfaction(
                            sat_goals,
                            builder,
                            node_prefix=to_prefix(successor_node),
                            extra_objects=[successor_node],
                        )
            case "delta":
                problem = dag.root.problem
                assert problem is not None, (
                    f"{problem!r} cannot be `None` in `delta` mode."
                )
                for successor_state, succ_action, idx in dag.successor_iter():
                    successor_facts = self._collect_target_atoms(successor_state)
                    literals, added_facts, removed_facts = _compute_change_literals(
                        root_facts, successor_facts, problem
                    )
                    successor_node = target_nodes[idx]
                    self._encode_literals(
                        literals,
                        builder,
                        node_prefix=to_prefix(successor_node),
                        extra_objects=[successor_node],
                        goal_level_map={},
                    )
                    if succ_action is not None:
                        self._encode_action(
                            succ_action, builder, extra_objects=[successor_node]
                        )
                    if self.relation_dict.goal_satisfaction_derivations and goals:
                        sat_goals = _compute_change_satisfaction(
                            added_facts, removed_facts, goals
                        )
                        self._encode_goal_satisfaction(
                            sat_goals,
                            builder,
                            node_prefix=to_prefix(successor_node),
                            extra_objects=[successor_node],
                        )
            case "action":
                for successor_state, succ_action, idx in dag.successor_iter():
                    successor_node = target_nodes[idx]
                    self._encode_action(
                        succ_action, builder, extra_objects=[successor_node]
                    )

        if self.enable_parent_relation:
            self._encode_parent_relations(builder, target_nodes, dag)
        if self.enable_sibling_relation or self.enable_cousin_relation:
            self._encode_family_relations(builder, target_nodes, dag)

        if self.include_lgan_edges:
            self._add_lgan_nn_edges(builder)

    def _build_data(self, builder: PygBuilderBase) -> HeteroData:
        nodes_dict = builder.node_keys
        data = HeteroData()

        object_names: List[str] = []
        for node_type, nodes in nodes_dict.items():
            if node_type == self.symbol_type_id:
                continue
            names = builder.node_attrs[node_type].get("name", [])
            if names:
                object_names.extend(names)

        symbol_nodes = nodes_dict.get(self.symbol_type_id, [])
        target_indices: List[int] = []
        target_positions: List[int] = []
        target_depths: List[int | None] = []
        target_names: List[str | None] = []

        symbol_attrs = builder.node_attrs.get(self.symbol_type_id, {})
        target_index_list = symbol_attrs.get("target_index", [])
        depth_list = symbol_attrs.get("depth", [])
        target_name_list = symbol_attrs.get("target", [])
        name_list = symbol_attrs.get("name", [])

        for idx, _node in enumerate(symbol_nodes):
            if idx >= len(target_index_list):
                continue
            target_index = target_index_list[idx]
            target_indices.append(target_index)
            target_positions.append(idx)
            depth = depth_list[idx] if idx < len(depth_list) else None
            target_depths.append(depth)
            target_name = target_name_list[idx] if idx < len(target_name_list) else None
            if target_name is None and idx < len(name_list):
                target_name = name_list[idx]
            if target_name is None:
                target_name = _node
            target_names.append(target_name)

        data.object_names = object_names
        data.target_symbol_prefix = self.target_symbol_node_prefix
        data.target_names = target_names
        data.target_positions = target_positions

        for node_type, nodes_of_type in nodes_dict.items():
            feature_size = (
                1 if node_type == self.symbol_type_id else self.relation_dict[node_type]
            )
            data[node_type].x = torch.zeros(
                (len(nodes_of_type), feature_size), dtype=torch.float32
            )
            data[node_type].node_names = list(nodes_of_type)

        for unused_node_type in self.relation_dict.keys() - nodes_dict.keys():
            feature_size = self.relation_dict[unused_node_type]
            data[unused_node_type].x = torch.empty(
                (0, feature_size), dtype=torch.float32
            )
            data[unused_node_type].node_names = []

        if self.symbol_type_id not in nodes_dict:
            data[self.symbol_type_id].x = torch.empty((0, 1), dtype=torch.float32)
            data[self.symbol_type_id].node_names = []
            data[self.symbol_type_id].is_choice = torch.empty(0, dtype=torch.bool)
        else:
            is_choice = torch.zeros(len(symbol_nodes), dtype=torch.bool)
            successor_positions = (
                target_positions[1:]
                if self.exclude_root_candidate
                else target_positions
            )
            if successor_positions:
                is_choice[successor_positions] = True
            data[self.symbol_type_id].is_choice = is_choice

        for edge_type, indices in builder.edge_indices.items():
            if indices:
                data[edge_type].edge_index = (
                    torch.tensor(indices, dtype=torch.long).t().contiguous()
                )
            else:
                data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

        for unused_edge_type in set(self.all_edge_types) - set(
            builder.edge_indices.keys()
        ):
            data[unused_edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

        data.parent_relation = self.parent_relation
        data.object_names = object_names
        data.target_depths = target_depths
        filtered_indices = [idx for idx in target_indices if idx is not None]
        data.target_indices = (
            torch.tensor(filtered_indices, dtype=torch.long)
            if filtered_indices
            else torch.empty(0, dtype=torch.long)
        )
        return data

    def _new_batch_builder(self) -> HorizonBatchBuilder:
        return HorizonBatchBuilder(
            relation_dict=self.relation_dict,
            symbol_type_id=self.symbol_type_id,
            edge_types=self.all_edge_types,
            target_symbol_prefix=self.target_symbol_node_prefix,
            parent_relation=self.parent_relation,
            exclude_root_candidate=self.exclude_root_candidate,
        )

    def _encode_parent_relations(
        self,
        builder: PygBuilderBase,
        target_nodes: Dict[int, str],
        dag: TransitionDAG,
    ) -> None:
        transitions = sorted(
            dag.transitions,
            key=lambda pair: (dag.index(pair[0]), dag.index(pair[1])),
        )
        for parent_state, child_state in transitions:
            parent_idx = dag.index(parent_state)
            child_idx = dag.index(child_state)
            transition_node = self._transition_node_label(parent_idx, child_idx)
            builder.add_node(
                transition_node,
                self.parent_relation,
            )
            builder.add_edge(
                target_nodes[parent_idx],
                transition_node,
                self.symbol_type_id,
                self.parent_relation,
                "0",
            )
            builder.add_edge(
                transition_node,
                target_nodes[parent_idx],
                self.parent_relation,
                self.symbol_type_id,
                "0",
            )
            builder.add_edge(
                target_nodes[child_idx],
                transition_node,
                self.symbol_type_id,
                self.parent_relation,
                "1",
            )
            builder.add_edge(
                transition_node,
                target_nodes[child_idx],
                self.parent_relation,
                self.symbol_type_id,
                "1",
            )

    def _encode_family_relations(
        self,
        builder: PygBuilderBase,
        target_nodes: Dict[int, str],
        dag: TransitionDAG,
    ) -> None:
        # Build parent->children once
        parent_to_children: Dict[int, set[int]] = {}
        for p_t, c_t in dag.transitions:
            p, c = dag.index(p_t), dag.index(c_t)
            parent_to_children.setdefault(p, set()).add(c)

        # Add siblings immediately while tracking seen pairs (to exclude from cousins)
        siblings_seen: set[tuple[int, int]] = set()
        if self.enable_sibling_relation:
            for children in parent_to_children.values():
                ch = sorted(children)
                for i in range(len(ch)):
                    for j in range(i + 1, len(ch)):
                        a, b = ch[i], ch[j]
                        pair = (a, b)
                        if pair in siblings_seen:
                            continue
                        siblings_seen.add(pair)
                        self._emplace_symmetric_relation(
                            builder, self.sibling_relation, target_nodes, a, b
                        )

        # Add cousins immediately while avoiding duplicates and sibling pairs
        if self.enable_cousin_relation:
            cousins_seen: set[tuple[int, int]] = set()
            for g, parents in parent_to_children.items():
                par = sorted(parents)
                for i in range(len(par)):
                    for j in range(i + 1, len(par)):
                        pu, pv = par[i], par[j]
                        cu = parent_to_children.get(pu, set())
                        cv = parent_to_children.get(pv, set())
                        for u in cu:
                            for v in cv:
                                if u == v:
                                    continue
                                a, b = (u, v) if u < v else (v, u)
                                if (a, b) in cousins_seen:
                                    continue
                                if (a, b) in siblings_seen:
                                    continue
                                cousins_seen.add((a, b))
                                self._emplace_symmetric_relation(
                                    builder, self.cousin_relation, target_nodes, a, b
                                )

    def _emplace_symmetric_relation(
        self,
        builder: PygBuilderBase,
        relation: str,
        target_nodes: Dict[int, str],
        a: int,
        b: int,
    ) -> None:
        node = f"{relation}({a}->{b})"
        builder.add_node(node, relation)
        builder.add_edge(
            target_nodes[a],
            node,
            self.symbol_type_id,
            relation,
            "0",
        )
        builder.add_edge(
            node,
            target_nodes[a],
            relation,
            self.symbol_type_id,
            "0",
        )
        builder.add_edge(
            target_nodes[b],
            node,
            self.symbol_type_id,
            relation,
            "1",
        )
        builder.add_edge(
            node,
            target_nodes[b],
            relation,
            self.symbol_type_id,
            "1",
        )

        node = f"{relation}({b}->{a})"
        builder.add_node(node, relation)
        builder.add_edge(
            target_nodes[b],
            node,
            self.symbol_type_id,
            relation,
            "0",
        )
        builder.add_edge(
            node,
            target_nodes[b],
            relation,
            self.symbol_type_id,
            "0",
        )

        builder.add_edge(
            target_nodes[a],
            node,
            self.symbol_type_id,
            relation,
            "1",
        )
        builder.add_edge(
            node,
            target_nodes[a],
            relation,
            self.symbol_type_id,
            "1",
        )

    def to_networkx(self, data: HeteroData) -> nx.MultiGraph:
        """Reconstruct a MultiGraph from HorizonHGraphEncoder data."""
        return self.from_pyg_data(data)

    def from_pyg_data(self, data: HeteroData) -> GraphT:
        graph = self._graph_t(encoding=self._watermark)
        symbol_type = self.symbol_type_id
        parent_type = getattr(data, "parent_relation", self.parent_relation)

        node_names_by_type: Dict[NodeType, List[str]] = {}
        for node_type in data.node_types:
            storage = data[node_type]
            names = list(getattr(storage, "node_names", []))
            node_names_by_type[node_type] = names

        symbol_nodes = node_names_by_type.get(symbol_type, [])
        target_names = tolist(getattr(data, "target_names", []))
        target_depths = tolist(getattr(data, "target_depths", []))
        target_indices = tolist(getattr(data, "target_indices", []))
        target_positions = tolist(getattr(data, "target_positions", []))
        object_names = list(getattr(data, "object_names", []))

        target_info: Dict[int, tuple[str, int | None, int | None]] = {}
        for pos, sym_idx in enumerate(target_positions):
            if sym_idx >= len(symbol_nodes):
                continue
            target_name = (
                target_names[pos] if pos < len(target_names) else symbol_nodes[sym_idx]
            )
            depth = target_depths[pos] if pos < len(target_depths) else None
            index = target_indices[pos] if pos < len(target_indices) else None
            target_info[sym_idx] = target_name, depth, index

        object_iter = iter(object_names)
        for idx, node_key in enumerate(symbol_nodes):
            if idx in target_info:
                target_name, depth, index = target_info[idx]
                node_attrs = {
                    "type": symbol_type,
                    "name": target_name,
                    "depth": depth,
                    "target_index": index,
                }
                graph.add_node(node_key, **node_attrs)
            else:
                obj_name = next(object_iter, node_key)
                graph.add_node(
                    node_key,
                    type=symbol_type,
                    name=obj_name,
                )

        for other_type, names in node_names_by_type.items():
            if other_type == symbol_type:
                continue
            for name in names:
                graph.add_node(name, type=other_type)

        for edge_type, edge_index in data.edge_index_dict.items():
            src_type, pos_str, dst_type = edge_type
            if src_type != symbol_type:
                continue
            pos = int(pos_str)
            src_names = node_names_by_type.get(src_type, [])
            dst_names = node_names_by_type.get(dst_type, [])
            if not src_names or not dst_names:
                continue
            for src_idx, dst_idx in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                if src_idx >= len(src_names) or dst_idx >= len(dst_names):
                    continue
                src_name = src_names[src_idx]
                dst_name = dst_names[dst_idx]
                graph.add_edge(src_name, dst_name, position=pos)

        if parent_type in node_names_by_type:
            target_name_to_idx = {name: idx for idx, name in enumerate(target_names)}
            for transition_name in node_names_by_type[parent_type]:
                parent_idx = None
                child_idx = None
                for neighbor, edge_dict in graph[transition_name].items():
                    for edge_data in edge_dict.values():
                        position = edge_data.get("position")
                        if position == 0:
                            parent_idx = target_name_to_idx.get(neighbor)
                        elif position == 1:
                            child_idx = target_name_to_idx.get(neighbor)
                if parent_idx is not None:
                    graph.nodes[transition_name]["parent"] = parent_idx
                if child_idx is not None:
                    graph.nodes[transition_name]["child"] = child_idx

        graph.graph["encoding"] = self._watermark
        graph.graph["encoder_hash"] = self._watermark
        return graph

    @staticmethod
    def _collect_target_atoms(state: XState) -> set[XAtom]:
        return set(state.atoms(with_statics=False))

    def _target_node_label(self, idx: int) -> str:
        return f"{self.target_symbol_node_prefix}{idx}"

    def _transition_node_label(self, parent_idx: int, child_idx: int) -> str:
        return f"{self.parent_relation}({parent_idx}->{child_idx})"

    def _target_index_from_name(self, node_name: str) -> int:
        if node_name.startswith(self.target_symbol_node_prefix):
            suffix = node_name[len(self.target_symbol_node_prefix) :]
            if suffix.isdigit():
                return int(suffix)
        return 0

    @check_encoded_by_this
    def draw(
        self,
        graph: nx.MultiGraph | HeteroData,
        *,
        ax=None,
        with_labels: bool = True,
        edge_labels: bool = True,
        node_kwargs: dict | None = None,
        edge_kwargs: dict | None = None,
        layout: dict | None = None,
        node_size: float | None = None,
        node_alpha: float | None = None,
        edge_width: float | None = None,
        edge_alpha: float | None = None,
        label_font_size: float | None = None,
        label_nodes: Iterable[str] | None = None,
        label_node_types: Iterable[str] | None = None,
        label_edges: Iterable[tuple[str, ...]] | None = None,
        align_target_nodes: bool = True,
        target_x_spacing: float = 4.0,
        target_y_spacing: float = 2.0,
        layout_seed: int | None = 7,
        symbol_node_scale: float = 1.5,
        non_symbol_linestyle: str | None = "--",
    ):
        if hasattr(graph, "edge_types"):  # HeteroData
            graph = self.to_networkx(graph)
        # Structured, bipartite-inspired layout: objects (left), atoms (middle),
        # and the target-tree (right). Tree relations (parent/sibling/cousin)
        # are placed with the tree, not in the middle.
        if layout is None:
            # derive spacing scale from node size to keep enough room for
            # family relation nodes; larger nodes -> larger spacing
            nk = node_kwargs or {}
            base_node_size_value = nk.get("node_size", node_size)
            if isinstance(base_node_size_value, (list, tuple)):
                base_node_size_value = (
                    base_node_size_value[0] if base_node_size_value else None
                )
            if base_node_size_value is None:
                base_node_size_value = 300.0
            try:
                size_scale = float(base_node_size_value) / 300.0
            except Exception:
                size_scale = 1.0
            # Increase default spacing noticeably to create more room in the
            # target tree for family relations. Scale with node size and keep a
            # healthy minimum even for small nodes.
            spacing_scale = max(20.5, 2.0 * size_scale * max(1.0, symbol_node_scale))
            eff_x = target_x_spacing * spacing_scale
            eff_y = target_y_spacing * spacing_scale
            # identify categories
            symbol_nodes = []  # objects and targets share the symbol type
            target_symbols = []  # symbol nodes with depth attribute
            for node, data in graph.nodes(data=True):
                if data.get("type") == self.symbol_type_id:
                    if "depth" in data:
                        target_symbols.append(node)
                    else:
                        symbol_nodes.append(node)

            tree_relation_types = {
                self.parent_relation,
            }
            tree_relation_nodes = [
                n
                for n, d in graph.nodes(data=True)
                if d.get("type") in tree_relation_types
            ]
            # sibling/cousin relations are treated as regular middle relations
            middle_nodes = [
                n
                for n, d in graph.nodes(data=True)
                if d.get("type") not in tree_relation_types | {self.symbol_type_id}
            ]

            fixed_positions: Dict[str, tuple[float, float]] = {}

            # Determine a common vertical span based on the largest column layer
            def _count_per_depth(nodes: list[str]) -> dict[int, int]:
                depth_map: dict[int, int] = defaultdict(int)
                for name in nodes:
                    d = int(graph.nodes[name].get("depth", 0))
                    depth_map[d] += 1
                return depth_map

            depth_counts = _count_per_depth(target_symbols) if target_symbols else {}
            max_depth_count = max(depth_counts.values()) if depth_counts else 0
            H_count = max(len(symbol_nodes), len(middle_nodes), max_depth_count, 1)
            y_min, y_max = -0.5 * eff_y * (H_count - 1), 0.5 * eff_y * (H_count - 1)

            def _spread(nodes: list[str]) -> dict[str, float]:
                if not nodes:
                    return {}
                nodes_sorted = sorted(nodes)
                if len(nodes_sorted) == 1:
                    return {nodes_sorted[0]: 0.0}
                step = (y_max - y_min) / (len(nodes_sorted) - 1)
                return {n: y_min + i * step for i, n in enumerate(nodes_sorted)}

            # left column: object symbols (equidistant across common span)
            for n, y in _spread(symbol_nodes).items():
                fixed_positions[n] = (-2.0 * eff_x, y)

            # middle column: atoms/literals (non-tree relations) across common span
            for n, y in _spread(middle_nodes).items():
                fixed_positions[n] = (0.0, y)

            # right area: target symbols arranged by depth
            if target_symbols:
                depth_to_nodes: Dict[int, list[str]] = defaultdict(list)
                for node in target_symbols:
                    depth_to_nodes[int(graph.nodes[node].get("depth", 0))].append(node)
                for depth, nodes in sorted(depth_to_nodes.items()):
                    # spread each depth across the same global span to "use all available space"
                    if len(nodes) == 1:
                        y_positions = {nodes[0]: 0.0}
                    else:
                        nodes_sorted = sorted(
                            nodes,
                            key=lambda name: (
                                graph.nodes[name].get(
                                    "target_index", self._target_index_from_name(name)
                                ),
                                name,
                            ),
                        )
                        step = (y_max - y_min) / (len(nodes_sorted) - 1)
                        y_positions = {
                            n: y_min + i * step for i, n in enumerate(nodes_sorted)
                        }
                    for node, y in y_positions.items():
                        fixed_positions[node] = (
                            2.0 * eff_x + depth * eff_x,
                            y,
                        )

            # place tree relation nodes near their incident target symbols; for
            # symmetric relations (sibling/cousin) that connect the same two
            # targets twice (a->b and b->a), offset them in opposite directions
            # along the perpendicular to avoid overlap.
            for rel in tree_relation_nodes:
                rel_type = graph.nodes[rel].get("type")
                # collect pos0/pos1 neighbors if available
                pos0 = pos1 = None
                for nbr, edge_dict in graph[rel].items():
                    if nbr not in fixed_positions:
                        continue
                    for attrs in edge_dict.values():
                        p = attrs.get("position")
                        if p == 0:
                            pos0 = nbr
                        elif p == 1:
                            pos1 = nbr
                if (
                    rel_type in {self.sibling_relation, self.cousin_relation}
                    and pos0 is not None
                    and pos1 is not None
                ):
                    # compute canonical perpendicular using min->max target order
                    try:
                        i0 = self._target_index_from_name(pos0)
                        i1 = self._target_index_from_name(pos1)
                    except Exception:
                        i0, i1 = 0, 1
                    a_name, b_name = (pos0, pos1) if i0 <= i1 else (pos1, pos0)
                    xa, ya = fixed_positions[a_name]
                    xb, yb = fixed_positions[b_name]
                    mx, my = (xa + xb) / 2.0, (ya + yb) / 2.0
                    dx, dy = (xb - xa), (yb - ya)
                    dist = (dx * dx + dy * dy) ** 0.5 or 1e-6
                    ox, oy = -dy / dist, dx / dist
                    # place min->max on +perp and max->min on -perp
                    is_min_to_max = i0 <= i1
                    sign = 1.0 if is_min_to_max else -1.0
                    offset = 0.4 * eff_y
                    fixed_positions[rel] = (
                        mx + sign * offset * ox,
                        my + sign * offset * oy,
                    )
                else:
                    # fallback: average of available neighbors
                    neighbors = [
                        nbr for nbr in graph.neighbors(rel) if nbr in fixed_positions
                    ]
                    if neighbors:
                        xs = [fixed_positions[n][0] for n in neighbors]
                        ys = [fixed_positions[n][1] for n in neighbors]
                        fixed_positions[rel] = (sum(xs) / len(xs), sum(ys) / len(ys))

            # Avoid spring_layout rescaling (which would squash spacing back
            # into a small box). Instead, use the fixed positions directly and
            # place any remaining nodes by averaging their neighbors.
            remaining = [n for n in graph.nodes if n not in fixed_positions]
            if remaining:
                for n in remaining:
                    nbrs = [nb for nb in graph.neighbors(n) if nb in fixed_positions]
                    if nbrs:
                        xs = [fixed_positions[nb][0] for nb in nbrs]
                        ys = [fixed_positions[nb][1] for nb in nbrs]
                        fixed_positions[n] = (sum(xs) / len(xs), sum(ys) / len(ys))
                    else:
                        fixed_positions[n] = (0.0, 0.0)
            layout = fixed_positions

        ax = super().draw(
            graph,
            ax=ax,
            with_labels=with_labels,
            edge_labels=edge_labels,
            node_kwargs=node_kwargs,
            edge_kwargs=edge_kwargs,
            layout=layout,
            node_size=node_size,
            node_alpha=node_alpha,
            edge_width=edge_width,
            edge_alpha=edge_alpha,
            label_font_size=label_font_size,
            label_nodes=label_nodes,
            label_node_types=label_node_types,
            label_edges=label_edges,
            symbol_node_scale=symbol_node_scale,
            non_symbol_linestyle=non_symbol_linestyle,
        )

        return ax
