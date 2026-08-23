"""Public facades for the derived-graph encoder family.

Derived-graph encoders build a homogeneous state graph by expanding a
problem's object and atom tables into a chosen view (star, object clique /
chain / star-first projection, or an atom line graph). All facades share one
backend-neutral configuration type and resolve a concrete backend per
instance via :func:`mifrost.backends._derived_runtime.create_derived_runtime`.

The input must be a *problem* (a ``pymimir.Problem`` or a ``pytyr``
``PlanningTask``). Domains are rejected because object tables are
problem-scoped: the derived views enumerate objects and atoms of one
concrete problem instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import torch
from torch_geometric.data import Batch, Data

from .. import _neutral_core
from ..backends._derived_runtime import (
    DerivedBackendName,
    create_derived_runtime,
)
from .base import (
    ActionBatchInput,
    ActionBatchParam,
    CollateSpecParam,
    EncoderBase,
    GoalBatchInput,
    GoalBatchParam,
    HistorySubgoalsBatchParam,
    StateBatchInput,
    StreamEncoderBase,
    SubgoalLayersInput,
    SubgoalLayersBatchParam,
)
from .types import HomoEncoding, StateInput

if TYPE_CHECKING:
    from .derived_graph_data import DerivedGraphData


def _reclassify(data: Any) -> Any:
    """Return ``data`` as a :class:`DerivedGraphData` with the same payload.

    The native conversion yields plain ``Data``/``Batch`` objects; the
    contract batching rules live on the ``DerivedGraphData`` subclass, so
    materializers reclassify before rewriting fields.
    """
    from .derived_graph_data import DerivedGraphData

    if isinstance(data, DerivedGraphData):
        return data
    converted = (
        Batch(_base_cls=DerivedGraphData)
        if isinstance(data, Batch)
        else DerivedGraphData()
    )
    for key, value in data.to_dict().items():
        converted[key] = value
    return converted


ROLE_NAMES: tuple[str, ...] = (
    "object",
    "fact",
    "goal",
    "subgoal",
    "history",
    "action",
)

EDGE_KIND_NAMES: tuple[str, ...] = (
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
)


@dataclass
class _DerivedEncoderStream(StreamEncoderBase[Data]):
    """Streaming wrapper shared by all derived-graph encoders."""

    _encoder: "_DerivedEncoderBase"

    def __post_init__(self) -> None:
        """Initialize an empty homo stream for incremental encoding."""
        self._stream = self._encoder._runtime.make_stream()
        self._reset_builder()

    def append(
        self,
        state: StateInput,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
        history_subgoals: Any | None = None,
        history_max_steps: int | None = None,
    ) -> int:
        """Append one state encoding to the derived-graph stream."""
        return self._coerce_stream_id(
            self._stream.append(
                state,
                goals=goals,
                actions=actions,
                subgoal_layers=subgoal_layers,
                history_subgoals=history_subgoals,
                history_max_steps=history_max_steps,
            )
        )

    def remove(self, stream_id: int) -> None:
        self._stream.remove(stream_id)

    def update(
        self,
        stream_id: int,
        state: StateInput,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
        history_subgoals: Any | None = None,
        history_max_steps: int | None = None,
    ) -> None:
        self._stream.update(
            stream_id,
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )

    def _reset_builder(self) -> None:
        """Reset stream accumulation state."""
        self._stream.reset()


class _DerivedEncoderBase(EncoderBase[Data]):
    """Shared machinery for the public derived-graph encoder facades."""

    role_names: tuple[str, ...] = ROLE_NAMES
    edge_kind_names: tuple[str, ...] = EDGE_KIND_NAMES

    def __init__(
        self,
        problem: Any,
        config: Any,
        *,
        backend: DerivedBackendName | str | None = None,
    ) -> None:
        self._runtime = create_derived_runtime(problem, config, backend=backend)
        self._engine = self._runtime.engine
        self._config = config
        self.backend = self._runtime.backend_name

    @property
    def engine(self) -> Any:
        """Expose the underlying backend engine."""
        return self._engine

    @property
    def config(self) -> Any:
        """Expose the backend-neutral resolved derived-graph config."""
        return self._config

    def to_networkx(
        self,
        data: Any,
        *,
        include_hyperedges: bool = True,
        include_reverse_edges: bool = True,
        include_line_shares: bool = False,
        include_self_loops: bool = False,
    ) -> Any:
        """Convert one encoded graph into a labeled NetworkX multigraph.

        Decodes the integer-id channels into node ``role``/``predicate``/
        ``sign``/``goal_level`` attributes and per-edge kind names with
        argument positions. Hyperedge memberships become auxiliary pentagon
        nodes wired by dotted membership edges when present.
        """
        from ._derived_visualization import derived_to_networkx

        return derived_to_networkx(
            data,
            include_hyperedges=include_hyperedges,
            include_reverse_edges=include_reverse_edges,
            include_line_shares=include_line_shares,
            include_self_loops=include_self_loops,
        )

    def draw(
        self,
        data: Any,
        *,
        with_labels: bool = True,
        edge_labels: bool = False,
        hide_reverse_edges: bool = True,
        ax: Any | None = None,
        node_size: int | None = None,
        font_size: int = 7,
        legend: bool = True,
    ) -> Any:
        """Render one encoded graph with role shapes and kind styling."""
        from ._derived_visualization import draw_derived

        graph = self.to_networkx(
            data,
            include_line_shares=True,
        )
        return draw_derived(
            graph,
            ax=ax,
            with_labels=with_labels,
            edge_labels=edge_labels,
            hide_reverse_edges=hide_reverse_edges,
            node_size=node_size,
            font_size=font_size,
            legend=legend,
        )

    def summarize(self, data: Any) -> str:
        """Return a short role/edge-kind histogram for one encoded graph."""
        from ._derived_visualization import summarize_derived

        return summarize_derived(data)

    def to_dgl(self, data: Any) -> Any:
        """Convert one encoded graph into a ``(dgl graph, metadata)`` pair.

        Thin delegation to :func:`mifrost.encoders.cross_stack.to_dgl`;
        DGL is imported lazily and an install hint is raised when missing.
        """
        from .cross_stack import to_dgl as _convert_to_dgl

        return _convert_to_dgl(data)

    def to_jraph(self, data: Any) -> Any:
        """Convert one encoded graph into a ``(GraphsTuple, metadata)`` pair.

        Thin delegation to :func:`mifrost.encoders.cross_stack.to_jraph`;
        Jraph/JAX are imported lazily and install hints are raised when
        missing.
        """
        from .cross_stack import to_jraph as _convert_to_jraph

        return _convert_to_jraph(data)

    def _accepted_kwargs(self) -> set[str]:
        return {"history_subgoals", "history_max_steps"}

    def _encode(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        history_subgoals: HistorySubgoalsBatchParam = None,
        history_max_steps: int | None = None,
    ) -> HomoEncoding:
        """Encode one state into homogeneous encoding dictionary."""
        return self._runtime.encode(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )

    def encode(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> HomoEncoding:
        """Encode one state into native ``BatchEncoding``."""
        return super().encode(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            include_metadata=include_metadata,
            **kwargs,
        )

    def _encode_batch(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchParam = None,
        actions: ActionBatchParam = None,
        subgoal_layers: SubgoalLayersBatchParam = None,
        history_subgoals: HistorySubgoalsBatchParam = None,
        history_max_steps: int | None = None,
    ) -> HomoEncoding:
        """Encode one or many states into homogeneous encoding dictionary."""
        return self._runtime.encode_batch(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )

    def encode_batch(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchParam = None,
        actions: ActionBatchParam = None,
        subgoal_layers: SubgoalLayersBatchParam = None,
        batch_attrs: Mapping[str, Any] | None = None,
        collate_spec: CollateSpecParam = None,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> HomoEncoding:
        """Encode one or many states into native ``BatchEncoding``."""
        return super().encode_batch(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            batch_attrs=batch_attrs,
            collate_spec=collate_spec,
            include_metadata=include_metadata,
            **kwargs,
        )

    def _stack_membership(self, data: DerivedGraphData) -> DerivedGraphData:
        """Restack raw hyperedge incidence fields into PyG hyperedge form.

        The native fields store ``hyperedge_sizes`` (members per hyperedge,
        possibly zero), ``hyperedge_node_indices`` (flattened member object
        nodes), ``hyperedge_ids`` (per-graph aranges, already offset across
        batches by the builder), and ``hyperedge_role_ids`` (one role per
        hyperedge). Ids are expanded along sizes into the stacked [2, M]
        ``hyperedge_index`` layout (node row, hyperedge row); roles become
        ``hyperedge_attr_ids``. Values stay global in batch mode so the
        result is a well-formed PyG batch whose :class:`DerivedGraphData`
        increment rules reproduce it under ``from_data_list``.
        """
        nodes = getattr(data, "hyperedge_node_indices", None)
        ids = getattr(data, "hyperedge_ids", None)
        sizes = getattr(data, "hyperedge_sizes", None)
        if nodes is None or ids is None or sizes is None:
            return data
        data = _reclassify(data)
        del data.hyperedge_node_indices
        del data.hyperedge_ids
        nodes = nodes.long()
        ids = ids.long()
        sizes = sizes.long()
        owner = torch.repeat_interleave(ids, sizes.clamp_min(0))
        roles = getattr(data, "hyperedge_role_ids", None)
        if roles is not None:
            del data.hyperedge_role_ids
            data.hyperedge_attr_ids = roles.long()
        for key in ("hyperedge_sizes", "hyperedge_sizes_ptr", "hyperedge_counts"):
            if hasattr(data, key):
                delattr(data, key)
        data.hyperedge_index = torch.stack([nodes, owner], 0)
        return data


class StarGraphEncoder(_DerivedEncoderBase):
    """
    Reified star view over objects and atoms.

    Each atom becomes a fact node connected to its argument object nodes via
    star (hub) edges; goal atoms are marked in the goal channel.

    The input must be a ``pymimir.Problem`` or ``pytyr.PlanningTask``.
    Domains are rejected because object tables are problem-scoped.
    """

    def __init__(
        self,
        problem: Any,
        *,
        backend: DerivedBackendName | str | None = None,
        include_reverse_edges: bool = True,
        export_node_names: bool = True,
    ) -> None:
        """Create a star-view derived-graph encoder for one problem."""
        config = _neutral_core.SemanticDerivedGraphEncoderConfig(
            node_universe="objects_and_atoms",
            atom_expansion="star",
            include_reverse_edges=include_reverse_edges,
            export_node_names=export_node_names,
        )
        super().__init__(problem, config, backend=backend)
        self.include_reverse_edges = include_reverse_edges
        self.export_node_names = export_node_names

    def stream(self) -> StarGraphEncoderStream:
        """Create a streaming encoder sharing this encoder's engine."""
        return StarGraphEncoderStream(self)


@dataclass
class StarGraphEncoderStream(_DerivedEncoderStream):
    """Streaming wrapper for ``StarGraphEncoder``."""

    _encoder: "StarGraphEncoder"


class ObjectGraphEncoder(_DerivedEncoderBase):
    """
    Objects-only projection with a configurable atom expansion.

    Only object nodes are materialized; each atom connects its arguments
    according to ``atom_expansion``:

    - ``"clique"``: fully connect the atom's arguments.
    - ``"chain"``: chain the arguments in order.
    - ``"star_first"``: hub on the first argument.

    The input must be a ``pymimir.Problem`` or ``pytyr.PlanningTask``.
    Domains are rejected because object tables are problem-scoped.
    """

    def __init__(
        self,
        problem: Any,
        *,
        atom_expansion: str = "clique",
        backend: DerivedBackendName | str | None = None,
        include_reverse_edges: bool = True,
        export_node_names: bool = True,
    ) -> None:
        """Create an objects-only derived-graph encoder for one problem."""
        config = _neutral_core.SemanticDerivedGraphEncoderConfig(
            node_universe="objects_only",
            atom_expansion=atom_expansion,
            include_reverse_edges=include_reverse_edges,
            export_node_names=export_node_names,
        )
        super().__init__(problem, config, backend=backend)
        self.atom_expansion = atom_expansion
        self.include_reverse_edges = include_reverse_edges
        self.export_node_names = export_node_names

    def _accepted_kwargs(self) -> set[str]:
        return super()._accepted_kwargs() | {"atom_expansion"}

    def _encode(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        history_subgoals: HistorySubgoalsBatchParam = None,
        history_max_steps: int | None = None,
        atom_expansion: str | None = None,
    ) -> HomoEncoding:
        self._check_atom_expansion(atom_expansion)
        return super()._encode(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )

    def _encode_batch(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchParam = None,
        actions: ActionBatchParam = None,
        subgoal_layers: SubgoalLayersBatchParam = None,
        history_subgoals: HistorySubgoalsBatchParam = None,
        history_max_steps: int | None = None,
        atom_expansion: str | None = None,
    ) -> HomoEncoding:
        self._check_atom_expansion(atom_expansion)
        return super()._encode_batch(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )

    def _check_atom_expansion(self, atom_expansion: str | None) -> None:
        if atom_expansion is not None and atom_expansion != self.atom_expansion:
            raise ValueError(
                "atom_expansion is fixed at construction "
                f"(configured {self.atom_expansion!r}, got {atom_expansion!r})"
            )

    def stream(self) -> ObjectGraphEncoderStream:
        """Create a streaming encoder sharing this encoder's engine."""
        return ObjectGraphEncoderStream(self)


@dataclass
class ObjectGraphEncoderStream(_DerivedEncoderStream):
    """Streaming wrapper for ``ObjectGraphEncoder``."""

    _encoder: "ObjectGraphEncoder"


class AtomLineGraphEncoder(_DerivedEncoderBase):
    """
    Atom line graph over the star universe.

    Builds the star view (objects and atoms) and additionally connects
    co-occurring argument edges of high-degree atoms through line-share
    edges bounded by ``line_graph_max_degree``.

    The input must be a ``pymimir.Problem`` or ``pytyr.PlanningTask``.
    Domains are rejected because object tables are problem-scoped.
    """

    def __init__(
        self,
        problem: Any,
        *,
        line_graph_max_degree: int = 32,
        backend: DerivedBackendName | str | None = None,
        include_reverse_edges: bool = True,
        export_node_names: bool = True,
    ) -> None:
        """Create an atom line-graph encoder for one problem."""
        config = _neutral_core.SemanticDerivedGraphEncoderConfig(
            node_universe="objects_and_atoms",
            atom_expansion="star",
            include_line_graph=True,
            include_reverse_edges=include_reverse_edges,
            line_graph_max_degree=line_graph_max_degree,
            export_node_names=export_node_names,
        )
        super().__init__(problem, config, backend=backend)
        self.line_graph_max_degree = line_graph_max_degree
        self.include_reverse_edges = include_reverse_edges
        self.export_node_names = export_node_names

    def stream(self) -> AtomLineGraphEncoderStream:
        """Create a streaming encoder sharing this encoder's engine."""
        return AtomLineGraphEncoderStream(self)


@dataclass
class AtomLineGraphEncoderStream(_DerivedEncoderStream):
    """Streaming wrapper for ``AtomLineGraphEncoder``."""

    _encoder: "AtomLineGraphEncoder"


class HypergraphIncidenceEncoder(_DerivedEncoderBase):
    """
    Hyperedge incidence view over objects and atoms.

    Every encoded literal instance (static facts, state facts, goal and
    subgoal literals, history literals, grounded actions) becomes one
    hyperedge over its argument object nodes. ``encode_pyg`` restacks the
    native incidence fields into ``hyperedge_index`` /
    ``hyperedge_attr_ids`` so stock PyG hypergraph layers consume the output.

    The input must be a ``pymimir.Problem`` or ``pytyr.PlanningTask``.
    Domains are rejected because object tables are problem-scoped.
    """

    def __init__(
        self,
        problem: Any,
        *,
        backend: DerivedBackendName | str | None = None,
        include_reverse_edges: bool = True,
        export_node_names: bool = True,
    ) -> None:
        """Create a hyperedge-incidence derived-graph encoder for one problem."""
        config = _neutral_core.SemanticDerivedGraphEncoderConfig(
            node_universe="objects_and_atoms",
            atom_expansion="star",
            include_hyperedge_incidence=True,
            include_reverse_edges=include_reverse_edges,
            export_node_names=export_node_names,
        )
        super().__init__(problem, config, backend=backend)
        self.include_reverse_edges = include_reverse_edges
        self.export_node_names = export_node_names

    def _accepted_kwargs(self) -> set[str]:
        return super()._accepted_kwargs() | {"hyperedge"}

    def _check_hyperedge(self, hyperedge: bool | None) -> None:
        if hyperedge is not None and not hyperedge:
            raise ValueError(
                "hyperedge is fixed at construction "
                "(hyperedge incidence is always enabled)"
            )

    def _encode(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        history_subgoals: HistorySubgoalsBatchParam = None,
        history_max_steps: int | None = None,
        hyperedge: bool | None = None,
    ) -> HomoEncoding:
        self._check_hyperedge(hyperedge)
        return super()._encode(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )

    def _encode_batch(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchParam = None,
        actions: ActionBatchParam = None,
        subgoal_layers: SubgoalLayersBatchParam = None,
        history_subgoals: HistorySubgoalsBatchParam = None,
        history_max_steps: int | None = None,
        hyperedge: bool | None = None,
    ) -> HomoEncoding:
        self._check_hyperedge(hyperedge)
        return super()._encode_batch(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )

    def encode_pyg(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> DerivedGraphData:
        """Encode one state into PyG data with stacked hyperedge membership."""
        data = super().encode_pyg(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            include_metadata=include_metadata,
            **kwargs,
        )
        return self._stack_membership(data)

    def encode_batch_pyg(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchParam = None,
        actions: ActionBatchParam = None,
        subgoal_layers: SubgoalLayersBatchParam = None,
        batch_attrs: Mapping[str, Any] | None = None,
        collate_spec: CollateSpecParam = None,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> DerivedGraphData:
        """Encode states into a PyG batch with stacked hyperedge membership."""
        data = super().encode_batch_pyg(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            batch_attrs=batch_attrs,
            collate_spec=collate_spec,
            include_metadata=include_metadata,
            **kwargs,
        )
        return self._stack_membership(data)

    def stream(self) -> HypergraphIncidenceEncoderStream:
        """Create a streaming encoder sharing this engine."""
        return HypergraphIncidenceEncoderStream(self)


@dataclass
class HypergraphIncidenceEncoderStream(_DerivedEncoderStream):
    """Streaming wrapper for ``HypergraphIncidenceEncoder``."""

    _encoder: "HypergraphIncidenceEncoder"

    def flush_pyg(
        self, *, as_batch: bool = True, include_metadata: bool = True
    ) -> DerivedGraphData:
        """Flush accumulated states and stack hyperedge membership."""
        data = super().flush_pyg(as_batch=as_batch, include_metadata=include_metadata)
        return self._encoder._stack_membership(data)


class TupleTensorEncoder(_DerivedEncoderBase):
    """
    Star view over objects and atoms with reified tuple channels.

    Builds the star view (objects and atoms) and additionally exposes every
    encoded literal instance as a variable-arity tuple: ``tuple_args`` holds
    the argument node ids, ``tuple_ptr`` is the per-graph CSR over tuples,
    and ``tuple_rel_ids`` / ``tuple_role_ids`` index the shared predicate and
    role vocabularies. All values are stored per-graph-local, matching the
    :class:`~mifrost.encoders.derived_graph_data.DerivedGraphData` batching
    contract.

    The input must be a ``pymimir.Problem`` or ``pytyr.PlanningTask``.
    Domains are rejected because object tables are problem-scoped.
    """

    def __init__(
        self,
        problem: Any,
        *,
        backend: DerivedBackendName | str | None = None,
        export_node_names: bool = True,
    ) -> None:
        """Create a tuple-channel derived-graph encoder for one problem."""
        config = _neutral_core.SemanticDerivedGraphEncoderConfig(
            node_universe="objects_and_atoms",
            atom_expansion="star",
            include_tuple_tensors=True,
            export_node_names=export_node_names,
        )
        super().__init__(problem, config, backend=backend)
        self.export_node_names = export_node_names

    def encode_pyg(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> DerivedGraphData:
        """Encode one state into PyG data with contract tuple channels."""
        data = super().encode_pyg(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            include_metadata=include_metadata,
            **kwargs,
        )
        return self._materialize_tuples(data)

    def encode_batch_pyg(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchParam = None,
        actions: ActionBatchParam = None,
        subgoal_layers: SubgoalLayersBatchParam = None,
        batch_attrs: Mapping[str, Any] | None = None,
        collate_spec: CollateSpecParam = None,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> DerivedGraphData:
        """Encode states into a PyG batch with contract tuple channels."""
        data = super().encode_batch_pyg(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            batch_attrs=batch_attrs,
            collate_spec=collate_spec,
            include_metadata=include_metadata,
            **kwargs,
        )
        return self._materialize_tuples(data)

    def _materialize_tuples(self, data: DerivedGraphData) -> DerivedGraphData:
        """Pop raw native tuple fields and install the contract channels.

        Builder increments already made batch values global (node-offset for
        ``tuple_args``, cumulative tuple counts for the id channels), so they
        are kept verbatim; ``tuple_ptr`` becomes one global CSR over all slot
        sizes. Re-batching single graphs via ``Batch.from_data_list``
        reproduces the global id channels exactly; ``tuple_ptr`` may carry
        duplicated boundary entries where graph fragments meet (per-graph
        CSRs concatenate, a global CSR does not) — slice with
        :meth:`DerivedGraphData.padded_tuple_matrix` to stay shape-stable.
        """
        slot_sizes = getattr(data, "tuple_slot_sizes", None)
        if slot_sizes is None:
            return data
        data = _reclassify(data)
        sizes = slot_sizes.long().view(-1)
        data.tuple_args = getattr(data, "tuple_args").long()
        data.tuple_rel_ids = getattr(data, "tuple_rel_ids").long()
        data.tuple_role_ids = getattr(data, "tuple_role_ids").long()
        data.tuple_ptr = torch.cat(
            (
                torch.zeros(1, dtype=torch.long),
                torch.cumsum(sizes, 0),
            )
        )
        for attr in ("tuple_slot_sizes", "tuple_slot_sizes_ptr", "tuple_counts"):
            try:
                delattr(data, attr)
            except (AttributeError, KeyError):
                pass
        return data

    def stream(self) -> TupleTensorEncoderStream:
        """Create a streaming encoder sharing this encoder's engine."""
        return TupleTensorEncoderStream(self)


@dataclass
class TupleTensorEncoderStream(_DerivedEncoderStream):
    """Streaming wrapper for ``TupleTensorEncoder``."""

    _encoder: "TupleTensorEncoder"

    def flush_pyg(
        self, *, as_batch: bool = True, include_metadata: bool = True
    ) -> DerivedGraphData:
        """Flush accumulated states and materialize contract tuple channels."""
        data = super().flush_pyg(as_batch=as_batch, include_metadata=include_metadata)
        return self._encoder._materialize_tuples(data)


class TransformerBiasEncoder(_DerivedEncoderBase):
    """
    Objects-only clique projection with shortest-path distance biases.

    Only object nodes are materialized (clique atom expansion); every pair of
    objects connected through shared fact instances within ``spd_max_hops``
    bipartite hops is reported via ``spd_src`` / ``spd_dst`` / ``spd_dist``
    for transformer attention-bias consumption.

    The input must be a ``pymimir.Problem`` or ``pytyr.PlanningTask``.
    Domains are rejected because object tables are problem-scoped.
    """

    def __init__(
        self,
        problem: Any,
        *,
        spd_max_hops: int = 4,
        backend: DerivedBackendName | str | None = None,
        include_reverse_edges: bool = True,
        export_node_names: bool = True,
    ) -> None:
        """Create an spd-annotated objects-only derived-graph encoder."""
        config = _neutral_core.SemanticDerivedGraphEncoderConfig(
            node_universe="objects_only",
            atom_expansion="clique",
            include_spd=True,
            spd_max_hops=spd_max_hops,
            include_reverse_edges=include_reverse_edges,
            export_node_names=export_node_names,
        )
        super().__init__(problem, config, backend=backend)
        self.spd_max_hops = spd_max_hops
        self.include_reverse_edges = include_reverse_edges
        self.export_node_names = export_node_names

    def stream(self) -> TransformerBiasEncoderStream:
        """Create a streaming encoder sharing this engine."""
        return TransformerBiasEncoderStream(self)


@dataclass
class TransformerBiasEncoderStream(_DerivedEncoderStream):
    """Streaming wrapper for ``TransformerBiasEncoder``."""

    _encoder: "TransformerBiasEncoder"


__all__ = [
    "AtomLineGraphEncoder",
    "AtomLineGraphEncoderStream",
    "EDGE_KIND_NAMES",
    "HypergraphIncidenceEncoder",
    "HypergraphIncidenceEncoderStream",
    "ObjectGraphEncoder",
    "ObjectGraphEncoderStream",
    "ROLE_NAMES",
    "StarGraphEncoder",
    "StarGraphEncoderStream",
    "TransformerBiasEncoder",
    "TransformerBiasEncoderStream",
    "TupleTensorEncoder",
    "TupleTensorEncoderStream",
]
