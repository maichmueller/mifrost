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
    every derived facade reclassifies before handing the carrier out. The
    private ``Batch`` bookkeeping (``_num_graphs`` / ``_slice_dict`` /
    ``_inc_dict``) lives outside ``to_dict()``, so it is copied across
    explicitly to keep ``to_data_list()`` working on rebuilt batches.
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
    source_store = getattr(data, "_store", None)
    if source_store is not None:
        for key, value in vars(source_store).items():
            if key.startswith("_") and key != "_parent":
                setattr(converted._store, key, value)
    return converted


ROLE_NAMES: tuple[str, ...] = (
    "object",
    "fact",
    "goal",
    "subgoal",
    "history",
    "action",
    "anchor",
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
    "unary_self",
)

#: ``x_ids`` column order, mirrored by ``hyperedge_attr_ids`` and by the
#: stacked ``tuple_attr_ids`` convenience channel.
_ATTR_COLUMN_ORDER: tuple[str, ...] = (
    "role",
    "relation_id_plus_one",
    "sign",
    "goal_level",
    "history_dt",
    "category",
)

#: Per-tuple id channels stacked into ``tuple_attr_ids``, in
#: ``_ATTR_COLUMN_ORDER``. ``tuple_rel_ids`` stores the *raw* relation id
#: (``-1`` = none), so column 1 is shifted by one to match ``x_ids``.
_TUPLE_ATTR_SOURCES: tuple[str, ...] = (
    "tuple_role_ids",
    "tuple_rel_ids",
    "tuple_sign_ids",
    "tuple_level_ids",
    "tuple_dt_ids",
    "tuple_category_ids",
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

    def _to_pyg(
        self,
        encoding: Any,
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> Any:
        """Materialize the shared derived carrier for flushed stream output."""
        return self._encoder._finalize_pyg(
            super()._to_pyg(
                encoding, as_batch=as_batch, include_metadata=include_metadata
            )
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
        config = _neutral_core.normalize_semantic_derived_config(config)
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
        """Expose the backend-neutral resolved derived-graph config.

        The config is the *normalized* one the engine actually runs with, so
        ``config.include_tuple_tensors`` reports the truth for
        ``objects_only`` facades (which force the tuple instance table on to
        stay lossless for arity >= 3 literals).
        """
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

    def _to_pyg(
        self,
        encoding: Any,
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> DerivedGraphData:
        """Materialize the shared derived carrier from a native encoding."""
        return self._finalize_pyg(
            super()._to_pyg(
                encoding, as_batch=as_batch, include_metadata=include_metadata
            )
        )

    def _finalize_pyg(self, data: Any) -> DerivedGraphData:
        """Turn raw native fields into the documented carrier surface.

        Every derived facade runs the same steps, so the tuple and hyperedge
        post-processing lives here rather than in the facades that happen to
        switch those channels on: ``objects_only`` forces the tuple instance
        table on, which means ``ObjectGraphEncoder`` and
        ``TransformerBiasEncoder`` carry tuple channels too.
        """
        data = _reclassify(data)
        data = self._drop_placeholder_x(data)
        data = self._nest_object_names(data)
        data = self._materialize_tuples(data)
        data = self._stack_membership(data)
        return data

    @staticmethod
    def _nest_object_names(data: DerivedGraphData) -> DerivedGraphData:
        """Split the natively flattened ``object_names`` back per graph.

        ``node_names`` is nested per graph on every path, but the shared
        native converter can only split ``object_names`` when the node-name
        table *is* the object table. This family always reifies more than
        objects (fact/action nodes, the anchor), so the native and stream
        paths hand over one flattened inner list covering the whole batch
        while ``Batch.from_data_list`` nests correctly - the one metadata key
        where the two batching paths disagreed in *shape*. For a same-problem
        batch that loses which name came from which graph; for a batch whose
        graphs have different object counts it is unrecoverable without
        re-deriving the split.

        The split is authoritative rather than heuristic: ``object_names``
        names exactly the role-0 (object) nodes of each graph, in node order,
        so counting role-0 rows per graph reproduces the per-graph object
        tables exactly. Verified on every facade and fixture:
        ``(x_ids[:, 0] == 0).sum() == len(object_names)`` and
        ``node_names[:that] == object_names``.
        """
        names = getattr(data, "object_names", None)
        if not isinstance(names, list) or not names:
            return data
        if not isinstance(names[0], list):
            return data  # single graph: already a flat list of names
        ptr = getattr(data, "ptr", None)
        num_graphs = (
            int(ptr.numel()) - 1 if isinstance(ptr, torch.Tensor) else len(names)
        )
        if len(names) == num_graphs or len(names) != 1:
            return data  # already per-graph, or a shape this cannot interpret
        flat = names[0]
        counts = _DerivedEncoderBase._object_counts(data, num_graphs)
        if counts is None:
            return data
        if sum(counts) != len(flat):
            raise ValueError(
                "cannot split object_names per graph: the batch carries "
                f"{len(flat)} object names but its graphs hold "
                f"{sum(counts)} role-0 object nodes ({counts}); object_names "
                "must name exactly the object nodes of each graph"
            )
        nested: list[list[str]] = []
        start = 0
        for count in counts:
            nested.append(flat[start : start + count])
            start += count
        data.object_names = nested
        return data

    @staticmethod
    def _object_counts(data: DerivedGraphData, num_graphs: int) -> list[int] | None:
        """Return the per-graph count of role-0 (object) nodes."""
        x_ids = getattr(data, "x_ids", None)
        batch = getattr(data, "batch", None)
        if not isinstance(x_ids, torch.Tensor) or x_ids.dim() != 2:
            return None
        if not isinstance(batch, torch.Tensor) or batch.numel() != x_ids.size(0):
            return None
        objects = x_ids[:, 0] == 0
        return torch.bincount(
            batch[objects].long(), minlength=max(num_graphs, 0)
        ).tolist()

    @staticmethod
    def _drop_placeholder_x(data: DerivedGraphData) -> DerivedGraphData:
        """Delete the native all-zero ``x`` lane; ``x_ids`` is the only source.

        The node-feature lane is shared with encoder families that put real
        features in ``x``. This family does not: every node channel is an
        integer id in ``x_ids``, so the lane arrives as an all-zero ``[N, 6]``
        tensor. Shipping it is a silent-loss trap - a consumer reaching for
        ``data.x`` out of PyG habit gets a constant graph
        (``GCNConv(data.x, data.edge_index)`` returns identical rows for every
        node) with no error anywhere. The family contract is that no
        information loss is silent, so the placeholder is dropped: ``data.x``
        is ``None`` and a stock layer raises instead.

        ``num_nodes`` is pinned into the store first, so nothing depends on
        ``x`` to size the graph.
        """
        x_ids = getattr(data, "x_ids", None)
        if not isinstance(x_ids, torch.Tensor) or x_ids.dim() == 0:
            return data
        if data._store.get("num_nodes", None) is None:
            data.num_nodes = int(x_ids.size(0))
        if data._store.get("x", None) is not None:
            del data.x
        return data

    def _materialize_tuples(self, data: DerivedGraphData) -> DerivedGraphData:
        """Install the contract tuple channels over the raw native fields.

        The builder already made batch values global (``tuple_args`` carries
        the node offset; the per-tuple id channels simply concatenate), so
        they are kept verbatim. The native ``tuple_slot_sizes`` is retained as
        the carrier's ``tuple_sizes``: it is the one representation that
        concatenates identically under native batching and under
        ``Batch.from_data_list``, and :class:`DerivedGraphData` derives
        ``tuple_ptr`` from it, so the global CSR agrees on every path
        (``tuple_ptr.numel() - 1 == tuple_rel_ids.numel()`` always holds).
        ``tuple_attr_ids`` stacks the six per-tuple id channels in ``x_ids``
        column order.
        """
        slot_sizes = getattr(data, "tuple_slot_sizes", None)
        if slot_sizes is None:
            return data
        data = _reclassify(data)
        data.tuple_sizes = slot_sizes.long().view(-1)
        data.tuple_args = data.tuple_args.long()
        for attr in _TUPLE_ATTR_SOURCES:
            value = getattr(data, attr, None)
            if value is not None:
                setattr(data, attr, value.long().view(-1))
        columns = [getattr(data, attr, None) for attr in _TUPLE_ATTR_SOURCES]
        if all(column is not None for column in columns):
            stacked = torch.stack(columns, dim=1)
            # ``tuple_rel_ids`` stores the raw relation id; ``x_ids`` column 1
            # stores ``relation_id + 1`` with 0 meaning "none".
            stacked[:, 1] = stacked[:, 1] + 1
            data.tuple_attr_ids = stacked
        counts = getattr(data, "tuple_counts", None)
        if counts is not None:
            data.num_tuples = counts.long().view(-1)
        # Force the derived CSR into the store so ``tuple_ptr`` is present in
        # ``keys()`` / ``to_dict()`` even before anything reads the property.
        data._sync_tuple_ptr()
        for attr in ("tuple_slot_sizes", "tuple_slot_sizes_ptr", "tuple_counts"):
            try:
                delattr(data, attr)
            except (AttributeError, KeyError):
                pass
        return data

    def _stack_membership(self, data: DerivedGraphData) -> DerivedGraphData:
        """Restack raw hyperedge incidence fields into PyG hyperedge form.

        The native fields store ``hyperedge_sizes`` (members per hyperedge,
        always >= 1 because an arity-0 instance takes the anchor node as its
        single member), ``hyperedge_node_indices`` (flattened member object
        nodes), ``hyperedge_ids`` (per-graph aranges, already offset across
        batches by the builder), ``hyperedge_counts`` (hyperedges per graph)
        and ``hyperedge_attr_rows`` (``[M, 6]``, one label row per hyperedge
        in ``x_ids`` column order). Ids are expanded along sizes into the
        stacked ``[2, sum(sizes)]`` ``hyperedge_index`` layout (node row,
        hyperedge row); the label rows are exposed unchanged as
        ``hyperedge_attr_ids`` and the counts as ``num_hyperedges``. Values
        stay global in batch mode so the result is a well-formed PyG batch
        whose :class:`DerivedGraphData` increment rules reproduce it under
        ``from_data_list``.
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
        rows = getattr(data, "hyperedge_attr_rows", None)
        if rows is not None:
            del data.hyperedge_attr_rows
            data.hyperedge_attr_ids = rows.long().view(-1, len(_ATTR_COLUMN_ORDER))
        counts = getattr(data, "hyperedge_counts", None)
        if counts is not None:
            data.num_hyperedges = counts.long().view(-1)
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

    Arity-1 literals emit a ``unary_self`` loop and arity-0 literals a
    ``nullary_self`` loop on the trailing anchor node, so every instance
    leaves a labeled trace. Pairwise edges still cannot say which pairs
    belong to the same arity >= 3 literal, so this universe *always* carries
    the tuple instance table (``include_tuple_tensors`` is forced on by the
    engine's config normalization) — that table is the authoritative,
    lossless instance list.

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

    ``hyperedge_attr_ids`` is ``[M, 6]`` in ``x_ids`` column order (role,
    ``relation_id + 1``, sign, goal level, history dt, category) — one label
    row per instance, not just its role. A zero-arity instance takes the
    anchor node as its single member, so no hyperedge is empty and
    ``hyperedge_index[1].max() + 1 == hyperedge_attr_ids.size(0)`` — exactly
    the edge count ``torch_geometric.nn.HypergraphConv`` infers when
    ``num_edges`` is not passed. ``num_hyperedges`` carries the authoritative
    per-graph count.

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

    def stream(self) -> HypergraphIncidenceEncoderStream:
        """Create a streaming encoder sharing this engine."""
        return HypergraphIncidenceEncoderStream(self)


@dataclass
class HypergraphIncidenceEncoderStream(_DerivedEncoderStream):
    """Streaming wrapper for ``HypergraphIncidenceEncoder``."""

    _encoder: "HypergraphIncidenceEncoder"


class TupleTensorEncoder(_DerivedEncoderBase):
    """
    Star view over objects and atoms with reified tuple channels.

    Builds the star view (objects and atoms) and additionally exposes every
    encoded literal instance as a variable-arity tuple: ``tuple_args`` holds
    the argument node ids, ``tuple_sizes`` their per-tuple arity and
    ``tuple_ptr`` the CSR derived from it, while ``tuple_rel_ids`` /
    ``tuple_role_ids`` / ``tuple_sign_ids`` / ``tuple_level_ids`` /
    ``tuple_dt_ids`` / ``tuple_category_ids`` mirror the six ``x_ids``
    channels one row per instance (also stacked as ``tuple_attr_ids``).

    Relation ids are now **unified**: ``tuple_rel_ids`` indexes
    ``vocab_relations``, which is ``vocab_predicates ++ vocab_actions``, so a
    predicate keeps id ``p`` and action schema ``a`` gets ``num_predicates +
    a``. Action-role and predicate-role tuples can no longer collide, one
    ``Embedding(len(vocab_relations) + 1, ...)`` covers the whole channel, and
    ``num_predicates`` splits the space without string compares. The one
    remaining caveat is the ``-1`` sentinel: ``tuple_rel_ids`` stores the
    *raw* relation id, using ``-1`` for "no relation", whereas ``x_ids[:, 1]``
    and ``tuple_attr_ids[:, 1]`` store ``relation_id + 1`` with ``0`` meaning
    the same thing — shift before sharing an embedding table with ``x_ids``.

    The graph view is a *set* (a repeated instance interns to one node and
    emits its edges once) while this tuple table is a *multiset*: two
    identical goal literals produce two rows pointing at the same node. That
    is what makes the table, not the topology, the lossless record.

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

    def stream(self) -> TupleTensorEncoderStream:
        """Create a streaming encoder sharing this encoder's engine."""
        return TupleTensorEncoderStream(self)


@dataclass
class TupleTensorEncoderStream(_DerivedEncoderStream):
    """Streaming wrapper for ``TupleTensorEncoder``."""

    _encoder: "TupleTensorEncoder"


class TransformerBiasEncoder(_DerivedEncoderBase):
    """
    Objects-only clique projection with shortest-path distance biases.

    Only object nodes are materialized (clique atom expansion); every pair of
    objects connected through shared fact instances within ``spd_max_hops``
    bipartite hops is reported via ``spd_src`` / ``spd_dst`` / ``spd_dist``
    for transformer attention-bias consumption.

    Being an ``objects_only`` view, it inherits that universe's contract: an
    anchor node is appended, and the tuple instance table is always carried
    (see :class:`ObjectGraphEncoder`). The anchor is not an object, so the
    ``spd_*`` lanes never reference it.

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
