from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Integral
from typing import (
    Any,
    TYPE_CHECKING,
    Generic,
    Iterable,
    Mapping,
    Sequence,
    TypeAlias,
    TypeGuard,
    TypeVar,
)

from .conversion import _encoding_dict_to_pyg, to_pyg
from ._rustworkx_dag import RXStateDAG
from ..graph_fields import CollateSpec
from .types import (
    BatchEncodingInput,
    BatchEncodingLike,
    BatchParam,
    EncodingDict,
    GoalLiteralInput,
    GroundActionInput,
    HistorySubgoalInput,
    StateInput,
)
from .._core import BatchEncoding, TransitionDAG

if TYPE_CHECKING:
    from torch_geometric.data import Data, HeteroData

    PygDataT = TypeVar("PygDataT", Data, HeteroData)
else:
    PygDataT = TypeVar("PygDataT")
StateBatchInput: TypeAlias = Iterable[StateInput] | StateInput
GoalBatchInput: TypeAlias = Iterable[GoalLiteralInput] | None
SubgoalLayersInput: TypeAlias = Iterable[Iterable[GoalLiteralInput]] | None
ActionBatchInput: TypeAlias = Iterable[GroundActionInput] | None
GoalBatchParam: TypeAlias = (
    BatchParam[Iterable[GoalLiteralInput]]
    | Iterable[GoalLiteralInput]
    | Sequence[Iterable[GoalLiteralInput] | None]
    | None
)
ActionBatchParam: TypeAlias = (
    BatchParam[Iterable[GroundActionInput]]
    | Iterable[GroundActionInput]
    | Sequence[Iterable[GroundActionInput] | None]
    | None
)
SubgoalLayersBatchParam: TypeAlias = (
    BatchParam[Iterable[Iterable[GoalLiteralInput]]]
    | Iterable[Iterable[GoalLiteralInput]]
    | Sequence[Iterable[Iterable[GoalLiteralInput]] | None]
    | None
)
HistorySubgoalsBatchParam: TypeAlias = (
    BatchParam[HistorySubgoalInput]
    | HistorySubgoalInput
    | Sequence[HistorySubgoalInput | None]
    | None
)
SuccessorBatchParam: TypeAlias = BatchParam[StateInput] | StateBatchInput | None
DagBatchParam: TypeAlias = (
    BatchParam[TransitionDAG | RXStateDAG]
    | Iterable[TransitionDAG | RXStateDAG | None]
    | TransitionDAG
    | RXStateDAG
    | None
)
CollateSpecParam: TypeAlias = Mapping[str, CollateSpec | Mapping[str, Any]] | None


def _is_batch_encoding_like(value: object) -> TypeGuard[BatchEncodingLike]:
    return isinstance(value, BatchEncoding) or isinstance(value, BatchEncodingLike)


def _normalize_collate_spec(
    collate_spec: CollateSpecParam,
) -> dict[str, dict[str, Any]] | None:
    if collate_spec is None:
        return None
    out: dict[str, dict[str, Any]] = {}
    for key, spec in collate_spec.items():
        out[str(key)] = CollateSpec.from_spec(spec).to_core_dict()
    return out


def _apply_collate_spec(
    encoding: BatchEncoding,
    collate_spec: dict[str, dict[str, Any]],
) -> None:
    """Call the private native collate hook hidden from generated stubs."""
    from .. import _core as _core_module

    setter = getattr(_core_module, "_set_batch_encoding_collate_spec")
    setter(encoding, collate_spec)


class _EncodingConversionMixin(Generic[PygDataT]):
    """Own conversion from native/dict encodings to the PyG-facing boundary."""

    def _dict_to_pyg(
        self,
        encoding_dict: EncodingDict,
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> PygDataT:
        return _encoding_dict_to_pyg(
            encoding_dict, as_batch=as_batch, include_metadata=include_metadata
        )

    def _to_pyg(
        self,
        encoding: BatchEncodingInput,
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> PygDataT:
        if isinstance(encoding, Mapping):
            return self._dict_to_pyg(
                encoding, as_batch=as_batch, include_metadata=include_metadata
            )
        if _is_batch_encoding_like(encoding):
            uses_default_converter = (
                type(self)._dict_to_pyg is _EncodingConversionMixin._dict_to_pyg
            )
            if uses_default_converter:
                return to_pyg(
                    encoding, as_batch=as_batch, include_metadata=include_metadata
                )
            return self._dict_to_pyg(
                encoding.as_dict(),
                as_batch=as_batch,
                include_metadata=include_metadata,
            )
        raise TypeError(f"Unsupported encoding type: {type(encoding)}")


class EncoderBase(_EncodingConversionMixin[PygDataT], ABC):
    """
    Base class for all non-stream encoders.

    Public contract (native-first):
    - ``encode(...)`` returns one native batch encoding object.
    - ``encode_batch(...)`` returns one native batch encoding object.
    - ``encode_pyg(...)`` / ``encode_batch_pyg(...)`` return PyG objects.
    """

    def _accepted_kwargs(self) -> set[str]:
        """Return keyword names accepted by this encoder."""
        return set()

    def _filter_kwargs(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and return keyword arguments supported by this encoder."""
        accepted = self._accepted_kwargs()
        unexpected = sorted(set(kwargs).difference(accepted))
        if unexpected:
            formatted = ", ".join(repr(name) for name in unexpected)
            plural = "s" if len(unexpected) != 1 else ""
            raise TypeError(
                f"{type(self).__name__} got unexpected keyword argument{plural}: "
                f"{formatted}"
            )
        return {key: value for key, value in kwargs.items() if key in accepted}

    def encode(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        include_metadata: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        # Native encodings are unchanged by include_metadata; conversion controls metadata.
        return self._encode(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            **self._filter_kwargs(kwargs),
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
        **kwargs,
    ) -> BatchEncoding:
        # Native encodings are unchanged by include_metadata; conversion controls metadata.
        encoding = self._encode_batch(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            **self._filter_kwargs(kwargs),
        )
        if batch_attrs:
            for key, value in batch_attrs.items():
                setattr(encoding, str(key), value)
        normalized_collate_spec = _normalize_collate_spec(collate_spec)
        if normalized_collate_spec:
            _apply_collate_spec(encoding, normalized_collate_spec)
        return encoding

    def encode_pyg(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        include_metadata: bool = True,
        **kwargs,
    ) -> PygDataT:
        encoding = self.encode(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            include_metadata=include_metadata,
            **kwargs,
        )
        return self._to_pyg(encoding, as_batch=False, include_metadata=include_metadata)

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
        **kwargs,
    ) -> PygDataT:
        encoding = self.encode_batch(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            batch_attrs=batch_attrs,
            collate_spec=collate_spec,
            include_metadata=include_metadata,
            **kwargs,
        )
        return self._to_pyg(encoding, as_batch=True, include_metadata=include_metadata)

    @abstractmethod
    def _encode(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        **kwargs,
    ) -> BatchEncoding:
        """Encode one input into native batch encoding."""
        ...

    @abstractmethod
    def _encode_batch(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchParam = None,
        actions: ActionBatchParam = None,
        subgoal_layers: SubgoalLayersBatchParam = None,
        **kwargs,
    ) -> BatchEncoding:
        """Encode one or many inputs into native batch encoding."""
        ...


class StreamEncoderBase(_EncodingConversionMixin[PygDataT], ABC):
    """Base class for stream encoders that accumulate graphs incrementally."""

    @abstractmethod
    def append(self, *args, **kwargs) -> int:
        """Append one item to the in-memory stream and return its stream id."""
        ...

    def _coerce_stream_id(self, value: Any) -> int:
        """Normalize a C++ stream id, falling back to a local counter when needed."""
        if isinstance(value, Integral):
            return int(value)
        stream_id = getattr(self, "_stream_id_counter", 0)
        self._stream_id_counter = stream_id + 1
        return stream_id

    def remove(self, _stream_id: int) -> None:
        """Remove a previously appended item from the stream."""
        raise NotImplementedError("remove is not implemented for this stream")

    def set_reuse_removed(self, value: bool) -> None:
        """Enable or disable removed-slot reuse (replace-in-order semantics)."""
        stream = getattr(self, "_stream", None)
        if stream is None or not hasattr(stream, "set_reuse_removed"):
            raise NotImplementedError(
                "set_reuse_removed is not implemented for this stream"
            )
        stream.set_reuse_removed(bool(value))

    def update(self, stream_id: int, *args, **kwargs) -> None:
        """Re-encode and replace a previously appended item in the stream."""
        raise NotImplementedError("update is not implemented for this stream")

    def flush(self) -> BatchEncoding:
        """Flush accumulated items and return native batch encoding."""
        stream = getattr(self, "_stream", None)
        if stream is not None and hasattr(stream, "flush"):
            encoding = stream.flush()
            self._reset_builder()
            return encoding

        builder = getattr(self, "_builder", None)
        if builder is not None and hasattr(builder, "build"):
            encoding = builder.build()
            self._reset_builder()
            return encoding

        raise NotImplementedError("flush is not implemented for this stream")

    def flush_pyg(
        self, *, as_batch: bool = True, include_metadata: bool = True
    ) -> PygDataT:
        encoding = self.flush()
        return self._to_pyg(
            encoding, as_batch=as_batch, include_metadata=include_metadata
        )

    @abstractmethod
    def _reset_builder(self) -> None:
        """Create/reset the internal builder used by ``append``."""
        ...
