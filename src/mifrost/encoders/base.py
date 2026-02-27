from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Integral
from typing import (
    TYPE_CHECKING,
    Generic,
    Iterable,
    Mapping,
    Sequence,
    TypeAlias,
    TypeGuard,
    TypeVar,
)

from .common import _encoding_dict_to_pyg
from .types import (
    BatchParam,
    EncodingDict,
    GoalLiteralInput,
    GroundActionInput,
    HistorySubgoalInput,
    NativeEncoding,
    NativeEncodingInput,
    StateInput,
)
from .._core import BatchEncoding

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


def _is_native_encoding(value: object) -> TypeGuard[NativeEncoding]:
    return hasattr(value, "as_dict") and hasattr(value, "as_pyg")


class EncoderBase(ABC, Generic[PygDataT]):
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

    def _filter_kwargs(self, kwargs: Mapping[str, object]) -> dict[str, object]:
        """Drop kwargs that are not explicitly supported by this encoder."""
        accepted = self._accepted_kwargs()
        if not accepted:
            return {}
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
        include_metadata: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        # Native encodings are unchanged by include_metadata; conversion controls metadata.
        return self._encode_batch(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            **self._filter_kwargs(kwargs),
        )

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
        include_metadata: bool = True,
        **kwargs,
    ) -> PygDataT:
        encoding = self.encode_batch(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
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
        encoding: NativeEncodingInput,
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> PygDataT:
        if _is_native_encoding(encoding):
            uses_default_converter = type(self)._dict_to_pyg is EncoderBase._dict_to_pyg
            if include_metadata and uses_default_converter:
                return encoding.as_pyg(as_batch=as_batch)
            return self._dict_to_pyg(
                encoding.as_dict(),
                as_batch=as_batch,
                include_metadata=include_metadata,
            )
        return self._dict_to_pyg(
            encoding, as_batch=as_batch, include_metadata=include_metadata
        )


class StreamEncoderBase(ABC, Generic[PygDataT]):
    """Base class for stream encoders that accumulate graphs incrementally."""

    @abstractmethod
    def append(self, *args: object, **kwargs) -> int:
        """Append one item to the in-memory stream and return its stream id."""
        ...

    def _coerce_stream_id(self, value: object) -> int:
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

    def update(self, stream_id: int, *args: object, **kwargs) -> None:
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

    @abstractmethod
    def _dict_to_pyg(
        self,
        encoding_dict: EncodingDict,
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> PygDataT:
        """Convert dictionary encoding into PyG output for stream flush."""
        ...

    def _to_pyg(
        self,
        encoding: NativeEncodingInput,
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> PygDataT:
        if _is_native_encoding(encoding):
            uses_default_converter = (
                type(self)._dict_to_pyg is StreamEncoderBase._dict_to_pyg
            )
            if include_metadata and uses_default_converter:
                return encoding.as_pyg(as_batch=as_batch)
            return self._dict_to_pyg(
                encoding.as_dict(),
                as_batch=as_batch,
                include_metadata=include_metadata,
            )
        return self._dict_to_pyg(
            encoding, as_batch=as_batch, include_metadata=include_metadata
        )
