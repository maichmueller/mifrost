from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Integral
from typing import Generic, Iterable, Mapping, TypeAlias, TypeVar

from torch_geometric.data import Data, HeteroData

from .common import _parts_to_pyg
from .types import GoalLiteralInput, GroundActionInput, StateInput

PygDataT = TypeVar("PygDataT", Data, HeteroData)
StateBatchInput: TypeAlias = Iterable[StateInput] | StateInput
GoalBatchInput: TypeAlias = Iterable[GoalLiteralInput] | None
SubgoalLayersInput: TypeAlias = Iterable[Iterable[GoalLiteralInput]] | None
ActionBatchInput: TypeAlias = Iterable[GroundActionInput] | None


class EncoderBase(ABC, Generic[PygDataT]):
    """
    Base class for all non-stream encoders.

    Public contract (native-first):
    - ``encode(...)`` returns one native batch encoding object.
    - ``encode_batch(...)`` returns one native batch encoding object.
    - ``encode_pyg(...)`` / ``encode_batch_pyg(...)`` return PyG objects.
    - ``export_encoding(...)`` / ``export_batch_encoding(...)`` return normalized encoding payloads.

    Unknown keyword arguments are ignored by default. Concrete encoders opt into
    accepted keywords by ``encode`` via ``_accepted_kwargs``.
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
    ) -> object:
        """
        Encode one input into native batch encoding.

        Parameters mirror the common encoder API. Concrete encoders decide which
        optional inputs (goals/actions/layers/custom kwargs) are meaningful.
        """
        return self.export_encoding(
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
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        include_metadata: bool = True,
        **kwargs,
    ) -> object:
        """
        Encode one or many inputs into native batch encoding.

        ``states`` may be a single state-like object or an iterable of states.
        """
        return self.export_batch_encoding(
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
        return self._encoding_to_pyg(
            encoding, as_batch=False, include_metadata=include_metadata
        )

    def encode_batch_pyg(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
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
        return self._encoding_to_pyg(
            encoding, as_batch=True, include_metadata=include_metadata
        )

    def export_encoding(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        **kwargs,
    ) -> Mapping[str, object] | object:
        return self.encode_parts(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            **kwargs,
        )

    def export_batch_encoding(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        **kwargs,
    ) -> Mapping[str, object] | object:
        return self.encode_batch_parts(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            **kwargs,
        )

    @abstractmethod
    def encode_parts(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        **kwargs,
    ) -> Mapping[str, object]:
        """Encode one input into the normalized batch encoding payload."""
        ...

    @abstractmethod
    def encode_batch_parts(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        **kwargs,
    ) -> Mapping[str, object]:
        """Encode one or many inputs into a batch-oriented parts payload."""
        ...

    def _parts_to_pyg(
        self,
        parts: Mapping[str, object],
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> PygDataT:
        """Convert normalized batch encoding into PyG objects."""
        return _parts_to_pyg(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )

    def _encoding_to_pyg(
        self,
        encoding: Mapping[str, object] | object,
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> PygDataT:
        if hasattr(encoding, "as_pyg"):
            if include_metadata:
                return encoding.as_pyg(as_batch=as_batch)
            if hasattr(encoding, "to_parts"):
                parts = encoding.to_parts()
                return self._parts_to_pyg(
                    parts, as_batch=as_batch, include_metadata=include_metadata
                )
            return encoding.as_pyg(as_batch=as_batch)
        return self._parts_to_pyg(
            encoding, as_batch=as_batch, include_metadata=include_metadata
        )


class StreamEncoderBase(ABC, Generic[PygDataT]):
    """
    Base class for stream encoders that accumulate graphs incrementally.

    Stream encoders append inputs into an internal encoder and expose:
    - ``flush()`` for native encoding
    - ``flush_pyg()`` for immediate PyG conversion
    """

    @abstractmethod
    def append(self, *args: object, **kwargs: object) -> int:
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

    def update(self, _stream_id: int, *args: object, **kwargs: object) -> None:
        """Re-encode and replace a previously appended item in the stream."""
        raise NotImplementedError("update is not implemented for this stream")

    def flush(self) -> Mapping[str, object] | object:
        """Flush accumulated items and return native batch encoding."""
        stream = getattr(self, "_stream", None)
        if stream is not None and hasattr(stream, "flush"):
            encoding = stream.flush()
            self._reset_builder()
            return encoding

        builder = getattr(self, "_builder", None)
        if builder is not None and hasattr(builder, "build_batch_encoding"):
            encoding = builder.build_batch_encoding()
            self._reset_builder()
            return encoding

        raise NotImplementedError("flush is not implemented for this stream")

    def flush_pyg(
        self, *, as_batch: bool = True, include_metadata: bool = True
    ) -> PygDataT:
        encoding = self.flush()
        return self._encoding_to_pyg(
            encoding, as_batch=as_batch, include_metadata=include_metadata
        )

    def flush_batch_encoding_py(self) -> Mapping[str, object]:
        """Flush accumulated items and return normalized batch encoding parts."""
        parts = self._flush_batch_encoding_py_impl()
        self._reset_builder()
        return parts

    @abstractmethod
    def _reset_builder(self) -> None:
        """Create/reset the internal builder used by ``append``."""
        ...

    @abstractmethod
    def _flush_batch_encoding_py_impl(self) -> Mapping[str, object]:
        """Return normalized batch encoding for the current stream contents."""
        ...

    @abstractmethod
    def _parts_to_pyg(
        self,
        parts: Mapping[str, object],
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> PygDataT:
        """Convert normalized batch encoding into PyG output for stream flush."""
        ...

    def _encoding_to_pyg(
        self,
        encoding: Mapping[str, object] | object,
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> PygDataT:
        if hasattr(encoding, "as_pyg"):
            if include_metadata:
                return encoding.as_pyg(consume=True, as_batch=as_batch)
            if hasattr(encoding, "to_parts"):
                return self._parts_to_pyg(
                    encoding.to_parts(),
                    as_batch=as_batch,
                    include_metadata=include_metadata,
                )
            return encoding.as_pyg(consume=True, as_batch=as_batch)
        return self._parts_to_pyg(
            encoding, as_batch=as_batch, include_metadata=include_metadata
        )
