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

    Public contract:
    - ``encode(...)`` returns one PyG object (``Data`` or ``HeteroData``).
    - ``encode_batch(...)`` returns a PyG batch object over multiple inputs.
    - ``encode_parts(...)`` and ``encode_batch_parts(...)`` return raw parts produced
      by C++ engines or Python encoders and consumed by ``_parts_to_pyg``.

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
        **kwargs: object,
    ) -> PygDataT:
        """
        Encode one input into a PyG object.

        Parameters mirror the common encoder API. Concrete encoders decide which
        optional inputs (goals/actions/layers/custom kwargs) are meaningful.
        """
        parts = self.encode_parts(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            **self._filter_kwargs(kwargs),
        )
        return self._parts_to_pyg(
            parts, as_batch=False, include_metadata=include_metadata
        )

    def encode_batch(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        include_metadata: bool = True,
        **kwargs: object,
    ) -> PygDataT:
        """
        Encode one or many inputs into a PyG batch object.

        ``states`` may be a single state-like object or an iterable of states.
        """
        parts = self.encode_batch_parts(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            **self._filter_kwargs(kwargs),
        )
        return self._parts_to_pyg(
            parts, as_batch=True, include_metadata=include_metadata
        )

    @abstractmethod
    def encode_parts(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        **kwargs: object,
    ) -> Mapping[str, object]:
        """Encode one input into the normalized parts payload."""
        ...

    @abstractmethod
    def encode_batch_parts(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        **kwargs: object,
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
        """Convert normalized parts into PyG objects."""
        return _parts_to_pyg(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )


class StreamEncoderBase(ABC, Generic[PygDataT]):
    """
    Base class for stream encoders that accumulate graphs incrementally.

    Stream encoders append inputs into an internal ``BatchBuilder`` and expose:
    - ``flush_parts()`` for raw parts
    - ``flush()`` for immediate PyG conversion
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

    def flush(
        self, *, as_batch: bool = True, include_metadata: bool = True
    ) -> PygDataT:
        """Flush accumulated items and return PyG output."""
        parts = self.flush_parts()
        return self._parts_to_pyg(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )

    def flush_parts(self) -> Mapping[str, object]:
        """Flush accumulated items and return normalized parts."""
        parts = self._flush_parts_impl()
        self._reset_builder()
        return parts

    @abstractmethod
    def _reset_builder(self) -> None:
        """Create/reset the internal builder used by ``append``."""
        ...

    @abstractmethod
    def _flush_parts_impl(self) -> Mapping[str, object]:
        """Return normalized parts for the current stream contents."""
        ...

    @abstractmethod
    def _parts_to_pyg(
        self,
        parts: Mapping[str, object],
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> PygDataT:
        """Convert normalized parts into PyG output for stream flush."""
        ...
