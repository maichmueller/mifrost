from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, Mapping, Sequence, TypeVar

from torch_geometric.data import Data, HeteroData

from .common import _parts_to_pyg

PygDataT = TypeVar("PygDataT", Data, HeteroData)


class EncoderBase(ABC, Generic[PygDataT]):
    """
    Common encoder API.

    Concrete encoders should implement encode_parts and encode_batch_parts.
    Unknown kwargs are silently ignored to keep a stable API surface across encoders.
    """

    def _accepted_kwargs(self) -> set[str]:
        return set()

    def _filter_kwargs(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        accepted = self._accepted_kwargs()
        if not accepted:
            return {}
        return {key: value for key, value in kwargs.items() if key in accepted}

    def encode(
        self,
        state: Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> PygDataT:
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
        states: Iterable[Any] | Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> PygDataT:
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
        state: Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]: ...

    @abstractmethod
    def encode_batch_parts(
        self,
        states: Iterable[Any] | Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]: ...

    def _parts_to_pyg(
        self,
        parts: Mapping[str, Any],
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> PygDataT:
        return _parts_to_pyg(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )


class StreamEncoderBase(ABC, Generic[PygDataT]):
    """
    Common stream encoder API.

    Concrete stream encoders must implement append and reset their builder after flush.
    """

    @abstractmethod
    def append(self, *args: Any, **kwargs: Any) -> None: ...

    def flush(
        self, *, as_batch: bool = True, include_metadata: bool = True
    ) -> PygDataT:
        parts = self.flush_parts()
        return self._parts_to_pyg(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )

    def flush_parts(self) -> Mapping[str, Any]:
        parts = self._builder.build_parts()
        self._reset_builder()
        return parts

    @abstractmethod
    def _reset_builder(self) -> None: ...

    @abstractmethod
    def _parts_to_pyg(
        self,
        parts: Mapping[str, Any],
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> PygDataT: ...
