from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    import torch
    from torch_geometric.data import Data, HeteroData
    from .._core import BatchEncoding
    from mifrost._core import HeteroBatchEncodingView, HomoBatchEncodingView

    PygDataLike: TypeAlias = Data | HeteroData
else:
    from .._core import BatchEncoding

    PygDataLike: TypeAlias = Any

# Backend-neutral public inputs. Concrete planner types are validated by the
# selected runtime rather than imported into this shared facade module.
DomainInput: TypeAlias = object
StateInput: TypeAlias = object
GroundActionInput: TypeAlias = object
AdvancedGroundLiteral: TypeAlias = object
GoalLiteralInput: TypeAlias = object
HistorySubgoalInput: TypeAlias = Iterable[tuple[int, Iterable[GoalLiteralInput]]]
GroundAtomInput: TypeAlias = object
PredicateInput: TypeAlias = object
ObjectInput: TypeAlias = object

_BatchT = TypeVar("_BatchT")


@dataclass(frozen=True)
class BatchParam(Generic[_BatchT]):
    """Explicit shared/separate wrapper for batch inputs."""

    kind: Literal["shared", "separate", "none"]
    value: Any | None = None

    @classmethod
    def shared(cls, value: _BatchT) -> "BatchParam[_BatchT]":
        return cls(kind="shared", value=value)

    @classmethod
    def separate(cls, values: Iterable[_BatchT | None]) -> "BatchParam[_BatchT]":
        return cls(kind="separate", value=list(values))

    @classmethod
    def none(cls) -> "BatchParam[_BatchT]":
        return cls(kind="none", value=None)


EncodingDict: TypeAlias = Mapping[str, Any]
EdgeType: TypeAlias = tuple[str, str, str]


@runtime_checkable
class HeteroBatchEncodingViewLike(Protocol):
    object_names: Sequence[str]
    base: "BatchEncodingLike"
    x_dict: Mapping[str, "torch.Tensor"]
    edge_index_dict: Mapping[EdgeType, "torch.Tensor"]
    batch_dict: Mapping[str, "torch.Tensor"]
    ptr_dict: Mapping[str, "torch.Tensor"]
    edge_attr_dict: Mapping[EdgeType, "torch.Tensor"]

    def to(self, device: Any) -> "HeteroBatchEncodingViewLike": ...


@runtime_checkable
class HomoBatchEncodingViewLike(Protocol):
    object_names: Sequence[str]
    base: "BatchEncodingLike"
    x: "torch.Tensor | None"
    edge_index: "torch.Tensor | None"
    batch: "torch.Tensor | None"
    ptr: "torch.Tensor | None"
    edge_attr: "torch.Tensor | None"

    def to(self, device: Any) -> "HomoBatchEncodingViewLike": ...


@runtime_checkable
class BatchEncodingLike(Protocol):
    """
    Structural type for native C++ ``BatchEncoding``-like objects.

    Concrete bindings expose this shape from C++ while Python code can
    type against the protocol without importing binding internals.
    The protocol mirrors PyG-level structural metadata (graph/node/edge
    counts and type lists) but intentionally does not model dynamic node
    storage attributes like ``data["atom"].x``.
    """

    num_graphs: int
    num_nodes: int
    num_edges: int
    graph_kind: str
    node_types: Sequence[str]
    edge_types: Sequence[tuple[str, str, str]]

    def as_dict(self) -> EncodingDict: ...

    def as_pyg(self, *, as_batch: bool | None = None) -> PygDataLike: ...

    def as_hetero(self) -> "HeteroBatchEncodingView": ...

    def as_homo(self) -> "HomoBatchEncodingView": ...

    def to(self, device: Any) -> "BatchEncodingLike": ...

    def schema_fingerprint(self) -> int: ...


@runtime_checkable
class HeteroEncoding(BatchEncodingLike, Protocol):
    """Refined protocol for hetero native encodings."""

    graph_kind: Literal["hetero"]

    def as_pyg(self, *, as_batch: bool | None = None) -> PygDataLike: ...


@runtime_checkable
class HomoEncoding(BatchEncodingLike, Protocol):
    """Refined protocol for homo native encodings."""

    graph_kind: Literal["homo"]

    def as_pyg(self, *, as_batch: bool | None = None) -> PygDataLike: ...


@runtime_checkable
class FlatEncoding(BatchEncodingLike, Protocol):
    """
    Refined protocol for flat native encodings.

    Flat encodings use `graph_kind == "flat"` even though their tensor layout is
    still homo-shaped and converts through the flat PyG carrier.
    """

    graph_kind: Literal["flat"]

    def as_pyg(self, *, as_batch: bool | None = None) -> PygDataLike: ...


BatchEncodingInput: TypeAlias = BatchEncoding | BatchEncodingLike | EncodingDict


def _pymimir_types():
    from ..backends import pymimir_types

    return pymimir_types


def register_state_adapter(state_type: type[object], adapter: Any) -> None:
    """Register a legacy exact-type Pymimir state adapter."""
    _pymimir_types().register_state_adapter(state_type, adapter)


def unregister_state_adapter(state_type: type[object]) -> None:
    _pymimir_types().unregister_state_adapter(state_type)


def _resolve_state_adapter(value: object) -> Any:
    return _pymimir_types()._resolve_state_adapter(value)


def is_state_input(value: object) -> bool:
    return _pymimir_types().is_state_input(value)


def register_domain_adapter(domain_type: type[object], adapter: Any) -> None:
    """Register a legacy exact-type Pymimir domain adapter."""
    _pymimir_types().register_domain_adapter(domain_type, adapter)


def unregister_domain_adapter(domain_type: type[object]) -> None:
    _pymimir_types().unregister_domain_adapter(domain_type)


def _resolve_domain_adapter(value: object) -> Any:
    return _pymimir_types()._resolve_domain_adapter(value)


def is_domain_input(value: object) -> bool:
    return _pymimir_types().is_domain_input(value)


def register_literal_adapter(literal_type: type[object], adapter: Any) -> None:
    """Register a legacy exact-type Pymimir literal adapter."""
    _pymimir_types().register_literal_adapter(literal_type, adapter)


def unregister_literal_adapter(literal_type: type[object]) -> None:
    _pymimir_types().unregister_literal_adapter(literal_type)


def _resolve_literal_adapter(value: object) -> Any:
    return _pymimir_types()._resolve_literal_adapter(value)


def is_goal_literal_input(value: object) -> bool:
    return _pymimir_types().is_goal_literal_input(value)


def register_action_adapter(action_type: type[object], adapter: Any) -> None:
    """Register a legacy exact-type Pymimir action adapter."""
    _pymimir_types().register_action_adapter(action_type, adapter)


def unregister_action_adapter(action_type: type[object]) -> None:
    _pymimir_types().unregister_action_adapter(action_type)


def _resolve_action_adapter(value: object) -> Any:
    return _pymimir_types()._resolve_action_adapter(value)


def is_action_input(value: object) -> bool:
    return _pymimir_types().is_action_input(value)


def to_advanced_domain(domain: object) -> Any:
    return _pymimir_types().to_advanced_domain(domain)


def to_advanced_state(state: object) -> Any:
    return _pymimir_types().to_advanced_state(state)


def to_advanced_literal(literal: object) -> Any:
    return _pymimir_types().to_advanced_literal(literal)


def to_advanced_action(action: object) -> Any:
    return _pymimir_types().to_advanced_action(action)


def default_goals_from_state(state: StateInput) -> list[Any]:
    return _pymimir_types().default_goals_from_state(state)


def is_goal_literal_iterable(values: Iterable[object]) -> bool:
    return _pymimir_types().is_goal_literal_iterable(values)
