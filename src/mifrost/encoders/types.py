from __future__ import annotations

from collections.abc import Iterable
from typing import (
    Any,
    Callable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    TypeAlias,
    runtime_checkable,
)

import pymimir.advanced.formalism as af
import pymimir.advanced.search as ase
import pymimir.wrapper_formalism as wf

if TYPE_CHECKING:
    import torch
    from torch_geometric.data import Data, HeteroData
    from mifrost._core import HeteroBatchEncodingView, HomoBatchEncodingView

# Canonical input types supported by mifrost encoders.
DomainInput: TypeAlias = wf.Domain | af.Domain
StateInput: TypeAlias = wf.State | ase.State
GroundActionInput: TypeAlias = wf.GroundAction | af.GroundAction

AdvancedGroundLiteral: TypeAlias = (
    af.StaticGroundLiteral | af.FluentGroundLiteral | af.DerivedGroundLiteral
)
GoalLiteralInput: TypeAlias = wf.GroundLiteral | AdvancedGroundLiteral
HistorySubgoalInput: TypeAlias = Iterable[tuple[int, Iterable[GoalLiteralInput]]]

GroundAtomInput: TypeAlias = (
    wf.GroundAtom | af.StaticGroundAtom | af.FluentGroundAtom | af.DerivedGroundAtom
)
PredicateInput: TypeAlias = (
    wf.Predicate | af.StaticPredicate | af.FluentPredicate | af.DerivedPredicate
)
ObjectInput: TypeAlias = wf.Object | af.Object

STATE_TYPES = (wf.State, ase.State)
WRAPPER_STATE_TYPES = (wf.State,)
DOMAIN_TYPES = (wf.Domain, af.Domain)
GOAL_LITERAL_TYPES = (
    wf.GroundLiteral,
    af.StaticGroundLiteral,
    af.FluentGroundLiteral,
    af.DerivedGroundLiteral,
)
ADVANCED_GOAL_LITERAL_TYPES = (
    af.StaticGroundLiteral,
    af.FluentGroundLiteral,
    af.DerivedGroundLiteral,
)
GROUND_ACTION_TYPES = (wf.GroundAction, af.GroundAction)
ATOM_TYPES = (
    wf.GroundAtom,
    af.StaticGroundAtom,
    af.FluentGroundAtom,
    af.DerivedGroundAtom,
)
PREDICATE_TYPES = (
    wf.Predicate,
    af.StaticPredicate,
    af.FluentPredicate,
    af.DerivedPredicate,
)
OBJECT_TYPES = (wf.Object, af.Object)

StateAdapter = Callable[[object], ase.State]
DomainAdapter = Callable[[object], af.Domain]
LiteralAdapter = Callable[[object], AdvancedGroundLiteral]
ActionAdapter = Callable[[object], af.GroundAction]
_STATE_ADAPTERS: dict[type[object], StateAdapter] = {}
_DOMAIN_ADAPTERS: dict[type[object], DomainAdapter] = {}
_LITERAL_ADAPTERS: dict[type[object], LiteralAdapter] = {}
_ACTION_ADAPTERS: dict[type[object], ActionAdapter] = {}

EncodingDict: TypeAlias = Mapping[str, Any]
EdgeType: TypeAlias = tuple[str, str, str]


@runtime_checkable
class HeteroBatchEncodingViewLike(Protocol):
    object_names: Sequence[str]
    x_dict: Mapping[str, "torch.Tensor"]
    edge_index_dict: Mapping[EdgeType, "torch.Tensor"]
    batch_dict: Mapping[str, "torch.Tensor"]
    ptr_dict: Mapping[str, "torch.Tensor"]
    edge_attr_dict: Mapping[EdgeType, "torch.Tensor"]


@runtime_checkable
class HomoBatchEncodingViewLike(Protocol):
    object_names: Sequence[str]
    x: "torch.Tensor | None"
    edge_index: "torch.Tensor | None"
    batch: "torch.Tensor | None"
    ptr: "torch.Tensor | None"
    edge_attr: "torch.Tensor | None"


@runtime_checkable
class NativeEncoding(Protocol):
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

    def as_pyg(self, *, as_batch: bool | None = None) -> Any: ...

    def as_hetero(self) -> "HeteroBatchEncodingView": ...

    def as_homo(self) -> "HomoBatchEncodingView": ...

    def schema_fingerprint(self) -> int: ...


@runtime_checkable
class HeteroEncoding(NativeEncoding, Protocol):
    """Refined protocol for hetero native encodings."""

    graph_kind: Literal["hetero"]

    def as_pyg(self, *, as_batch: bool | None = None) -> "HeteroData": ...


@runtime_checkable
class HomoEncoding(NativeEncoding, Protocol):
    """Refined protocol for homo native encodings."""

    graph_kind: Literal["homo"]

    def as_pyg(self, *, as_batch: bool | None = None) -> "Data": ...


NativeEncodingInput: TypeAlias = NativeEncoding | EncodingDict


def register_state_adapter(state_type: type[object], adapter: StateAdapter) -> None:
    """
    Register a custom state adapter.

    The adapter must return a ``pymimir.advanced.search.State`` instance.
    Adapters are matched by exact concrete type.
    """
    _STATE_ADAPTERS[state_type] = adapter


def unregister_state_adapter(state_type: type[object]) -> None:
    """Remove a previously registered custom state adapter."""
    _STATE_ADAPTERS.pop(state_type, None)


def _resolve_state_adapter(value: object) -> StateAdapter | None:
    return _STATE_ADAPTERS.get(type(value))


def is_state_input(value: object) -> bool:
    """Return whether ``value`` is accepted as a state input."""
    return isinstance(value, STATE_TYPES) or _resolve_state_adapter(value) is not None


def register_domain_adapter(domain_type: type[object], adapter: DomainAdapter) -> None:
    """Register a custom domain adapter matched by exact concrete type."""
    _DOMAIN_ADAPTERS[domain_type] = adapter


def unregister_domain_adapter(domain_type: type[object]) -> None:
    """Remove a previously registered custom domain adapter."""
    _DOMAIN_ADAPTERS.pop(domain_type, None)


def _resolve_domain_adapter(value: object) -> DomainAdapter | None:
    return _DOMAIN_ADAPTERS.get(type(value))


def is_domain_input(value: object) -> bool:
    """Return whether ``value`` is accepted as a domain input."""
    return isinstance(value, DOMAIN_TYPES) or _resolve_domain_adapter(value) is not None


def register_literal_adapter(
    literal_type: type[object], adapter: LiteralAdapter
) -> None:
    """Register a custom goal-literal adapter matched by exact concrete type."""
    _LITERAL_ADAPTERS[literal_type] = adapter


def unregister_literal_adapter(literal_type: type[object]) -> None:
    """Remove a previously registered custom literal adapter."""
    _LITERAL_ADAPTERS.pop(literal_type, None)


def _resolve_literal_adapter(value: object) -> LiteralAdapter | None:
    return _LITERAL_ADAPTERS.get(type(value))


def is_goal_literal_input(value: object) -> bool:
    """Return whether ``value`` is accepted as a goal literal input."""
    return (
        isinstance(value, GOAL_LITERAL_TYPES)
        or _resolve_literal_adapter(value) is not None
    )


def register_action_adapter(action_type: type[object], adapter: ActionAdapter) -> None:
    """Register a custom ground-action adapter matched by exact concrete type."""
    _ACTION_ADAPTERS[action_type] = adapter


def unregister_action_adapter(action_type: type[object]) -> None:
    """Remove a previously registered custom action adapter."""
    _ACTION_ADAPTERS.pop(action_type, None)


def _resolve_action_adapter(value: object) -> ActionAdapter | None:
    return _ACTION_ADAPTERS.get(type(value))


def is_action_input(value: object) -> bool:
    """Return whether ``value`` is accepted as an action input."""
    return (
        isinstance(value, GROUND_ACTION_TYPES)
        or _resolve_action_adapter(value) is not None
    )


def to_advanced_domain(domain: object) -> af.Domain:
    if isinstance(domain, wf.Domain):
        return domain._advanced_domain
    if isinstance(domain, af.Domain):
        return domain
    adapter = _resolve_domain_adapter(domain)
    if adapter is not None:
        advanced_domain = adapter(domain)
        if not isinstance(advanced_domain, af.Domain):
            raise TypeError(
                "Domain adapter must return pymimir.advanced.formalism.Domain, "
                f"got {type(advanced_domain)!r}"
            )
        return advanced_domain
    raise TypeError(f"Unsupported domain type: {type(domain)!r}")


def to_advanced_state(state: object) -> ase.State:
    if isinstance(state, wf.State):
        return state._advanced_state
    if isinstance(state, ase.State):
        return state
    adapter = _resolve_state_adapter(state)
    if adapter is not None:
        advanced_state = adapter(state)
        if not isinstance(advanced_state, ase.State):
            raise TypeError(
                "State adapter must return pymimir.advanced.search.State, "
                f"got {type(advanced_state)!r}"
            )
        return advanced_state
    raise TypeError(f"Unsupported state type: {type(state)!r}")


def to_advanced_literal(literal: object) -> AdvancedGroundLiteral:
    if isinstance(literal, wf.GroundLiteral):
        return literal._advanced_ground_literal
    if isinstance(literal, ADVANCED_GOAL_LITERAL_TYPES):
        return literal
    adapter = _resolve_literal_adapter(literal)
    if adapter is not None:
        advanced_literal = adapter(literal)
        if not isinstance(advanced_literal, ADVANCED_GOAL_LITERAL_TYPES):
            raise TypeError(
                "Literal adapter must return one of "
                "(StaticGroundLiteral, FluentGroundLiteral, DerivedGroundLiteral), "
                f"got {type(advanced_literal)!r}"
            )
        return advanced_literal
    raise TypeError(f"Unsupported goal literal type: {type(literal)!r}")


def to_advanced_action(action: object) -> af.GroundAction:
    if isinstance(action, wf.GroundAction):
        return action._advanced_ground_action
    if isinstance(action, af.GroundAction):
        return action
    adapter = _resolve_action_adapter(action)
    if adapter is not None:
        advanced_action = adapter(action)
        if not isinstance(advanced_action, af.GroundAction):
            raise TypeError(
                "Action adapter must return pymimir.advanced.formalism.GroundAction, "
                f"got {type(advanced_action)!r}"
            )
        return advanced_action
    raise TypeError(f"Unsupported action type: {type(action)!r}")


def default_goals_from_state(state: StateInput) -> list[wf.GroundLiteral]:
    if isinstance(state, wf.State):
        return list(state.get_problem().get_goal_condition().get_literals())
    raise ValueError("goals must be provided when passing an advanced state")


def is_goal_literal_iterable(values: Iterable[object]) -> bool:
    return all(isinstance(value, GOAL_LITERAL_TYPES) for value in values)
