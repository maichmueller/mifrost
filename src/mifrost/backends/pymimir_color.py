"""Pymimir runtime for the public homogeneous Color encoder."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, cast

from .._core import ColorEncoderConfig, ColorEncoderEngine, ColorStreamEncoder
from ..encoders._batch_contract import prepare_core_batch_inputs
from ..encoders._lane_specs import prepare_optional_payloads
from ..encoders.common import _advanced_domain, _advanced_state, _split_goals
from ..encoders.types import WRAPPER_STATE_TYPES, default_goals_from_state


_PYMIMIR_WRAPPER_STATE_TYPE = WRAPPER_STATE_TYPES[0]


def _unwrap_homogeneous_wrapper_state_list(states: object) -> list[Any] | None:
    """Unwrap the common Pymimir batch shape in one allocation/pass.

    The generic batch contract remains authoritative for every other shape.
    Exact type checks are intentional: list subclasses, mixed/native values,
    and custom registered adapters retain their established normalization and
    error behavior.
    """

    if type(states) is not list:
        return None
    unwrapped: list[Any] = []
    for state in cast(list[Any], states):
        if type(state) is not _PYMIMIR_WRAPPER_STATE_TYPE:
            return None
        unwrapped.append(state._advanced_state)
    return unwrapped


class PymimirColorRuntime:
    """Preserve the historical native Pymimir Color fast path."""

    backend_name: Literal["pymimir"] = "pymimir"

    def __init__(self, domain: object, config: Any) -> None:
        native_config = ColorEncoderConfig(
            edge_features=bool(config.edge_features),
            enable_global_predicate_nodes=bool(config.enable_global_predicate_nodes),
        )
        self.engine = ColorEncoderEngine(_advanced_domain(domain), native_config)

    @staticmethod
    def _single_payload(
        state: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
    ) -> tuple[Any, Any | None, list[Any]]:
        action_values = prepare_optional_payloads(
            actions=cast(Iterable[Any] | None, actions), history_subgoals=None
        ).actions
        if goals is None and subgoal_layers is None and not action_values:
            return _advanced_state(state), None, []
        goal_values = goals if goals is not None else default_goals_from_state(state)
        layers = (
            None
            if subgoal_layers is None
            else list(cast(Iterable[Iterable[Any]], subgoal_layers))
        )
        return (
            _advanced_state(state),
            _split_goals(cast(Iterable[Any], goal_values), layers),
            action_values,
        )

    def encode_one(self, state: object, **kwargs: Any) -> Any:
        native_state, goals, actions = self._single_payload(state, **kwargs)
        if goals is None:
            return self.engine.encode(native_state)
        return self.engine.encode(native_state, goals, actions)

    def encode_batch(
        self,
        states: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
    ) -> Any:
        if goals is None and actions is None and subgoal_layers is None:
            native_states = _unwrap_homogeneous_wrapper_state_list(states)
            if native_states is not None:
                return self.engine.encode_batch(native_states)

        inputs = prepare_core_batch_inputs(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
        )
        return self.engine.encode_batch(
            inputs.states,
            goals=inputs.goals,
            actions=inputs.actions,
            subgoal_layers=inputs.subgoal_layers,
        )

    def make_stream(self) -> Any:
        return _PymimirColorStream(self, ColorStreamEncoder(self.engine))


class _PymimirColorStream:
    def __init__(self, runtime: PymimirColorRuntime, native: Any) -> None:
        self._runtime = runtime
        self._native = native

    def append(
        self,
        state: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
    ) -> Any:
        if goals is None and actions is None and subgoal_layers is None:
            return self._native.append(_advanced_state(state))
        native_state, goals, actions = self._runtime._single_payload(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
        )
        if actions:
            raise ValueError("ColorEncoderEngine does not support action encoding")
        if goals is None:
            return self._native.append(native_state)
        return self._native.append(native_state, goals)

    def update(
        self,
        stream_id: int,
        state: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
    ) -> None:
        if goals is None and actions is None and subgoal_layers is None:
            self._native.update(stream_id, _advanced_state(state))
            return
        native_state, goals, actions = self._runtime._single_payload(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
        )
        if actions:
            raise ValueError("ColorEncoderEngine does not support action encoding")
        if goals is None:
            self._native.update(stream_id, native_state)
        else:
            self._native.update(stream_id, native_state, goals)

    def remove(self, stream_id: int) -> None:
        self._native.remove(stream_id)

    def set_reuse_removed(self, value: bool) -> None:
        self._native.set_reuse_removed(bool(value))

    def flush(self) -> Any:
        return self._native.flush()

    def reset(self) -> None:
        self._native.reset()


__all__ = ["PymimirColorRuntime"]
