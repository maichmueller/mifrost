"""Per-instance backend resolution for public transition HGraph encoders."""

from __future__ import annotations

from typing import Any, Literal, Protocol


TransitionBackendName = Literal["pymimir", "pytyr"]


class TransitionRuntime(Protocol):
    """Backend-local implementation owned by one transition encoder."""

    engine: Any

    @property
    def backend_name(self) -> str: ...

    @property
    def relation_dict(self) -> Any: ...

    def encode_one(self, current: object, successor: object, **kwargs: Any) -> Any: ...

    def encode_batch(
        self,
        states: object,
        *,
        successors: object,
        goals: object = None,
        subgoal_layers: object = None,
    ) -> Any: ...

    def append_into_builder(
        self,
        current: object,
        successor: object,
        builder: Any,
        **kwargs: Any,
    ) -> None: ...

    def update_relations(self, relation_dict: Any) -> None: ...

    def make_stream(self) -> Any: ...


def _normalize_backend(value: str | None) -> TransitionBackendName | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in {"pymimir", "pytyr"}:
        raise ValueError("backend must be 'pymimir' or 'pytyr'")
    return normalized  # type: ignore[return-value]


def _looks_like_pytyr_task(value: object) -> bool:
    try:
        from pytyr.formalism.planning import PlanningTask
    except ImportError:
        return False
    return isinstance(value, PlanningTask)


def create_transition_runtime(
    domain_or_task: object,
    config: Any,
    *,
    backend: str | None = None,
) -> TransitionRuntime:
    """Resolve one transition backend without process-global state."""

    selected = _normalize_backend(backend)
    if selected is None:
        selected = "pytyr" if _looks_like_pytyr_task(domain_or_task) else "pymimir"

    if selected == "pytyr":
        from .pytyr_transition import PyTyrTransitionRuntime

        return PyTyrTransitionRuntime(domain_or_task, config)

    from .pymimir_transition import PymimirTransitionRuntime

    return PymimirTransitionRuntime(domain_or_task, config)


__all__ = [
    "TransitionBackendName",
    "TransitionRuntime",
    "create_transition_runtime",
]
