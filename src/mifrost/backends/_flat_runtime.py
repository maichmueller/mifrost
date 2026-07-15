"""Per-instance backend resolution for the public flat encoder facade."""

from __future__ import annotations

from typing import Any, Literal, Protocol


FlatBackendName = Literal["pymimir", "pytyr"]


class FlatRuntime(Protocol):
    """Backend-local implementation used by one flat encoder instance."""

    engine: Any

    @property
    def backend_name(self) -> str: ...

    @property
    def relation_dict(self) -> Any: ...

    def encode_one(
        self,
        state: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
        history_subgoals: object = None,
        history_max_steps: int | None = None,
    ) -> Any: ...

    def encode_batch(
        self,
        states: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
        history_subgoals: object = None,
        history_max_steps: int | None = None,
    ) -> Any: ...

    def append_into_builder(
        self,
        state: object,
        builder: Any,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
        history_subgoals: object = None,
        history_max_steps: int | None = None,
    ) -> None: ...

    def make_stream(self, *, mutable: bool) -> Any: ...


def _normalize_backend(value: str | None) -> FlatBackendName | None:
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


def create_flat_runtime(
    domain_or_task: object,
    config: Any,
    *,
    backend: str | None = None,
) -> FlatRuntime:
    """Resolve one backend without changing process-global state."""

    selected = _normalize_backend(backend)
    if selected is None:
        selected = "pytyr" if _looks_like_pytyr_task(domain_or_task) else "pymimir"

    if selected == "pytyr":
        from .pytyr_flat import PyTyrFlatRuntime

        return PyTyrFlatRuntime(domain_or_task, config)

    from .pymimir_flat import PymimirFlatRuntime

    return PymimirFlatRuntime(domain_or_task, config)


__all__ = ["FlatBackendName", "FlatRuntime", "create_flat_runtime"]
