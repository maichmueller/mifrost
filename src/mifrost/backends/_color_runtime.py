"""Per-instance backend resolution for the public Color encoder facade."""

from __future__ import annotations

from typing import Any, Literal, Protocol


ColorBackendName = Literal["pymimir", "pytyr"]


class ColorRuntime(Protocol):
    """Backend-local implementation used by one Color encoder instance."""

    engine: Any

    @property
    def backend_name(self) -> str: ...

    def encode_one(
        self,
        state: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
    ) -> Any: ...

    def encode_batch(
        self,
        states: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
    ) -> Any: ...

    def make_stream(self) -> Any: ...


def _normalize_backend(value: str | None) -> ColorBackendName | None:
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


def create_color_runtime(
    domain_or_task: object,
    config: Any,
    *,
    backend: str | None = None,
) -> ColorRuntime:
    """Resolve one backend without changing process-global state."""

    selected = _normalize_backend(backend)
    if selected is None:
        selected = "pytyr" if _looks_like_pytyr_task(domain_or_task) else "pymimir"

    if selected == "pytyr":
        from .pytyr_color import PyTyrColorRuntime

        return PyTyrColorRuntime(domain_or_task, config)

    from .pymimir_color import PymimirColorRuntime

    return PymimirColorRuntime(domain_or_task, config)


__all__ = ["ColorBackendName", "ColorRuntime", "create_color_runtime"]
