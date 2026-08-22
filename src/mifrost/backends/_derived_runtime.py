"""Per-instance backend resolution for the public derived-graph encoder facades."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

if TYPE_CHECKING:
    from pytyr.formalism.planning import PlanningTask


DerivedBackendName = Literal["pymimir", "pytyr"]


class DerivedRuntime(Protocol):
    """Backend-local implementation used by one derived-graph encoder instance."""

    engine: Any

    @property
    def backend_name(self) -> str: ...

    def encode(
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

    def make_stream(self) -> Any: ...


def _normalize_backend(value: str | None) -> DerivedBackendName | None:
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


def create_derived_runtime(
    problem_or_task: object,
    config: Any,
    *,
    backend: str | None = None,
) -> DerivedRuntime:
    """Resolve one backend without changing process-global state."""

    selected = _normalize_backend(backend)
    if selected is None:
        selected = "pytyr" if _looks_like_pytyr_task(problem_or_task) else "pymimir"

    if selected == "pytyr":
        from .pytyr_derived import PyTyrDerivedRuntime

        return PyTyrDerivedRuntime(cast("PlanningTask", problem_or_task), config)

    from .pymimir_derived import PymimirDerivedRuntime

    return PymimirDerivedRuntime(problem_or_task, config)


__all__ = ["DerivedBackendName", "DerivedRuntime", "create_derived_runtime"]
