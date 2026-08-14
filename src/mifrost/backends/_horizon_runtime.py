"""Per-instance backend resolution for the public Horizon encoder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

if TYPE_CHECKING:
    from pytyr.formalism.planning import PlanningTask


HorizonBackendName = Literal["pymimir", "pytyr"]


class HorizonRuntime(Protocol):
    """Backend-local implementation owned by one Horizon encoder."""

    engine: Any

    @property
    def backend_name(self) -> str: ...

    @property
    def relation_dict(self) -> Any: ...

    def encode(self, root: object, dag: object = None, **kwargs: Any) -> Any: ...

    def encode_batch(
        self,
        roots: object,
        *,
        dags: object = None,
        goals: object = None,
        subgoal_layers: object = None,
    ) -> Any: ...

    def append_into_builder(
        self,
        root: object,
        builder: Any,
        *,
        dag: object = None,
        **kwargs: Any,
    ) -> None: ...

    def update_relations(self, relation_dict: Any) -> None: ...

    def make_stream(self) -> Any: ...


def _normalize_backend(value: str | None) -> HorizonBackendName | None:
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


def create_horizon_runtime(
    domain_or_task: object,
    config: Any,
    *,
    backend: str | None = None,
) -> HorizonRuntime:
    """Resolve one Horizon backend without process-global state."""

    selected = _normalize_backend(backend)
    if selected is None:
        selected = "pytyr" if _looks_like_pytyr_task(domain_or_task) else "pymimir"

    if selected == "pytyr":
        from .pytyr_horizon import PyTyrHorizonRuntime

        return PyTyrHorizonRuntime(cast("PlanningTask", domain_or_task), config)

    from .pymimir_horizon import PymimirHorizonRuntime

    return PymimirHorizonRuntime(domain_or_task, config)


__all__ = ["HorizonBackendName", "HorizonRuntime", "create_horizon_runtime"]
