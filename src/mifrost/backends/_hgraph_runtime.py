"""Per-instance backend resolution for the public HGraph encoder facade."""

from __future__ import annotations

from typing import Any, Literal, Protocol


HGraphBackendName = Literal["pymimir", "pytyr"]


class HGraphRuntime(Protocol):
    """Backend-local implementation used by one HGraph encoder instance."""

    engine: Any

    @property
    def backend_name(self) -> str: ...

    @property
    def relation_dict(self) -> Any: ...

    def encode_one(self, state: object, **kwargs: Any) -> Any: ...

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
        self, state: object, builder: Any, **kwargs: Any
    ) -> None: ...

    def update_relations(self, relation_dict: Any) -> None: ...

    def make_stream(self, *, mutable: bool) -> Any: ...


def _normalize_backend(value: str | None) -> HGraphBackendName | None:
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


def create_hgraph_runtime(
    domain_or_task: object,
    config: Any,
    *,
    backend: str | None = None,
) -> HGraphRuntime:
    """Resolve one backend without changing process-global state."""

    selected = _normalize_backend(backend)
    if selected is None:
        selected = "pytyr" if _looks_like_pytyr_task(domain_or_task) else "pymimir"

    if selected == "pytyr":
        from .pytyr_hgraph import PyTyrHGraphRuntime

        return PyTyrHGraphRuntime(domain_or_task, config)

    from .pymimir_hgraph import PymimirHGraphRuntime

    return PymimirHGraphRuntime(domain_or_task, config)


__all__ = ["HGraphBackendName", "HGraphRuntime", "create_hgraph_runtime"]
