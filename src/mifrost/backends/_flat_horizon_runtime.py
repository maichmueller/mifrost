"""Per-instance backend resolution for the public Flat Horizon encoder."""

from __future__ import annotations

from typing import Any, Literal, Protocol


FlatHorizonBackendName = Literal["pymimir", "pytyr"]


class FlatHorizonRuntime(Protocol):
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

    def make_stream(self) -> Any: ...


def _normalize_backend(value: str | None) -> FlatHorizonBackendName | None:
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


def create_flat_horizon_runtime(
    domain_or_task: object,
    config: Any,
    *,
    backend: str | None = None,
    assembly: Any = None,
) -> FlatHorizonRuntime:
    selected = _normalize_backend(backend)
    if selected is None:
        selected = "pytyr" if _looks_like_pytyr_task(domain_or_task) else "pymimir"
    if selected == "pytyr":
        if assembly is not None:
            raise ValueError(
                "Flat Horizon assemblies are not supported by the PyTyr backend"
            )
        from .pytyr_flat_horizon import PyTyrFlatHorizonRuntime

        return PyTyrFlatHorizonRuntime(domain_or_task, config)
    from .pymimir_flat_horizon import PymimirFlatHorizonRuntime

    return PymimirFlatHorizonRuntime(domain_or_task, config, assembly=assembly)


__all__ = [
    "FlatHorizonBackendName",
    "FlatHorizonRuntime",
    "create_flat_horizon_runtime",
]
