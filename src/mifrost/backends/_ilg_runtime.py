"""Planner-neutral inputs and per-instance backend resolution for ILG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


ILGBackendName = Literal["pymimir", "pytyr"]


@dataclass(frozen=True, slots=True)
class ILGAtom:
    predicate: str
    arguments: tuple[str, ...]
    display_name: str

    @property
    def signature(self) -> tuple[str, tuple[str, ...]]:
        return self.predicate, self.arguments


@dataclass(frozen=True, slots=True)
class ILGLiteral:
    atom: ILGAtom
    positive: bool = True


@dataclass(frozen=True, slots=True)
class ILGAction:
    name: str
    arguments: tuple[str, ...]
    display_name: str


@dataclass(frozen=True, slots=True)
class ILGInput:
    objects: tuple[str, ...]
    facts: tuple[ILGAtom, ...]
    goals: tuple[ILGLiteral, ...]
    actions: tuple[ILGAction, ...]
    subgoal_layers: tuple[tuple[ILGLiteral, ...], ...]


class ILGRuntime(Protocol):
    predicate_arities: dict[str, int]
    action_feature_dim: int

    @property
    def backend_name(self) -> str: ...

    def make_input(
        self,
        state: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
    ) -> ILGInput: ...

    def make_batch_inputs(
        self,
        states: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
    ) -> list[ILGInput]: ...


def _normalize_backend(value: str | None) -> ILGBackendName | None:
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


def create_ilg_runtime(
    domain_or_task: object,
    *,
    backend: str | None = None,
) -> ILGRuntime:
    selected = _normalize_backend(backend)
    if selected is None:
        selected = "pytyr" if _looks_like_pytyr_task(domain_or_task) else "pymimir"
    if selected == "pytyr":
        from .pytyr_ilg import PyTyrILGRuntime

        return PyTyrILGRuntime(domain_or_task)
    from .pymimir_ilg import PymimirILGRuntime

    return PymimirILGRuntime(domain_or_task)


def semantic_display(name: str, arguments: tuple[str, ...]) -> str:
    suffix = "" if not arguments else f" {' '.join(arguments)}"
    return f"({name}{suffix})"


__all__ = [
    "ILGAction",
    "ILGAtom",
    "ILGBackendName",
    "ILGInput",
    "ILGLiteral",
    "ILGRuntime",
    "create_ilg_runtime",
    "semantic_display",
]
