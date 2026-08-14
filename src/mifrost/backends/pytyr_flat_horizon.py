"""PyTyr runtime for the public Flat Horizon encoder."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from pytyr.formalism.planning import PlanningTask

from .pytyr import SemanticPlanningTaskAdapter
from .pytyr_horizon import PyTyrHorizonRuntime


class PyTyrFlatHorizonRuntime(PyTyrHorizonRuntime):
    """Reuse owned PyTyr DAG conversion with the neutral packed-flat engine."""

    def __init__(self, planning_task: PlanningTask, config: Any) -> None:
        self._adapter = SemanticPlanningTaskAdapter(planning_task)
        self.engine = self._adapter.make_flat_horizon_engine(config)
        self._relation_dict = MappingProxyType(
            dict(
                zip(
                    self.engine.relation_names,
                    self.engine.relation_arities,
                    strict=True,
                )
            )
        )


__all__ = ["PyTyrFlatHorizonRuntime"]
