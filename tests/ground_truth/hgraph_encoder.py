from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import pymimir
from torch_geometric.data import HeteroData

from .pyg_builder import PygHeteroBuilder


class HGraphEncoder:
    """Minimal pymimir-based HGraph encoder aligned with C++ semantics."""

    def __init__(self, domain: pymimir.Domain, symbol_type_id: str = "_symbol_") -> None:
        self.domain = domain
        self.symbol_type_id = symbol_type_id

    def encode_state(self, state: pymimir.State) -> HeteroData:
        builder = PygHeteroBuilder()
        problem = state.get_problem()

        objects = list(problem.get_objects()) + list(self.domain.get_constants())
        objects = sorted(objects, key=lambda obj: obj.get_index())
        for obj in objects:
            builder.add_node(
                str(obj.get_index()),
                self.symbol_type_id,
                x=[1.0],
                name=obj.get_name(),
            )

        facts = state.get_atoms(ignore_static=True)
        self._encode_facts(facts, builder)
        return builder.build()

    def _encode_facts(
        self, facts: Sequence[pymimir.GroundAtom], builder: PygHeteroBuilder
    ) -> None:
        pred_counts = defaultdict(int)
        for atom in facts:
            predicate = atom.get_predicate()
            pred_name = predicate.get_name()
            pred_idx = pred_counts[pred_name]
            pred_counts[pred_name] += 1

            node_key = f"{pred_name}:{pred_idx}"
            builder.add_node(node_key, pred_name, x=[1.0])

            for pos, obj in enumerate(atom.get_terms()):
                builder.add_edge(
                    node_key,
                    str(obj.get_index()),
                    pred_name,
                    self.symbol_type_id,
                    f"arg{pos}",
                )
