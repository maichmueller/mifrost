from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from torch_geometric.data import HeteroData


class PygHeteroBuilder:
    """Minimal hetero builder for parity tests."""

    def __init__(self) -> None:
        self.node_keys: Dict[str, List[str]] = defaultdict(list)
        self.node_indices: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.node_attrs: Dict[str, Dict[str, List[Any]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.edge_indices: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = (
            defaultdict(list)
        )

    def add_node(self, node_key: str, node_type: str, **attrs: Any) -> int:
        if node_key in self.node_indices[node_type]:
            idx = self.node_indices[node_type][node_key]
            for attr, value in attrs.items():
                values = self.node_attrs[node_type][attr]
                while len(values) <= idx:
                    values.append(None)
                values[idx] = value
            return idx

        idx = len(self.node_keys[node_type])
        self.node_indices[node_type][node_key] = idx
        self.node_keys[node_type].append(node_key)
        for attr, value in attrs.items():
            self.node_attrs[node_type][attr].append(value)
        return idx

    def add_edge(
        self,
        src_key: str,
        dst_key: str,
        src_type: str,
        dst_type: str,
        edge_type: str,
    ) -> None:
        src_idx = self.node_indices[src_type][src_key]
        dst_idx = self.node_indices[dst_type][dst_key]
        self.edge_indices[(src_type, edge_type, dst_type)].append((src_idx, dst_idx))

    def build(self) -> HeteroData:
        data = HeteroData()
        for node_type, keys in self.node_keys.items():
            data[node_type].node_names = list(keys)
            for attr, values in self.node_attrs[node_type].items():
                if any(value is None or isinstance(value, str) for value in values):
                    data[node_type][attr] = list(values)
                else:
                    data[node_type][attr] = torch.as_tensor(values)
            if "x" not in data[node_type]:
                data[node_type].x = torch.zeros((len(keys), 1))

        for (src_type, edge_type, dst_type), indices in self.edge_indices.items():
            if indices:
                edge_index = torch.tensor(indices, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            data[(src_type, edge_type, dst_type)].edge_index = edge_index

        return data
