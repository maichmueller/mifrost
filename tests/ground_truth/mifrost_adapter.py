from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import torch
from torch_geometric.data import HeteroData


def mifrost_dict_to_heterodata(payload: Dict[str, object]) -> HeteroData:
    data = HeteroData()
    edge_parts: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]] = defaultdict(dict)

    for key, value in payload.items():
        if key.endswith("/ptr"):
            continue

        if "/edge_index_" in key:
            base, suffix = key.rsplit("/edge_index_", 1)
            src_type, rel, dst_type = base.split("|")
            edge_parts[(src_type, rel, dst_type)][suffix] = torch.as_tensor(
                value, dtype=torch.long
            )
            continue

        node_type, attr = key.split("/", 1)
        data[node_type][attr] = torch.as_tensor(value)

    for (src_type, rel, dst_type), parts in edge_parts.items():
        src = parts.get("0")
        dst = parts.get("1")
        if src is None or dst is None:
            raise ValueError(f"Incomplete edge_index parts for {src_type, rel, dst_type}")
        edge_index = torch.stack((src, dst), dim=0)
        data[(src_type, rel, dst_type)].edge_index = edge_index

    for node_type in data.node_types:
        if "x" not in data[node_type]:
            num_nodes = data[node_type].num_nodes
            data[node_type].x = torch.zeros((num_nodes, 1))

    return data
