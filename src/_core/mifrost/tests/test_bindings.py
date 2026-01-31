import pytest
import numpy as np
import torch
import mifrost
from torch_geometric.data import Batch


def test_batch_builder_basics():
    builder = mifrost.BatchBuilder()

    # 1. Add some data
    x = torch.randn(10, 5)
    builder.add_node_features("atom", "x", x)

    # 2. Add edges
    src = torch.tensor([0, 1, 2], dtype=torch.int64)
    dst = torch.tensor([1, 2, 0], dtype=torch.int64)
    builder.add_edges("atom", "rel", "atom", src, dst)

    # 3. Next graph
    builder.next_graph()

    # 4. Add second graph (offset logic implicit)
    builder.add_node_features("atom", "x", x)
    builder.add_edges("atom", "rel", "atom", src, dst)  # Should be offset by 10
    builder.next_graph()

    # 5. Build
    batch = builder.build()
    assert isinstance(batch, Batch)
    assert "atom" in batch.node_types

    # Verify content
    out_x = batch["atom"].x
    assert out_x.shape == (20, 5)
    assert torch.allclose(out_x[0:10], x)
    assert torch.allclose(out_x[10:20], x)

    # Verify edge offsets
    out_src = batch[("atom", "rel", "atom")].edge_index[0]
    # First graph: 0, 1, 2
    # Second graph: 10, 11, 12
    expected_src = torch.cat([src, src + 10])
    assert torch.equal(out_src, expected_src)

    ptr = batch["atom"].ptr
    assert torch.equal(ptr, torch.tensor([0, 10, 20], dtype=torch.int64))


def test_hgraph_encoder_instantiation():
    # Needs domain binding or mock
    pass
