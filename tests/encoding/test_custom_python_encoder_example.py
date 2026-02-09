from __future__ import annotations

import torch

from mifrost.encoders import ExampleConstantEncoder


def test_example_constant_encoder_single_and_batch() -> None:
    encoder = ExampleConstantEncoder(default_value=2.0)

    single = encoder.encode_pyg("s0")
    assert "node" in single.node_types
    assert torch.equal(single["node"].x, torch.tensor([[2.0]], dtype=torch.float32))
    assert list(single["node"].node_names) == ["s0"]

    batch = encoder.encode_batch_pyg(["a", "b", "c"], value=3.5)
    assert batch["node"].x.shape[0] == 3
    assert torch.equal(
        batch["node"].x,
        torch.tensor([[3.5], [3.5], [3.5]], dtype=torch.float32),
    )
    assert torch.equal(batch["node"].ptr, torch.tensor([0, 1, 2, 3], dtype=torch.int64))


def test_example_constant_encoder_stream() -> None:
    stream = ExampleConstantEncoder(default_value=1.5).stream()
    stream.append("x")
    stream.append("y", value=4.0)
    out = stream.flush_pyg(as_batch=True)

    assert torch.equal(out["node"].x, torch.tensor([[1.5], [4.0]], dtype=torch.float32))
    assert torch.equal(out["node"].ptr, torch.tensor([0, 1, 2], dtype=torch.int64))
