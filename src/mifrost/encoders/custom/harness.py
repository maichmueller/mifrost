"""Ready-made verification harnesses for custom encoders.

These helpers package the parity and compatibility checks that the
built-in families ship as tests, so a third-party encoder author can run
the same guarantees in three lines instead of copying assertion blocks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch


def assert_backend_parity(
    make_encoder: Callable[[Any], Any],
    cases: Iterable[tuple[Any, Iterable[Any]]],
) -> None:
    """Assert byte-identical encodings across backends for the same PDDL.

    ``cases`` yields ``(source, states)`` pairs — one per backend (pymimir
    problem + pymimir states, pytyr task + the corresponding pytyr states;
    states are backend-native and not interchangeable). ``make_encoder``
    builds the encoder from a source. Encodings are compared via their
    canonical ``dumps()`` serialization, so any tensor or metadata
    difference fails the assertion.
    """

    dumps_per_backend: dict[str, list[bytes]] = {}
    for source, states in cases:
        encoder = make_encoder(source)
        backend = getattr(encoder, "backend", "unknown")
        dumps_per_backend[backend] = [encoder.encode(state).dumps() for state in states]
    reference = None
    for backend, dumps in sorted(dumps_per_backend.items()):
        if reference is None:
            reference = dumps
            continue
        for index, (left, right) in enumerate(zip(reference, dumps, strict=True)):
            if left != right:
                raise AssertionError(
                    f"backend parity failure for {backend!r} on state {index}"
                )


def channel_summary(data: Any) -> dict[str, Any]:
    """Return a small human-readable summary of one derived-style graph."""

    summary: dict[str, Any] = {}
    x_ids = getattr(data, "x_ids", None)
    if torch.is_tensor(x_ids):
        summary["nodes"] = int(x_ids.size(0))
        summary["channels"] = int(x_ids.size(1))
        summary["roles"] = sorted({int(value) for value in x_ids[:, 0].tolist()})
    edge_index = getattr(data, "edge_index", None)
    if torch.is_tensor(edge_index):
        summary["edges"] = int(edge_index.size(1))
    edge_attr = getattr(data, "edge_attr", None)
    if torch.is_tensor(edge_attr) and edge_attr.numel() > 0:
        summary["edge_kinds"] = sorted(
            {int(value) for value in edge_attr[:, 0].tolist()}
        )
    for extra in ("hyperedge_index", "tuple_args", "spd_src"):
        tensor = getattr(data, extra, None)
        if torch.is_tensor(tensor):
            summary[extra] = (
                int(tensor.shape[-1]) if tensor.dim() else int(tensor.numel())
            )
    return summary


def conformance_smoke(
    data: Any,
    *,
    hidden_channels: int = 8,
    with_edge_attr: bool = True,
) -> dict[str, Any]:
    """Run GCNConv and GATv2Conv forward+backward on one encoded graph.

    Skips cleanly (returns ``{"skipped": reason}``) when torch_geometric is
    unavailable; raises AssertionError on non-finite outputs or missing
    gradients, mirroring the repository conformance suite.
    """

    try:
        import torch_geometric  # noqa: F401
        from torch_geometric.nn import GATv2Conv, GCNConv
    except ImportError as exc:
        return {"skipped": f"torch_geometric unavailable: {exc}"}

    torch = _torch()
    torch.manual_seed(0)
    x = data.x_ids.float() if hasattr(data, "x_ids") else data.x.float()
    results: dict[str, Any] = {}
    edge_attr = (
        data.edge_attr.float()
        if with_edge_attr and getattr(data, "edge_attr", None) is not None
        else None
    )
    for name, layer in (
        ("gcn", GCNConv(x.size(1), hidden_channels)),
        (
            "gatv2",
            GATv2Conv(x.size(1), hidden_channels, edge_dim=edge_attr.size(1))
            if edge_attr is not None
            else GATv2Conv(x.size(1), hidden_channels),
        ),
    ):
        out = (
            layer(x, data.edge_index, edge_attr)
            if edge_attr is not None and name == "gatv2"
            else layer(x, data.edge_index)
        )
        loss = out.square().mean()
        loss.backward()
        grads = [p.grad for p in layer.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise AssertionError(f"{name} backward produced no finite gradients")
        if not torch.isfinite(out).all():
            raise AssertionError(f"{name} produced non-finite outputs")
        results[name] = int(out.size(-1))
    return results


def _torch() -> Any:
    import torch

    return torch


__all__ = ["assert_backend_parity", "channel_summary", "conformance_smoke"]
