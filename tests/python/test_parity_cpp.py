import torch
import mifrost
from tests.ground_truth.pyencoding_ref.pyg_batch_builder import HGraphBatchBuilder
from tests.ground_truth.pyencoding_ref.pyg_builder import PygHeteroBuilder


def _compare_batches(py_batch, cpp_batch) -> None:
    assert set(py_batch.node_types) == set(cpp_batch.node_types)
    for node_type in py_batch.node_types:
        assert torch.equal(py_batch[node_type].x, cpp_batch[node_type].x)
        if hasattr(py_batch[node_type], "node_names"):
            assert list(py_batch[node_type].node_names) == list(
                cpp_batch[node_type].node_names
            )

    assert list(py_batch.object_names) == list(cpp_batch.object_names)

    assert set(py_batch.edge_types) == set(cpp_batch.edge_types)
    for edge_type in py_batch.edge_types:
        assert torch.equal(
            py_batch[edge_type].edge_index, cpp_batch[edge_type].edge_index
        )


def test_batch_builder_parity():
    relation_dict = {"atom": 2}
    symbol_type_id = "_symbol_"
    edge_types = [
        (symbol_type_id, "0", "atom"),
    ]

    # Python builder + batch builder
    py_builder = PygHeteroBuilder()
    py_builder.add_node("o0", symbol_type_id, name="o0")
    py_builder.add_node("o1", symbol_type_id, name="o1")
    py_builder.add_node("a0", "atom")
    py_builder.add_node("a1", "atom")
    py_builder.add_edge("o0", "a0", symbol_type_id, "atom", "0")
    py_builder.add_edge("o1", "a1", symbol_type_id, "atom", "0")

    py_batch_builder = HGraphBatchBuilder(
        relation_dict=relation_dict,
        symbol_type_id=symbol_type_id,
        edge_types=edge_types,
    )
    py_batch_builder.append(py_builder)
    py_batch = py_batch_builder.build()

    # C++ builder
    cpp_builder = mifrost.BatchBuilder()
    cpp_builder.set_graph_kind("hetero")
    cpp_builder.add_node_features(
        symbol_type_id, "x", torch.zeros(2, 1, dtype=torch.float32)
    )
    cpp_builder.add_node_features("atom", "x", torch.zeros(2, 2, dtype=torch.float32))
    cpp_builder.set_node_names(symbol_type_id, ["o0", "o1"])
    cpp_builder.set_node_names("atom", ["a0", "a1"])
    cpp_builder.set_object_names(["o0", "o1"])
    cpp_builder.add_edges(
        symbol_type_id,
        "0",
        "atom",
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int64),
    )
    cpp_batch = cpp_builder.build()

    _compare_batches(py_batch, cpp_batch)


def test_offset_logic_explicit():
    """Verify C++ offsets match expected PyG behavior"""
    b = mifrost.BatchBuilder()

    # Graph 0: 2 atoms
    b.add_node_features("atom", "x", torch.zeros(2, 1))
    b.add_edges("atom", "rel", "atom", torch.tensor([0]), torch.tensor([1]))
    b.next_graph()

    # Graph 1: 3 atoms
    b.add_node_features("atom", "x", torch.zeros(3, 1))
    b.add_edges("atom", "rel", "atom", torch.tensor([0, 1]), torch.tensor([1, 2]))
    b.next_graph()

    out = b.build()

    # Check ptr
    ptr = out["atom"].ptr
    assert torch.equal(ptr, torch.tensor([0, 2, 5], dtype=torch.int64))

    # Check edge indices (src)
    src = out[("atom", "rel", "atom")].edge_index[0]
    # Graph 0: 0->1. Graph 1 (offset 2): 0+2->1+2, 1+2->2+2 => 2->3, 3->4
    expected_src = torch.tensor([0, 2, 3], dtype=torch.int64)
    assert torch.equal(src, expected_src)
