import torch
import mifrost


def test_batch_builder_parity():
    symbol_type_id = "_symbol_"

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
    assert set(cpp_batch.node_types) == {symbol_type_id, "atom"}
    symbol_names = list(cpp_batch[symbol_type_id].node_names)
    atom_names = list(cpp_batch["atom"].node_names)
    object_names = list(cpp_batch.object_names)
    if symbol_names and isinstance(symbol_names[0], list):
        symbol_names = symbol_names[0]
    if atom_names and isinstance(atom_names[0], list):
        atom_names = atom_names[0]
    if object_names and isinstance(object_names[0], list):
        object_names = object_names[0]
    assert symbol_names == ["o0", "o1"]
    assert atom_names == ["a0", "a1"]
    assert object_names == ["o0", "o1"]

    edge_type = (symbol_type_id, "0", "atom")
    assert edge_type in cpp_batch.edge_types
    edge_index = cpp_batch[edge_type].edge_index
    assert torch.equal(edge_index, torch.tensor([[0, 1], [0, 1]], dtype=torch.int64))


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
