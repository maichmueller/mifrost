import pytest
import torch
import mifrost
from plangolin.encoding.pyg_batch_builder import HGraphBatchBuilder, PygBuilder

# Mock or use real encoder logic?
# Ideally we import the real python HGraphStreamEncoder-equivalent logic if accessible.
# For now, let's verify BatchBuilder parity.

def test_batch_builder_parity():
    # Python Builder
    py_builder = HGraphBatchBuilder(
        relation_dict={"atom": 1}, 
        symbol_type_id="_symbol_", 
        edge_types=[("atom", "rel", "atom")]
    )
    
    # C++ Builder
    cpp_builder = mifrost.BatchBuilder()
    
    # --- Step 1 ---
    x = torch.randn(10, 1)
    
    # Python: intermediate PygBuilder
    pb = PygBuilder()
    pb.node_keys["atom"] = [f"a{i}" for i in range(10)]
    pb.node_attrs["atom"]["type"] = x.tolist() # Mock attribute logic
    # Actually PygBatchBuilder is complex to mock inputs for without full stack.
    # Let's verify output structure logic instead.
    
    # ... (This test requires deeper setup of the Python side infrastructure to be meaningful)
    # Skipping deep parity logic for this skeleton phase.
    pass

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
    ptr = out["atom/ptr"]
    assert torch.equal(ptr, torch.tensor([0, 2, 5], dtype=torch.int64))
    
    # Check edge indices (src)
    src = out["atom|rel|atom/edge_index_0"]
    # Graph 0: 0->1. Graph 1 (offset 2): 0+2->1+2, 1+2->2+2 => 2->3, 3->4
    expected_src = torch.tensor([0, 2, 3], dtype=torch.int64)
    assert torch.equal(src.flatten(), expected_src)
