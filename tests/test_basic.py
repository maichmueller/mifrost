import mifrost as mif
from torch_geometric.data import Batch


def test_add():
    builder = mif.BatchBuilder()
    out = builder.build().as_pyg()
    assert isinstance(out, Batch)
