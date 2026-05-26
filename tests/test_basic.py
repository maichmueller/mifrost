from pathlib import Path

import mifrost as mif
from torch_geometric.data import Batch


def test_add():
    builder = mif.BatchBuilder()
    out = builder.build().as_pyg()
    assert isinstance(out, Batch)


def test_sdk_helpers_expose_installed_layout():
    include_dir = Path(mif.get_include_dir())
    library_dir = Path(mif.get_library_dir())
    cmake_dir = Path(mif.get_cmake_dir())

    assert mif.get_include() == str(include_dir)
    assert include_dir.is_dir()
    assert (include_dir / "mifrost" / "core" / "batch_builder.hpp").is_file()

    assert library_dir.is_dir()
    assert any(
        path.name.startswith("libmifrost_core") for path in library_dir.iterdir()
    )

    assert cmake_dir.is_dir()
    assert (cmake_dir / "mifrostConfig.cmake").is_file()
