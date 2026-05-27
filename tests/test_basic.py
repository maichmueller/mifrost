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
    batch_builder_header = include_dir / "mifrost" / "core" / "batch_builder.hpp"
    assert batch_builder_header.is_file()
    assert "nanobind" not in batch_builder_header.read_text()
    assert not (include_dir / "mifrost" / "core" / "map_view.hpp").exists()

    assert library_dir.is_dir()
    assert any(
        path.name.startswith("libmifrost_core") for path in library_dir.iterdir()
    )

    assert cmake_dir.is_dir()
    config_path = cmake_dir / "mifrostConfig.cmake"
    targets_path = cmake_dir / "mifrostTargets.cmake"
    assert config_path.is_file()
    assert targets_path.is_file()

    config_text = config_path.read_text()
    assert "find_dependency(nanobind" not in config_text

    targets_text = targets_path.read_text()
    assert "nanobind::nanobind" not in targets_text
    assert ".conan" not in targets_text
