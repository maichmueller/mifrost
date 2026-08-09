from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_flat_horizon_import_does_not_eagerly_load_pyg() -> None:
    script = """
import sys

from mifrost.encoders import FlatHorizonEncoder

assert FlatHorizonEncoder.__name__ == "FlatHorizonEncoder"
assert "torch_geometric" not in sys.modules
assert "mifrost.encoders.flat_data" not in sys.modules

# The public PyG-specific export remains available and loads its optional
# dependency only when that API is actually requested.
from mifrost.encoders import FlatRelationData

assert FlatRelationData.__name__ == "FlatRelationData"
assert "torch_geometric" in sys.modules
assert "mifrost.encoders.flat_data" in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
