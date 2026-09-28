"""Guards on the order in which native planner libraries enter the process."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_ORDER_PROBE = textwrap.dedent(
    """
    import json
    import sys

    # `sys.modules` keeps insertion order, and a module enters it as its load
    # begins -- so this is the order modules were *loaded*, which is what the
    # rpath lookup depends on. A meta-path hook would instead record lookups,
    # and looking a module up (e.g. `importlib.util.find_spec`) loads nothing.
    preloaded = set(sys.modules)

    import mifrost
    from mifrost import _core

    print(
        json.dumps(
            {
                "order": [name for name in sys.modules if name not in preloaded],
                "adapter_loaded": _core._pymimir_adapter is not None,
            }
        )
    )
    """
)


def _run_probe() -> dict:
    import json

    result = subprocess.run(
        [sys.executable, "-c", _ORDER_PROBE],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"import probe failed:\n{result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_pymimir_is_imported_before_its_native_adapter() -> None:
    """`mifrost` must not be able to load a foreign copy of libmimir.

    ``mifrost._pymimir_adapter`` and ``pymimir`` both link
    ``@rpath/libmimir_core``, and the adapter carries an absolute rpath into
    the environment it was *built* in. A shared editable ``src/`` tree is
    therefore importable from a second environment, and loading the adapter
    before ``pymimir`` resolves that rpath to the builder's libmimir. The
    process then holds two mimir images with disjoint static repositories and
    objects passed between them fail deep inside mimir -- historically
    ``StateSpace.create`` raising ``IndexError: absl::...raw_hash_map<>::at``.

    Importing ``pymimir`` first pins the resolution to this interpreter's copy,
    so the ordering is a load-bearing invariant rather than an accident.
    """
    probe = _run_probe()
    if not probe["adapter_loaded"]:
        pytest.skip("built without the pymimir backend")

    order = probe["order"]
    assert "pymimir" in order, "pymimir was never imported by `import mifrost`"
    assert "mifrost._pymimir_adapter" in order

    assert order.index("pymimir") < order.index("mifrost._pymimir_adapter"), (
        "mifrost._pymimir_adapter was imported before pymimir; a foreign "
        "libmimir can win the @rpath lookup"
    )
