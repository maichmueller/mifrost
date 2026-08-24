from __future__ import annotations

import pytest

import mifrost
from mifrost.encoders.derived import (
    AtomLineGraphEncoder,
    ObjectGraphEncoder,
    StarGraphEncoder,
)

try:
    from tests.conftest import load_problem
except ImportError:  # pragma: no cover - wheel test layouts
    from conftest import load_problem  # type: ignore[no-redef]

_RUNTIME_MISSING = (
    ModuleNotFoundError,
    ImportError,
)

_KNOWN_OPTIONAL_BACKEND_MODULES = (
    "pymimir",
    "pytyr",
    "_pymimir_adapter",
    "_pytyr_adapter",
)


def _blocks_state():
    _domain, problem, state, _domain_path, _problem_path = load_problem(
        "blocks", "small"
    )
    return problem, state


def _skip_if_runtime_unavailable(exc: Exception) -> None:
    if isinstance(exc, _RUNTIME_MISSING) and any(
        module in str(exc) for module in _KNOWN_OPTIONAL_BACKEND_MODULES
    ):
        pytest.skip("pymimir derived runtime not yet available")
    raise exc


@pytest.fixture(scope="module")
def blocks_small():
    return _blocks_state()


def test_star_graph_encoder_encode_pyg(blocks_small) -> None:
    problem, state = blocks_small
    try:
        encoder = StarGraphEncoder(problem)
    except Exception as exc:
        _skip_if_runtime_unavailable(exc)
    data = encoder.encode_pyg(state)
    assert data.x_ids is not None
    assert data.x_ids.dim() == 2
    assert data.x_ids.shape[1] == 6
    roles = set(data.x_ids[:, 0].unique().tolist())
    assert roles.issubset({0, 1, 2})
    assert data.edge_index.shape[0] == 2
    assert data.edge_attr.shape[1] == 3


def test_object_graph_encoder_chain_objects_only(blocks_small) -> None:
    problem, state = blocks_small
    try:
        encoder = ObjectGraphEncoder(problem, atom_expansion="chain")
    except Exception as exc:
        _skip_if_runtime_unavailable(exc)
    data = encoder.encode_pyg(state)
    assert data.x_ids.shape[1] == 6
    assert (data.x_ids[:, 0] == 0).all()


def test_atom_line_graph_encoder_includes_line_share(blocks_small) -> None:
    problem, state = blocks_small
    try:
        encoder = AtomLineGraphEncoder(problem)
    except Exception as exc:
        _skip_if_runtime_unavailable(exc)
    data = encoder.encode_pyg(state)
    assert data.edge_attr.shape[1] == 3
    edge_kinds = set(data.edge_attr[:, 0].unique().tolist())
    assert 11 in edge_kinds


def test_encode_batch_doubles_nodes(blocks_small) -> None:
    problem, state = blocks_small
    try:
        encoder = StarGraphEncoder(problem)
    except Exception as exc:
        _skip_if_runtime_unavailable(exc)
    single = encoder.encode_pyg(state)
    batch = encoder.encode_batch_pyg([state, state])
    assert batch.num_graphs == 2
    assert batch.num_nodes == 2 * single.num_nodes


def test_unknown_encode_kwarg_raises_type_error(blocks_small) -> None:
    problem, state = blocks_small
    try:
        encoder = StarGraphEncoder(problem)
    except Exception as exc:
        _skip_if_runtime_unavailable(exc)
    with pytest.raises(TypeError):
        encoder.encode_pyg(state, bogus_lane=1)


def test_object_graph_encoder_rejects_bad_expansion(blocks_small) -> None:
    problem, _state = blocks_small
    with pytest.raises(Exception) as excinfo:
        ObjectGraphEncoder(problem, atom_expansion="bogus")
    assert not isinstance(excinfo.value, _RUNTIME_MISSING)


def test_manifest_exports() -> None:
    for name in (
        "StarGraphEncoder",
        "StarGraphEncoderStream",
        "ObjectGraphEncoder",
        "ObjectGraphEncoderStream",
        "AtomLineGraphEncoder",
        "AtomLineGraphEncoderStream",
        "HypergraphIncidenceEncoder",
        "HypergraphIncidenceEncoderStream",
        "TransformerBiasEncoder",
        "TransformerBiasEncoderStream",
        "TupleTensorEncoder",
        "TupleTensorEncoderStream",
    ):
        assert hasattr(mifrost, name)
