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
    assert data.x_ids.shape[1] == len(data.channel_names)
    roles = set(data.x_ids[:, 0].unique().tolist())
    assert roles.issubset({0, 1, 2})
    assert data.edge_index.shape[0] == 2
    # The carrier declares its own edge width; reading it from
    # ``edge_channel_names`` means a future channel change is caught here
    # instead of silently passing a hard-coded number.
    assert data.edge_attr.shape[1] == len(data.edge_channel_names)
    assert data.edge_channel_names[:3] == ["kind", "pos_a", "pos_b"]


def test_object_graph_encoder_chain_objects_only(blocks_small) -> None:
    """``objects_only`` materializes objects, then actions, then the anchor.

    The universe is "objects only" in the sense that no *atom* is reified,
    not that every node is an object: the anchor node (role 6) carries the
    arity-0 literals' ``nullary_self`` loops and grounded actions keep their
    own reified node (role 5) in every universe.
    """
    problem, state = blocks_small
    try:
        encoder = ObjectGraphEncoder(problem, atom_expansion="chain")
    except Exception as exc:
        _skip_if_runtime_unavailable(exc)
    data = encoder.encode_pyg(state)
    assert data.x_ids.shape[1] == len(data.channel_names)

    num_objects = len(data.object_names)
    assert data.x_ids[:, 0].long().tolist() == [0] * num_objects + [6]
    assert list(data.node_names) == [*data.object_names, "<nullary>"]
    # Only the role channel is populated for object and anchor nodes.
    assert (data.x_ids[:, 1:] == 0).all()
    assert bool(data.has_anchor)
    assert data.anchor_node_index == data.num_nodes - 1

    action = state.generate_applicable_actions()[0]
    with_action = encoder.encode_pyg(state, actions=[action])
    assert with_action.x_ids[:, 0].long().tolist() == [0] * num_objects + [5, 6]
    assert with_action.anchor_node_index == with_action.num_nodes - 1


def test_atom_line_graph_encoder_includes_line_share(blocks_small) -> None:
    problem, state = blocks_small
    try:
        encoder = AtomLineGraphEncoder(problem)
    except Exception as exc:
        _skip_if_runtime_unavailable(exc)
    data = encoder.encode_pyg(state)
    assert data.edge_attr.shape[1] == len(data.edge_channel_names)
    edge_kinds = set(data.edge_attr[:, 0].unique().tolist())
    assert data.vocab_edge_kinds.index("line_share") in edge_kinds


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
