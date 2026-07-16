from __future__ import annotations

from io import BytesIO
import pickle
from typing import Any

import pytest
import torch

import mifrost

from .test_horizon_public_backends import (
    _aligned_horizon_transitions,
    _paired_dags,
)
from .test_semantic_parity import _backend_pair
from .test_transition_public_backends import _aligned_transition


def _assert_mixed_batch_roundtrip(
    pymimir_encoding: Any,
    pytyr_encoding: Any,
) -> Any:
    assert pymimir_encoding.schema_fingerprint() == (
        pytyr_encoding.schema_fingerprint()
    )
    mixed = mifrost.batch_encodings([pymimir_encoding, pytyr_encoding], fast_path=True)
    assert mixed.num_graphs == 2
    assert mixed.schema_fingerprint() == pymimir_encoding.schema_fingerprint()

    payload = mixed.dumps(include_metadata=True)
    restored = mifrost.BatchEncoding.loads(payload)
    assert restored.dumps(include_metadata=True) == payload
    restored = pickle.loads(pickle.dumps(restored))
    assert restored.dumps(include_metadata=True) == payload

    data = restored.as_pyg(as_batch=True)
    if hasattr(data, "batch"):
        assert set(data.batch.tolist()) == {0, 1}
    else:
        assert any(set(batch.tolist()) == {0, 1} for batch in data.batch_dict.values())
    return restored


@pytest.mark.parametrize(
    "encoder_type",
    [
        mifrost.FlatRelationEncoder,
        mifrost.ColorEncoder,
        mifrost.HGraphEncoder,
    ],
)
def test_state_encoder_outputs_batch_and_serialize_across_backends(
    encoder_type: type[Any],
) -> None:
    _pymimir_reader, problem, pytyr_reader, successor_generator = _backend_pair()
    pymimir_encoder = encoder_type(problem.get_domain())
    pytyr_encoder = encoder_type(pytyr_reader._planning_task)

    _assert_mixed_batch_roundtrip(
        pymimir_encoder.encode(problem.get_initial_state()),
        pytyr_encoder.encode(successor_generator.get_initial_node().get_state()),
    )


@pytest.mark.parametrize(
    "encoder_type",
    [
        mifrost.TransitionHGraphEncoder,
        mifrost.TransitionEffectsHGraphEncoder,
        mifrost.FlatTransitionEncoder,
        mifrost.FlatTransitionEffectsEncoder,
    ],
)
def test_transition_outputs_batch_and_serialize_across_backends(
    encoder_type: type[Any],
) -> None:
    (
        _pymimir_reader,
        problem,
        pymimir_current,
        _pymimir_action,
        pymimir_successor,
        pytyr_reader,
        pytyr_current,
        _pytyr_action,
        pytyr_successor,
    ) = _aligned_transition()
    pymimir_encoder = encoder_type(problem.get_domain())
    pytyr_encoder = encoder_type(pytyr_reader._planning_task)

    _assert_mixed_batch_roundtrip(
        pymimir_encoder.encode(
            pymimir_current,
            successor=pymimir_successor,
        ),
        pytyr_encoder.encode(
            pytyr_current,
            successor=pytyr_successor,
        ),
    )


@pytest.mark.parametrize(
    "encoder_type",
    [mifrost.HorizonEncoder, mifrost.FlatHorizonEncoder],
)
def test_horizon_outputs_batch_and_serialize_across_backends(
    encoder_type: type[Any],
) -> None:
    problem, pymimir_root, pytyr_reader, pytyr_root, transitions = (
        _aligned_horizon_transitions(count=1)
    )
    pymimir_dag, pytyr_dag = _paired_dags(transitions, pymimir_root, pytyr_root)
    config = {"goal_derivations": {mifrost.GoalDerivation.plain}}
    pymimir_encoder = encoder_type(problem.get_domain(), **config)
    pytyr_encoder = encoder_type(pytyr_reader._planning_task, **config)

    _assert_mixed_batch_roundtrip(
        pymimir_encoder.encode(pymimir_root, dag=pymimir_dag),
        pytyr_encoder.encode(pytyr_root, dag=pytyr_dag),
    )


def test_mixed_backend_hgraph_runs_one_learning_and_checkpoint_step() -> None:
    _pymimir_reader, problem, pytyr_reader, successor_generator = _backend_pair()
    pymimir_encoder = mifrost.HGraphEncoder(problem.get_domain())
    pytyr_encoder = mifrost.HGraphEncoder(pytyr_reader._planning_task)
    mixed = _assert_mixed_batch_roundtrip(
        pymimir_encoder.encode(problem.get_initial_state()),
        pytyr_encoder.encode(successor_generator.get_initial_node().get_state()),
    )
    data = mixed.as_pyg(as_batch=True)
    object_x = data["object"].x.float()
    object_batch = data["object"].batch
    model = torch.nn.Linear(object_x.shape[1], 1)
    node_scores = model(object_x).squeeze(-1)
    graph_scores = torch.zeros(mixed.num_graphs, dtype=node_scores.dtype)
    graph_scores.index_add_(0, object_batch, node_scores)
    graph_scores.square().mean().backward()
    assert model.weight.grad is not None

    checkpoint_buffer = BytesIO()
    torch.save(
        {
            "schema_fingerprint": mixed.schema_fingerprint(),
            "model": model.state_dict(),
            "encoding": mixed.dumps(include_metadata=True),
        },
        checkpoint_buffer,
    )
    checkpoint_buffer.seek(0)
    checkpoint = torch.load(checkpoint_buffer, weights_only=True)
    restored = mifrost.BatchEncoding.loads(checkpoint["encoding"])
    restored_model = torch.nn.Linear(object_x.shape[1], 1)
    restored_model.load_state_dict(checkpoint["model"])
    assert checkpoint["schema_fingerprint"] == restored.schema_fingerprint()
    assert torch.equal(restored_model.weight, model.weight)
