"""StateView-backed facade tests, plus the end-to-end relmo integration check.

``relmo`` (the sibling ``relm`` repository's package) is not a normal
dependency of mifrost, so the integration test guards its import with
``pytest.importorskip`` and is skipped in environments that only have
mifrost installed.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import problem_setup

from mifrost.encoders.sparse_atom import (
    CHANNEL_SATISFIED,
    CHANNEL_UNSATISFIED,
    STATUS_ENCODING_VOCABULARY,
    SparseAtomCompositionEncoder,
    validate_sparse_atom_composition,
)

relmo_models = pytest.importorskip(
    "relmo.models", reason="relmo is not installed in this environment"
)


@pytest.fixture(scope="module")
def blocks_problem():
    _space, _domain, problem = problem_setup("blocks", "smedium")
    return problem


def test_encoder_schema_is_stable_and_arity_normalized(blocks_problem) -> None:
    encoder = SparseAtomCompositionEncoder(blocks_problem)
    assert "on" in encoder.predicate_names
    on_id = encoder.schema.name_to_id["on"]
    handempty_id = encoder.schema.name_to_id["handempty"]
    assert encoder.predicate_arities[on_id] == 2
    # handempty is nullary in PDDL but reported with encoded arity 1
    assert encoder.predicate_arities[handempty_id] == 1
    assert encoder.schema.logical_arities[handempty_id] == 0


def test_goal_free_vs_goal_aware_encode(blocks_problem) -> None:
    encoder = SparseAtomCompositionEncoder(blocks_problem)
    state = blocks_problem.get_initial_state()

    goal_free = encoder.encode(state)
    validate_sparse_atom_composition(
        goal_free, predicate_arities=encoder.predicate_arities
    )
    assert bool(goal_free.goal_available[0].item()) is False
    assert goal_free.counterpart_occurrence_ids is None

    goal_literals = encoder.view.goal_literals(state)
    goal_aware = encoder.encode(state, goals=goal_literals)
    validate_sparse_atom_composition(
        goal_aware, predicate_arities=encoder.predicate_arities
    )
    assert bool(goal_aware.goal_available[0].item()) is True
    num_current = len(encoder.view.static_facts) + len(encoder.view.state_facts(state))
    assert goal_aware.num_atoms == goal_free.num_atoms + len(goal_literals)
    goal_channels = goal_aware.atom_channel_ids[
        num_current : num_current + len(goal_literals)
    ]
    assert set(goal_channels.tolist()) <= {CHANNEL_SATISFIED, CHANNEL_UNSATISFIED}


def test_supplied_empty_vs_omitted_goal_via_encoder(blocks_problem) -> None:
    encoder = SparseAtomCompositionEncoder(blocks_problem)
    state = blocks_problem.get_initial_state()

    omitted = encoder.encode(state)
    supplied_empty = encoder.encode(state, goals=[])

    assert omitted.num_atoms == supplied_empty.num_atoms
    assert torch.equal(omitted.atom_predicate_ids, supplied_empty.atom_predicate_ids)
    assert bool(omitted.goal_available[0].item()) is False
    assert bool(supplied_empty.goal_available[0].item()) is True


def test_negative_goal_literal_rejected(blocks_problem) -> None:
    encoder = SparseAtomCompositionEncoder(blocks_problem)
    state = blocks_problem.get_initial_state()
    goal_literals = encoder.view.goal_literals(state)
    negated = [type(goal_literals[0])(atom=goal_literals[0].atom, positive=False)]
    with pytest.raises(ValueError, match="positive"):
        encoder.encode(state, goals=negated)


def test_encode_batch_matches_solo_encodes(blocks_problem) -> None:
    encoder = SparseAtomCompositionEncoder(blocks_problem)
    state = blocks_problem.get_initial_state()
    goal_literals = encoder.view.goal_literals(state)

    solo_goal_free = encoder.encode(state)
    solo_goal_aware = encoder.encode(state, goals=goal_literals)

    batch = encoder.encode_batch([state, state], goals=[None, goal_literals])
    validate_sparse_atom_composition(batch, predicate_arities=encoder.predicate_arities)

    assert batch.num_atoms == solo_goal_free.num_atoms + solo_goal_aware.num_atoms
    assert torch.equal(
        batch.atom_predicate_ids,
        torch.cat(
            [solo_goal_free.atom_predicate_ids, solo_goal_aware.atom_predicate_ids]
        ),
    )
    assert bool((batch.goal_available == torch.tensor([False, True])).all())
    # no pair or witness crosses the two graphs
    object_graph = batch.object_batch
    if batch.num_pairs:
        left = object_graph.index_select(0, batch.pair_objects[:, 0])
        right = object_graph.index_select(0, batch.pair_objects[:, 1])
        assert bool((left == right).all())


# --------------------------------------------------------------------------
# End-to-end: real encoder output straight into the real consumer model
# --------------------------------------------------------------------------


def test_encoder_output_accepted_by_sparse_atom_composition_gnn(blocks_problem) -> None:
    encoder = SparseAtomCompositionEncoder(blocks_problem)
    state = blocks_problem.get_initial_state()
    goal_literals = encoder.view.goal_literals(state)
    encoding = encoder.encode(state, goals=goal_literals)
    validate_sparse_atom_composition(
        encoding, predicate_arities=encoder.predicate_arities
    )

    model = relmo_models.SparseAtomCompositionGNN(
        embedding_size=16,
        num_layers=2,
        predicate_arities=list(encoder.predicate_arities),
        num_channels=encoder.num_channels,
    )
    prepared = model.prepare(encoding)  # must not raise
    output = model(prepared)

    num_graphs = int(encoding.goal_available.numel())
    assert output.state.shape == (num_graphs, 16)
    assert output.occurrence.shape == (encoding.num_occurrences, 16)
    assert output.object.shape == (encoding.num_objects, 16)
    assert output.atom.shape == (encoding.num_atoms, 16)
    for tensor in (output.state, output.occurrence, output.object, output.atom):
        assert torch.isfinite(tensor).all()


def test_encoder_batch_output_accepted_by_sparse_atom_composition_gnn(
    blocks_problem,
) -> None:
    encoder = SparseAtomCompositionEncoder(blocks_problem)
    state = blocks_problem.get_initial_state()
    goal_literals = encoder.view.goal_literals(state)
    batch = encoder.encode_batch([state, state], goals=[None, goal_literals])
    validate_sparse_atom_composition(batch, predicate_arities=encoder.predicate_arities)

    model = relmo_models.SparseAtomCompositionGNN(
        embedding_size=8,
        num_layers=2,
        predicate_arities=list(encoder.predicate_arities),
        num_channels=encoder.num_channels,
    )
    output = model(batch)
    assert output.state.shape == (2, 8)
    assert torch.isfinite(output.state).all()


def test_exact_tuple_exchange_end_to_end(blocks_problem) -> None:
    encoder = SparseAtomCompositionEncoder(blocks_problem, exact_tuple_exchange=True)
    state = blocks_problem.get_initial_state()
    goal_literals = encoder.view.goal_literals(state)
    encoding = encoder.encode(state, goals=goal_literals)
    validate_sparse_atom_composition(
        encoding, predicate_arities=encoder.predicate_arities
    )

    model = relmo_models.SparseAtomCompositionGNN(
        embedding_size=8,
        num_layers=1,
        predicate_arities=list(encoder.predicate_arities),
        num_channels=encoder.num_channels,
        exact_tuple_exchange=True,
    )
    output = model(encoding)
    assert torch.isfinite(output.state).all()


def test_relmo_prepare_rejects_a_broken_carrier(blocks_problem) -> None:
    """Sanity-check that relmo's own prepare() and our validator agree."""
    encoder = SparseAtomCompositionEncoder(blocks_problem)
    state = blocks_problem.get_initial_state()
    encoding = encoder.encode(state, goals=encoder.view.goal_literals(state))
    encoding.pair_objects = encoding.pair_objects.clone()
    encoding.pair_objects[0, 1] = encoding.pair_objects[0, 0]

    with pytest.raises(ValueError):
        validate_sparse_atom_composition(
            encoding, predicate_arities=encoder.predicate_arities
        )

    model = relmo_models.SparseAtomCompositionGNN(
        embedding_size=8,
        num_layers=1,
        predicate_arities=list(encoder.predicate_arities),
        num_channels=encoder.num_channels,
    )
    with pytest.raises(ValueError):
        model.prepare(encoding)


# --------------------------------------------------------------------------
# R4: object(o) as the carrier, on a real backend problem
# --------------------------------------------------------------------------


def test_real_backend_object_facts_cover_every_object_no_fallback_needed(
    blocks_problem,
) -> None:
    """On a real pymimir problem, `object(o)` is already a static fact for
    every domain object, so the R4 carrier fallback synthesis is never
    exercised here -- only the synthetic star (auxiliary) carrier, when a
    nullary predicate like `handempty` is represented, needs synthesis."""
    from mifrost.encoders.sparse_atom import (
        CHANNEL_STATE,
        CHANNEL_AUXILIARY,
        OBJECT_PREDICATE,
    )

    encoder = SparseAtomCompositionEncoder(blocks_problem)
    state = blocks_problem.get_initial_state()
    encoding = encoder.encode(state, goals=encoder.view.goal_literals(state))

    object_base_id = encoder.schema.base_name_to_id[OBJECT_PREDICATE]
    object_mask = encoding.atom_predicate_ids == object_base_id
    object_channels = encoding.atom_channel_ids[object_mask].tolist()
    real_objects = len(encoder.view.objects)
    # Every real object's carrier is in the state channel (a genuine static
    # fact); at most one extra auxiliary entry is the synthesized star.
    assert object_channels[:real_objects] == [CHANNEL_STATE] * real_objects
    assert all(
        channel == CHANNEL_AUXILIARY for channel in object_channels[real_objects:]
    )


# --------------------------------------------------------------------------
# status_encoding="vocabulary": same topology, native GNN round-trip
# --------------------------------------------------------------------------


def test_vocabulary_status_encoding_matches_channel_topology_on_real_problem(
    blocks_problem,
) -> None:
    channel_encoder = SparseAtomCompositionEncoder(blocks_problem)
    vocab_encoder = SparseAtomCompositionEncoder(
        blocks_problem, status_encoding=STATUS_ENCODING_VOCABULARY
    )
    assert vocab_encoder.num_channels == 1
    assert channel_encoder.num_channels == 4

    state = blocks_problem.get_initial_state()
    goal_literals = channel_encoder.view.goal_literals(state)
    channel_encoding = channel_encoder.encode(state, goals=goal_literals)
    vocab_encoding = vocab_encoder.encode(state, goals=goal_literals)

    validate_sparse_atom_composition(
        vocab_encoding,
        predicate_arities=vocab_encoder.predicate_arities,
        num_channels=vocab_encoder.num_channels,
    )
    assert torch.equal(vocab_encoding.pair_objects, channel_encoding.pair_objects)
    assert torch.equal(
        vocab_encoding.composition_triplets, channel_encoding.composition_triplets
    )
    assert torch.equal(vocab_encoding.atom_pair_ids, channel_encoding.atom_pair_ids)
    assert bool((vocab_encoding.atom_channel_ids == 0).all())


def test_vocabulary_status_encoding_accepted_by_sparse_atom_composition_gnn(
    blocks_problem,
) -> None:
    encoder = SparseAtomCompositionEncoder(
        blocks_problem, status_encoding=STATUS_ENCODING_VOCABULARY
    )
    state = blocks_problem.get_initial_state()
    goal_literals = encoder.view.goal_literals(state)
    encoding = encoder.encode(state, goals=goal_literals)
    validate_sparse_atom_composition(
        encoding, predicate_arities=encoder.predicate_arities, num_channels=1
    )

    model = relmo_models.SparseAtomCompositionGNN(
        embedding_size=8,
        num_layers=2,
        predicate_arities=list(encoder.predicate_arities),
        num_channels=encoder.num_channels,
    )
    prepared = model.prepare(encoding)
    output = model(prepared)
    assert output.state.shape == (1, 8)
    assert torch.isfinite(output.state).all()
