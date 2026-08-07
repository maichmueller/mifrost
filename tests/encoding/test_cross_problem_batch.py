"""One encoder, several instances of its domain, one batch.

This is the public-API face of the schema/problem context split: an encoder is
constructed from a *domain*, so every problem of that domain is fair game --
including two of them in the same ``encode_batch`` call. Instances differ in
object count, object names, static facts and goals, and none of that prevents
batching, because each graph carries its own object table and ``BatchBuilder``
offsets the graph-local node indices independently.

Before the split, a second instance either raised (flat) or made the facade
throw away its engine and build a new one bound to that instance (hgraph,
color) -- which also discarded whatever ``update_relations`` had put there.
"""

from __future__ import annotations

import pytest

from mifrost.encoders import ColorEncoder, FlatRelationEncoder, HGraphEncoder
from tests.conftest import load_problem


# Two instances of one domain, deliberately of different size.
INSTANCE_PAIRS = [
    ("blocks", "small", "smedium"),
    ("gripper", "gripper_b-1", "gripper_b-5"),
    ("spanner", "small", "medium"),
    ("delivery", "instance_2x2_p-1_0", "instance_3x3_p-3_0"),
]

ENCODERS = [FlatRelationEncoder, HGraphEncoder, ColorEncoder]


@pytest.fixture(
    scope="module",
    params=INSTANCE_PAIRS,
    ids=[f"{domain}:{first}+{second}" for domain, first, second in INSTANCE_PAIRS],
)
def instance_pair(request):
    domain_name, first_name, second_name = request.param
    domain, first, _first_state, _dp, _pp = load_problem(domain_name, first_name)
    _domain2, second, _second_state, _dp2, _pp2 = load_problem(domain_name, second_name)
    return domain, first, second


@pytest.mark.parametrize("encoder_class", ENCODERS, ids=lambda cls: cls.__name__)
def test_two_instances_encode_in_one_batch(instance_pair, encoder_class):
    domain, first, second = instance_pair
    encoder = encoder_class(domain)

    first_state = first.get_initial_state()
    second_state = second.get_initial_state()

    # The instances really are different, or the test proves nothing.
    assert len(first.get_objects()) != len(second.get_objects())

    mixed = encoder.encode_batch([first_state, second_state])

    assert mixed.num_graphs == 2

    # Node for node, the mixed batch is the two single-instance encodings.
    first_only = encoder.encode_batch([first_state])
    second_only = encoder.encode_batch([second_state])
    assert mixed.num_nodes == first_only.num_nodes + second_only.num_nodes
    assert mixed.num_edges == first_only.num_edges + second_only.num_edges
    assert set(mixed.node_types) == set(first_only.node_types) | set(
        second_only.node_types
    )


@pytest.mark.parametrize("encoder_class", ENCODERS, ids=lambda cls: cls.__name__)
def test_instances_alternate_through_one_encoder(instance_pair, encoder_class):
    """Interleaved single-state calls must not disturb one another.

    A per-problem cache that dropped its previous entry, or an engine that
    re-bound on every switch, would still pass a straight A, B sequence. This
    goes back to A afterwards and requires the original answer.
    """
    domain, first, second = instance_pair
    encoder = encoder_class(domain)

    first_state = first.get_initial_state()
    second_state = second.get_initial_state()

    before = encoder.encode(first_state)
    _ = encoder.encode(second_state)
    after = encoder.encode(first_state)

    assert after.num_nodes == before.num_nodes
    assert after.num_edges == before.num_edges
