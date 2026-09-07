"""Tests for the pure-Python sparse atom-composition encoder core.

These exercise ``mifrost.encoders.sparse_atom`` directly on hand-built
``Atom`` lists -- no pymimir/pytyr problem is needed for most of this file,
matching the "Stage 1, pure Python" scope of the encoder itself. See
``test_sparse_atom_composition_relmo_integration.py`` for the StateView-backed
facade and the end-to-end check against ``relmo.models.SparseAtomCompositionGNN``.
"""

from __future__ import annotations

import pytest
import torch

from mifrost.encoders.custom.state_view import Atom
from mifrost.encoders.sparse_atom import (
    CHANNEL_AUXILIARY,
    CHANNEL_NAMES,
    CHANNEL_SATISFIED,
    CHANNEL_STATE,
    CHANNEL_STATUS_SUFFIXES,
    CHANNEL_UNSATISFIED,
    OBJECT_PREDICATE,
    ROOT_TYPE_NAME,
    STATUS_ENCODING_VOCABULARY,
    SparseAtomPredicateSchema,
    batch_sparse_atom_encodings,
    build_predicate_schema,
    build_type_schema,
    encode_sparse_atom_facts,
    validate_sparse_atom_composition,
)


def _schema(predicates: list[tuple[str, int]]) -> SparseAtomPredicateSchema:
    return build_predicate_schema(predicates)


def _encode(
    objects,
    current,
    goals,
    schema,
    *,
    exact_tuple_exchange: bool = False,
    object_types=None,
    type_schema=None,
):
    encoding = encode_sparse_atom_facts(
        objects,
        current,
        goals,
        schema,
        exact_tuple_exchange=exact_tuple_exchange,
        object_types=object_types,
        type_schema=type_schema,
    )
    validate_sparse_atom_composition(
        encoding,
        predicate_arities=schema.arities,
        num_channels=schema.num_channels,
        num_object_types=(len(type_schema.names) if type_schema is not None else None),
    )
    return encoding


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


def test_build_predicate_schema_appends_object_carrier_predicate() -> None:
    schema = _schema([("on", 2), ("clear", 1)])
    assert schema.names[-1] == OBJECT_PREDICATE
    assert schema.arities[-1] == 1
    assert schema.name_to_id["on"] == 0
    assert schema.arities == (2, 1, 1)
    assert schema.base_names == schema.names
    assert schema.num_channels == len(CHANNEL_NAMES)


def test_build_predicate_schema_reuses_declared_object_predicate() -> None:
    # Real backends (pymimir, pytyr) already declare `object` as a static
    # unary predicate for the PDDL root type -- verified on `blocks:small` --
    # so the schema must reuse it, not append a second carrier predicate.
    schema = _schema([("on", 2), (OBJECT_PREDICATE, 1)])
    assert schema.names.count(OBJECT_PREDICATE) == 1
    assert schema.arities == (2, 1)


def test_build_predicate_schema_rejects_non_unary_object_predicate() -> None:
    with pytest.raises(ValueError, match="arity 1"):
        build_predicate_schema([(OBJECT_PREDICATE, 2)])


def test_build_predicate_schema_normalizes_nullary_arity() -> None:
    schema = _schema([("handempty", 0)])
    assert schema.logical_arities[schema.name_to_id["handempty"]] == 0
    assert schema.arities[schema.name_to_id["handempty"]] == 1


def test_build_predicate_schema_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_predicate_schema([("on", 2), ("on", 2)])


def test_build_predicate_schema_rejects_unknown_status_encoding() -> None:
    with pytest.raises(ValueError, match="status_encoding"):
        build_predicate_schema([("on", 2)], status_encoding="bogus")


def test_vocabulary_status_encoding_expands_schema_and_names() -> None:
    assert CHANNEL_STATUS_SUFFIXES == ("[state]", "[sat]", "[unsat]", "[g]")
    schema = build_predicate_schema(
        [("on", 2), ("clear", 1)], status_encoding=STATUS_ENCODING_VOCABULARY
    )
    assert schema.num_channels == 1
    assert schema.base_names == ("on", "clear", OBJECT_PREDICATE)
    assert len(schema.names) == len(schema.base_names) * len(CHANNEL_NAMES)

    on_base = schema.base_name_to_id["on"]
    assert schema.names[on_base * 4 + CHANNEL_STATE] == "on[state]"
    assert schema.names[on_base * 4 + CHANNEL_SATISFIED] == "on[sat]"
    assert schema.names[on_base * 4 + CHANNEL_UNSATISFIED] == "on[unsat]"
    assert schema.names[on_base * 4 + CHANNEL_AUXILIARY] == "on[g]"
    assert schema.arities[on_base * 4 + CHANNEL_STATE] == 2
    assert schema.relation_id(on_base, CHANNEL_STATE) == (on_base * 4, 0)


# --------------------------------------------------------------------------
# R13: object-type schema
# --------------------------------------------------------------------------


def test_build_type_schema_appends_root_type() -> None:
    schema = build_type_schema(["truck", "airplane", "city"])
    assert schema.names == ("truck", "airplane", "city", ROOT_TYPE_NAME)
    assert schema.name_to_id["truck"] == 0
    assert schema.name_to_id[ROOT_TYPE_NAME] == 3


def test_build_type_schema_reuses_declared_root_type() -> None:
    # Real backends (pymimir) already declare "object" as the implicit root
    # of every type hierarchy, even in an untyped domain -- so the schema
    # must reuse it, not append a second root entry.
    schema = build_type_schema(["truck", ROOT_TYPE_NAME])
    assert schema.names.count(ROOT_TYPE_NAME) == 1
    assert schema.names == ("truck", ROOT_TYPE_NAME)


def test_build_type_schema_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_type_schema(["truck", "truck"])


def test_build_type_schema_degrades_to_root_only_for_untyped_domains() -> None:
    schema = build_type_schema([])
    assert schema.names == (ROOT_TYPE_NAME,)


# --------------------------------------------------------------------------
# R1/R4: arities, carriers, isolated objects
# --------------------------------------------------------------------------


def test_unary_atom_and_isolated_object_get_carriers() -> None:
    schema = _schema([("clear", 1)])
    encoding = _encode(["a", "b"], [Atom("clear", ("a",))], None, schema)
    # 1 clear atom + 2 object carriers (a is used, b is isolated); neither
    # object has a real backend-supplied `object(o)` fact here, so both
    # carriers are synthesized fallbacks -- in the *state* channel, since
    # `object(o)` is a genuine fact, not an encoding artefact.
    assert encoding.num_atoms == 3
    assert encoding.num_objects == 2
    assert torch.equal(
        encoding.object_carrier_occurrence_ids,
        torch.tensor([int(encoding.atom_offsets[1]), int(encoding.atom_offsets[2])]),
    )
    object_id = schema.name_to_id[OBJECT_PREDICATE]
    assert bool((encoding.atom_predicate_ids[1:] == object_id).all())
    assert bool((encoding.atom_channel_ids[1:] == CHANNEL_STATE).all())


def test_existing_object_carrier_from_current_atoms_is_reused() -> None:
    schema = _schema([("clear", 1)])
    current = [
        Atom(OBJECT_PREDICATE, ("a",)),
        Atom(OBJECT_PREDICATE, ("b",)),
        Atom("clear", ("a",)),
    ]
    encoding = _encode(["a", "b"], current, None, schema)
    # both carriers already exist as genuine current facts: no fallback
    # synthesis is needed, so num_atoms is exactly the 3 supplied atoms.
    assert encoding.num_atoms == 3
    assert bool((encoding.atom_channel_ids == CHANNEL_STATE).all())
    assert torch.equal(
        encoding.object_carrier_occurrence_ids,
        torch.tensor([int(encoding.atom_offsets[0]), int(encoding.atom_offsets[1])]),
    )


def test_missing_object_carrier_is_synthesized_per_object() -> None:
    schema = _schema([("clear", 1)])
    # 'a' has a real carrier already; 'b' (isolated) needs a synthesized one
    # -- decided per object, both land in the state channel.
    current = [Atom(OBJECT_PREDICATE, ("a",)), Atom("clear", ("a",))]
    encoding = _encode(["a", "b"], current, None, schema)
    assert encoding.num_atoms == 3
    assert bool((encoding.atom_channel_ids == CHANNEL_STATE).all())
    object_id = schema.name_to_id[OBJECT_PREDICATE]
    a_carrier, b_carrier = encoding.object_carrier_occurrence_ids.tolist()
    assert a_carrier == 0  # the real object(a) fact, first atom
    assert encoding.atom_predicate_ids[2].item() == object_id  # synthesized object(b)


def test_star_carrier_is_always_auxiliary_even_with_real_object_facts() -> None:
    schema = _schema([("handempty", 0)])
    current = [Atom(OBJECT_PREDICATE, ("a",)), Atom("handempty", ())]
    encoding = _encode(["a"], current, None, schema)
    assert encoding.num_objects == 2  # 'a' plus the synthetic star object
    object_id = schema.name_to_id[OBJECT_PREDICATE]
    # the star carrier is always synthesized last
    assert encoding.atom_predicate_ids[-1].item() == object_id
    assert encoding.atom_channel_ids[-1].item() == CHANNEL_AUXILIARY
    # 'a's carrier is the real fact, in the state channel
    assert encoding.atom_channel_ids[0].item() == CHANNEL_STATE


def test_binary_atom_produces_ordered_pair() -> None:
    schema = _schema([("on", 2)])
    encoding = _encode(["a", "b"], [Atom("on", ("a", "b"))], None, schema)
    assert encoding.num_pairs == 2
    assert torch.equal(encoding.pair_objects, torch.tensor([[0, 1], [1, 0]]))
    assert encoding.composition_triplets.numel() == 0


def test_ternary_atom_wires_all_ordered_distinct_pairs() -> None:
    schema = _schema([("between", 3)])
    encoding = _encode(
        ["a", "b", "c"], [Atom("between", ("a", "b", "c"))], None, schema
    )
    # 3 distinct objects, all 6 ordered pairs represented
    assert encoding.num_pairs == 6
    # a ternary atom on 3 distinct objects is itself a support triangle
    assert encoding.composition_triplets.size(0) == 6


def test_repeated_argument_excluded_from_pairs_and_atom_pair_maps() -> None:
    schema = _schema([("touches", 2)])
    encoding = _encode(["a"], [Atom("touches", ("a", "a"))], None, schema)
    assert encoding.num_pairs == 0
    assert encoding.atom_pair_ids.numel() == 0
    assert encoding.atom_pair_occurrence_i.numel() == 0


def test_nullary_atom_normalized_onto_star_object() -> None:
    schema = _schema([("handempty", 0)])
    encoding = _encode(["a"], [Atom("handempty", ())], None, schema)
    # object 'a' plus the synthetic star object
    assert encoding.num_objects == 2
    assert bool((encoding.atom_offsets[1:] - encoding.atom_offsets[:-1] == 1).all())
    # no arity-0 atom was ever emitted (validate_sparse_atom_composition
    # already asserts this, but check explicitly too)
    arities = encoding.atom_offsets[1:] - encoding.atom_offsets[:-1]
    assert bool((arities > 0).all())


def test_star_object_only_added_when_needed() -> None:
    schema = _schema([("clear", 1)])
    encoding = _encode(["a"], [Atom("clear", ("a",))], None, schema)
    assert encoding.num_objects == 1


def test_star_participates_in_pairs() -> None:
    schema = _schema([("handempty", 0), ("holding", 1)])
    encoding = _encode(
        ["a"],
        [Atom("handempty", ())],
        [Atom("holding", ("a",))],
        schema,
    )
    # star and 'a' are different objects; a binary composition never forms
    # here since both atoms are unary, but both get their own carrier and
    # neither raises despite one being goal-derived.
    assert encoding.num_objects == 2


# --------------------------------------------------------------------------
# R2: goal-status split (satisfied XOR unsatisfied, never plain)
# --------------------------------------------------------------------------


def test_goal_status_split_exactly_one_atom_per_goal() -> None:
    schema = _schema([("on", 2)])
    current = [Atom("on", ("a", "b"))]
    goals = [Atom("on", ("a", "b")), Atom("on", ("b", "a"))]
    encoding = _encode(["a", "b"], current, goals, schema)
    goal_channels = encoding.atom_channel_ids[len(current) : len(current) + len(goals)]
    assert sorted(goal_channels.tolist()) == [CHANNEL_SATISFIED, CHANNEL_UNSATISFIED]
    # exactly one status atom per goal atom: never both, never plain, never
    # dropped (only CHANNEL_STATE/CHANNEL_SATISFIED/CHANNEL_UNSATISFIED/
    # CHANNEL_AUXILIARY ever appear -- there is no "plain" channel at all).
    assert set(encoding.atom_channel_ids.tolist()) <= {
        CHANNEL_STATE,
        CHANNEL_SATISFIED,
        CHANNEL_UNSATISFIED,
        CHANNEL_AUXILIARY,
    }
    num_goal_atoms = int(
        (
            (encoding.atom_channel_ids == CHANNEL_SATISFIED)
            | (encoding.atom_channel_ids == CHANNEL_UNSATISFIED)
        ).sum()
    )
    assert num_goal_atoms == len(goals)


def test_satisfied_goal_keeps_separate_current_atom() -> None:
    schema = _schema([("on", 2)])
    current = [Atom("on", ("a", "b"))]
    goals = [Atom("on", ("a", "b"))]
    encoding = _encode(["a", "b"], current, goals, schema)
    # 1 current + 1 satisfied-goal + 2 carriers = 4 atoms; the current atom
    # is not merged away by the satisfied goal.
    assert encoding.num_atoms == 4
    assert encoding.atom_channel_ids[0].item() == CHANNEL_STATE
    assert encoding.atom_channel_ids[1].item() == CHANNEL_SATISFIED


# --------------------------------------------------------------------------
# R6: the within-atom equality pattern is derived downstream, not encoded
# --------------------------------------------------------------------------


def test_encoding_carries_no_equality_pattern_vocabulary() -> None:
    # eps_q = (1[o_{q,i}=o_{q,j}])_{i,j} is derived by relmo's prepare()
    # directly from atom_args (row j, width max_arity); interning it into a
    # categorical id was a relmo implementation artefact that has since been
    # removed, so the encoder must not compute or carry one at all.
    schema = _schema([("touches", 2)])
    encoding = _encode(["a"], [Atom("touches", ("a", "a"))], None, schema)
    assert not hasattr(encoding, "equality_pattern_ids")


# --------------------------------------------------------------------------
# R7/R8: pairs and witness triplets
# --------------------------------------------------------------------------


def test_empty_pair_and_witness_sets_for_unary_only_state() -> None:
    schema = _schema([("clear", 1), ("ontable", 1)])
    encoding = _encode(
        ["a", "b"],
        [Atom("clear", ("a",)), Atom("ontable", ("b",))],
        None,
        schema,
    )
    assert encoding.num_pairs == 0
    assert encoding.composition_triplets.numel() == 0
    assert encoding.atom_pair_ids.numel() == 0


def test_state_only_blocksworld_yields_zero_support_triangles() -> None:
    schema = _schema(
        [("on", 2), ("clear", 1), ("ontable", 1), ("holding", 1), ("handempty", 0)]
    )
    # z on y on x, w on the table, clear on top of the stack.
    current = [
        Atom("on", ("z", "y")),
        Atom("on", ("y", "x")),
        Atom("ontable", ("x",)),
        Atom("ontable", ("w",)),
        Atom("clear", ("z",)),
        Atom("clear", ("w",)),
        Atom("handempty", ()),
    ]
    encoding = _encode(["x", "y", "z", "w"], current, None, schema)
    # on-chains form a forest: no two on-atoms can share both endpoints in a
    # triangle, and every other represented predicate here is unary, so the
    # undirected support graph has no triangles at all.
    assert encoding.composition_triplets.numel() == 0


def test_six_cycle_vs_two_triangles_same_counts_different_triangle_count() -> None:
    schema = _schema([("edge", 2)])

    def graph(edges: list[tuple[str, str]]):
        atoms = []
        for u, v in edges:
            atoms.append(Atom("edge", (u, v)))
            atoms.append(Atom("edge", (v, u)))
        objects = sorted({o for edge in edges for o in edge})
        return _encode(objects, atoms, None, schema)

    six_cycle = graph(
        [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("f", "a")]
    )
    two_triangles = graph(
        [("a", "b"), ("b", "c"), ("c", "a"), ("d", "e"), ("e", "f"), ("f", "d")]
    )

    assert six_cycle.num_objects == two_triangles.num_objects == 6
    assert six_cycle.num_pairs == two_triangles.num_pairs == 12
    assert six_cycle.composition_triplets.numel() == 0
    # each of the 2 triangles yields 6 ordered (target, left, right) rows
    assert two_triangles.composition_triplets.size(0) == 12


def test_two_tower_blocksworld_goal_witness_separation() -> None:
    schema = _schema([("on", 2), ("clear", 1), ("ontable", 1)])
    objects = ["b1", "b2", "t1", "t2", "m1", "m2"]
    current = [
        Atom("on", ("m1", "b1")),
        Atom("on", ("t1", "m1")),
        Atom("on", ("m2", "b2")),
        Atom("on", ("t2", "m2")),
        Atom("ontable", ("b1",)),
        Atom("ontable", ("b2",)),
        Atom("clear", ("t1",)),
        Atom("clear", ("t2",)),
    ]

    def has_goal_witness(
        goal_pairs: list[tuple[str, str]], target: tuple[str, str]
    ) -> bool:
        goals = [Atom("on", pair) for pair in goal_pairs]
        encoding = _encode(objects, current, goals, schema)
        # every displayed goal atom must be unsatisfied
        goal_start = len(current)
        goal_channels = encoding.atom_channel_ids[goal_start : goal_start + len(goals)]
        assert bool((goal_channels == CHANNEL_UNSATISFIED).all())
        object_index = {name: index for index, name in enumerate(objects)}
        u, v = object_index[target[0]], object_index[target[1]]
        pair_rows = (encoding.pair_objects == torch.tensor([u, v])).all(dim=1)
        if not bool(pair_rows.any()):
            return False
        pair_id = int(pair_rows.nonzero()[0].item())
        return bool((encoding.composition_triplets[:, 0] == pair_id).any())

    goal_a = [("b1", "t1"), ("b2", "t2")]
    goal_b = [("b1", "t2"), ("b2", "t1")]

    assert has_goal_witness(goal_a, ("b1", "t1"))
    assert has_goal_witness(goal_a, ("b2", "t2"))
    assert not has_goal_witness(goal_b, ("b1", "t2"))
    assert not has_goal_witness(goal_b, ("b2", "t1"))


# --------------------------------------------------------------------------
# R9: atom-to-pair maps
# --------------------------------------------------------------------------


def test_atom_pair_occurrence_maps_are_global_occurrence_indices() -> None:
    schema = _schema([("on", 2), ("clear", 1)])
    encoding = _encode(
        ["a", "b", "c"],
        [Atom("clear", ("a",)), Atom("on", ("b", "c"))],
        None,
        schema,
    )
    # the "on" atom is the second atom (index 1), occupying occurrences [1, 3)
    on_start = int(encoding.atom_offsets[1])
    assert on_start == 1
    assert set(encoding.atom_pair_occurrence_i.tolist()) <= {on_start, on_start + 1}
    assert set(encoding.atom_pair_occurrence_j.tolist()) <= {on_start, on_start + 1}
    # 2 ordered position pairs (i=0,j=1) and (i=1,j=0)
    assert encoding.atom_pair_ids.numel() == 2


# --------------------------------------------------------------------------
# R10: goal-free / supplied-empty vs omitted
# --------------------------------------------------------------------------


def test_goal_free_mode_emits_no_goal_atoms_and_no_counterpart_field() -> None:
    schema = _schema([("on", 2)])
    encoding = _encode(
        ["a", "b"], [Atom("on", ("a", "b"))], None, schema, exact_tuple_exchange=True
    )
    assert bool((encoding.atom_channel_ids != CHANNEL_SATISFIED).all())
    assert bool((encoding.atom_channel_ids != CHANNEL_UNSATISFIED).all())
    assert encoding.counterpart_occurrence_ids is None
    assert bool(encoding.goal_available[0].item()) is False


def test_supplied_empty_vs_omitted_goal_same_atoms_different_flag() -> None:
    schema = _schema([("on", 2)])
    current = [Atom("on", ("a", "b"))]
    omitted = _encode(["a", "b"], current, None, schema)
    supplied_empty = _encode(["a", "b"], current, [], schema)

    assert omitted.num_atoms == supplied_empty.num_atoms
    assert torch.equal(omitted.atom_predicate_ids, supplied_empty.atom_predicate_ids)
    assert torch.equal(omitted.atom_channel_ids, supplied_empty.atom_channel_ids)
    assert bool(omitted.goal_available[0].item()) is False
    assert bool(supplied_empty.goal_available[0].item()) is True


# --------------------------------------------------------------------------
# R11: optional exact-tuple counterpart exchange
# --------------------------------------------------------------------------


def test_exact_tuple_exchange_links_current_to_satisfied_goal_only() -> None:
    schema = _schema([("on", 2), ("clear", 1)])
    current = [Atom("on", ("a", "b")), Atom("clear", ("a",))]
    goals = [Atom("on", ("a", "b"))]
    encoding = _encode(["a", "b"], current, goals, schema, exact_tuple_exchange=True)
    on_state_start = int(encoding.atom_offsets[0])
    on_goal_index = 2  # after on(state), clear(state)
    on_goal_start = int(encoding.atom_offsets[on_goal_index])
    assert encoding.atom_channel_ids[on_goal_index].item() == CHANNEL_SATISFIED

    counterpart = encoding.counterpart_occurrence_ids
    assert counterpart is not None
    assert counterpart[on_state_start].item() == on_goal_start
    assert counterpart[on_state_start + 1].item() == on_goal_start + 1
    # clear(a) is a non-goal current atom: no counterpart
    clear_start = int(encoding.atom_offsets[1])
    assert counterpart[clear_start].item() == -1
    # the goal atom's own occurrence is not itself linked
    assert counterpart[on_goal_start].item() == -1


def test_exact_tuple_exchange_defaults_off() -> None:
    schema = _schema([("on", 2)])
    current = [Atom("on", ("a", "b"))]
    goals = [Atom("on", ("a", "b"))]
    encoding = _encode(["a", "b"], current, goals, schema)
    assert encoding.counterpart_occurrence_ids is None


def test_unsatisfied_goal_has_no_counterpart() -> None:
    schema = _schema([("on", 2)])
    current: list[Atom] = []
    goals = [Atom("on", ("a", "b"))]
    encoding = _encode(["a", "b"], current, goals, schema, exact_tuple_exchange=True)
    assert bool((encoding.counterpart_occurrence_ids == -1).all())


# --------------------------------------------------------------------------
# R13: object types
# --------------------------------------------------------------------------


def test_object_type_ids_default_to_none() -> None:
    schema = _schema([("on", 2)])
    encoding = _encode(["a", "b"], [Atom("on", ("a", "b"))], None, schema)
    assert encoding.object_type_ids is None


def test_object_type_ids_are_resolved_per_object() -> None:
    schema = _schema([("at", 2)])
    type_schema = build_type_schema(["truck", "city"])
    encoding = _encode(
        ["t1", "c1", "c2"],
        [Atom("at", ("t1", "c1"))],
        None,
        schema,
        object_types=["truck", "city", "city"],
        type_schema=type_schema,
    )
    assert encoding.object_type_ids is not None
    assert encoding.object_type_ids.tolist() == [
        type_schema.name_to_id["truck"],
        type_schema.name_to_id["city"],
        type_schema.name_to_id["city"],
    ]


def test_star_object_is_typed_as_root_type() -> None:
    schema = _schema([("handempty", 0)])
    type_schema = build_type_schema(["block"])
    encoding = _encode(
        ["a"],
        [Atom("handempty", ())],
        None,
        schema,
        object_types=["block"],
        type_schema=type_schema,
    )
    assert encoding.num_objects == 2  # 'a' plus the synthetic star object
    assert encoding.object_type_ids.tolist() == [
        type_schema.name_to_id["block"],
        type_schema.name_to_id[ROOT_TYPE_NAME],
    ]


def test_object_types_and_type_schema_must_be_supplied_together() -> None:
    schema = _schema([("on", 2)])
    type_schema = build_type_schema(["truck"])
    with pytest.raises(ValueError, match="together"):
        encode_sparse_atom_facts(
            ["a", "b"],
            [Atom("on", ("a", "b"))],
            None,
            schema,
            object_types=["truck", "truck"],
        )
    with pytest.raises(ValueError, match="together"):
        encode_sparse_atom_facts(
            ["a", "b"], [Atom("on", ("a", "b"))], None, schema, type_schema=type_schema
        )


def test_object_types_length_must_match_objects() -> None:
    schema = _schema([("on", 2)])
    type_schema = build_type_schema(["truck"])
    with pytest.raises(ValueError, match="one entry per object"):
        encode_sparse_atom_facts(
            ["a", "b"],
            [Atom("on", ("a", "b"))],
            None,
            schema,
            object_types=["truck"],
            type_schema=type_schema,
        )


def test_unknown_object_type_name_is_rejected() -> None:
    schema = _schema([("on", 2)])
    type_schema = build_type_schema(["truck"])
    with pytest.raises(ValueError, match="unknown object type"):
        encode_sparse_atom_facts(
            ["a", "b"],
            [Atom("on", ("a", "b"))],
            None,
            schema,
            object_types=["truck", "spaceship"],
            type_schema=type_schema,
        )


def test_untyped_domain_degrades_to_single_root_type_id() -> None:
    # The "single id, not an absent field" degrade-gracefully decision: every
    # object of a domain whose only type is the implicit PDDL root still gets
    # a real (constant) object_type_ids entry.
    schema = _schema([("on", 2)])
    type_schema = build_type_schema([])
    assert type_schema.names == (ROOT_TYPE_NAME,)
    encoding = _encode(
        ["a", "b"],
        [Atom("on", ("a", "b"))],
        None,
        schema,
        object_types=[ROOT_TYPE_NAME, ROOT_TYPE_NAME],
        type_schema=type_schema,
    )
    assert encoding.object_type_ids.tolist() == [0, 0]


# --------------------------------------------------------------------------
# R12: batching / index rebasing
# --------------------------------------------------------------------------


def test_batching_produces_zero_cross_graph_pairs_and_matches_solo_encodes() -> None:
    schema = _schema([("on", 2)])

    def triangle(names: list[str]):
        a, b, c = names
        atoms = [
            Atom("on", (a, b)),
            Atom("on", (b, a)),
            Atom("on", (b, c)),
            Atom("on", (c, b)),
            Atom("on", (a, c)),
            Atom("on", (c, a)),
        ]
        return _encode(names, atoms, None, schema)

    first = triangle(["x", "y", "z"])
    second = triangle(["p", "q", "r"])
    batch = batch_sparse_atom_encodings([first, second])
    validate_sparse_atom_composition(batch, predicate_arities=schema.arities)

    assert batch.num_pairs == first.num_pairs + second.num_pairs
    assert batch.composition_triplets.size(0) == (
        first.composition_triplets.size(0) + second.composition_triplets.size(0)
    )
    object_graph = batch.object_batch
    left_graph = object_graph.index_select(0, batch.pair_objects[:, 0])
    right_graph = object_graph.index_select(0, batch.pair_objects[:, 1])
    assert bool((left_graph == right_graph).all())
    triplet_graphs = object_graph.index_select(
        0, batch.pair_objects.index_select(0, batch.composition_triplets[:, 0])[:, 0]
    )
    assert bool((torch.bincount(triplet_graphs) > 0).sum() <= 2)

    # matches encoding each graph alone
    assert torch.equal(
        batch.atom_predicate_ids,
        torch.cat([first.atom_predicate_ids, second.atom_predicate_ids]),
    )
    assert bool((batch.atom_batch[: first.num_atoms] == 0).all())
    assert bool((batch.atom_batch[first.num_atoms :] == 1).all())


def test_batching_rebases_object_carrier_and_atom_pair_indices() -> None:
    schema = _schema([("on", 2)])
    first = _encode(["a", "b"], [Atom("on", ("a", "b"))], None, schema)
    second = _encode(["c", "d"], [Atom("on", ("c", "d"))], None, schema)
    batch = batch_sparse_atom_encodings([first, second])
    validate_sparse_atom_composition(batch, predicate_arities=schema.arities)

    # object_carrier_occurrence_ids must still point at exactly one occurrence
    # per object, ordered by (global) object id.
    carried_objects = batch.atom_args.index_select(
        0, batch.object_carrier_occurrence_ids
    )
    assert torch.equal(carried_objects, torch.arange(batch.num_objects))


def test_batching_single_encoding_is_identity() -> None:
    schema = _schema([("on", 2)])
    solo = _encode(["a", "b"], [Atom("on", ("a", "b"))], None, schema)
    batched = batch_sparse_atom_encodings([solo])
    assert batched is solo


def test_batch_sparse_atom_encodings_requires_nonempty_list() -> None:
    with pytest.raises(ValueError):
        batch_sparse_atom_encodings([])


def test_batching_concatenates_object_type_ids_without_offset() -> None:
    # object_type_ids indexes a *shared, domain-scoped* type vocabulary, not
    # a per-graph local object array, so batching must concatenate the raw
    # values verbatim -- unlike atom_args/pair_objects, which get rebased by
    # each graph's running object offset.
    schema = _schema([("on", 2)])
    type_schema = build_type_schema(["truck", "city"])
    first = _encode(
        ["a", "b"],
        [Atom("on", ("a", "b"))],
        None,
        schema,
        object_types=["truck", "city"],
        type_schema=type_schema,
    )
    second = _encode(
        ["c", "d"],
        [Atom("on", ("c", "d"))],
        None,
        schema,
        object_types=["city", "truck"],
        type_schema=type_schema,
    )
    batch = batch_sparse_atom_encodings([first, second])
    validate_sparse_atom_composition(
        batch, predicate_arities=schema.arities, num_object_types=len(type_schema.names)
    )
    assert batch.object_type_ids.tolist() == [
        type_schema.name_to_id["truck"],
        type_schema.name_to_id["city"],
        type_schema.name_to_id["city"],
        type_schema.name_to_id["truck"],
    ]


def test_batching_rejects_mixed_object_type_presence() -> None:
    schema = _schema([("on", 2)])
    type_schema = build_type_schema(["truck"])
    typed = _encode(
        ["a", "b"],
        [Atom("on", ("a", "b"))],
        None,
        schema,
        object_types=["truck", "truck"],
        type_schema=type_schema,
    )
    untyped = _encode(["c", "d"], [Atom("on", ("c", "d"))], None, schema)
    with pytest.raises(ValueError, match="object_type_ids"):
        batch_sparse_atom_encodings([typed, untyped])


# --------------------------------------------------------------------------
# Object renaming invariance
# --------------------------------------------------------------------------


def test_object_renaming_produces_isomorphic_topology() -> None:
    schema = _schema([("on", 2), ("clear", 1)])

    def scenario(names: list[str]):
        a, b, c = names
        current = [
            Atom("on", (a, b)),
            Atom("on", (b, a)),
            Atom("on", (b, c)),
            Atom("on", (c, b)),
            Atom("clear", (a,)),
        ]
        return _encode(names, current, None, schema)

    first = scenario(["x1", "y1", "z1"])
    second = scenario(["alpha", "beta", "gamma"])

    assert first.num_atoms == second.num_atoms
    assert first.num_objects == second.num_objects
    assert first.num_pairs == second.num_pairs
    assert torch.equal(first.pair_objects, second.pair_objects)
    assert torch.equal(first.composition_triplets, second.composition_triplets)
    assert torch.equal(first.atom_channel_ids, second.atom_channel_ids)
    assert torch.equal(first.atom_predicate_ids, second.atom_predicate_ids)


# --------------------------------------------------------------------------
# Goal relabeling after a simulated action
# --------------------------------------------------------------------------


def test_goal_relabeling_leaves_pairs_and_triplets_unchanged() -> None:
    """See the architecture note's "Search and static facts" paragraph:
    relabeling a goal's satisfied/unsatisfied status changes its channel and
    routing, not the object pairs that tuple contributes -- this is the basis
    for reusing pair/witness indices across states in a search.
    """

    schema = _schema([("on", 2)])
    objects = ["a", "b", "c"]
    # This exactly mirrors the architecture note's worked Blocksworld
    # example: on(b, a), on(c, b) with goal on(a, c) -- one triangle {a,b,c}
    # with witness b.
    before_current = [Atom("on", ("b", "a")), Atom("on", ("c", "b"))]
    goal = [Atom("on", ("a", "c"))]
    before = _encode(objects, before_current, goal, schema)
    assert before.atom_channel_ids[len(before_current)].item() == CHANNEL_UNSATISFIED

    # Simulate an action's effect making the same tuple true, without
    # touching any other current atom, so only the goal atom's label moves.
    after_current = before_current + [Atom("on", ("a", "c"))]
    after = _encode(objects, after_current, goal, schema)
    assert after.atom_channel_ids[len(after_current)].item() == CHANNEL_SATISFIED

    assert torch.equal(before.pair_objects, after.pair_objects)
    assert torch.equal(before.composition_triplets, after.composition_triplets)


# --------------------------------------------------------------------------
# status_encoding="channel" vs "vocabulary": identical atoms and topology
# --------------------------------------------------------------------------


def test_vocabulary_and_channel_encodings_share_topology() -> None:
    channel_schema = _schema([("on", 2), ("clear", 1)])
    vocab_schema = build_predicate_schema(
        [("on", 2), ("clear", 1)], status_encoding=STATUS_ENCODING_VOCABULARY
    )
    current = [Atom("on", ("a", "b")), Atom("clear", ("a",))]
    goals = [Atom("on", ("a", "b")), Atom("on", ("b", "a"))]

    channel_encoding = _encode(["a", "b"], current, goals, channel_schema)
    vocab_encoding = _encode(["a", "b"], current, goals, vocab_schema)

    # Same atoms, same sparse topology -- only the predicate/channel
    # labelling differs.
    assert torch.equal(vocab_encoding.atom_offsets, channel_encoding.atom_offsets)
    assert torch.equal(vocab_encoding.atom_args, channel_encoding.atom_args)
    assert torch.equal(vocab_encoding.pair_objects, channel_encoding.pair_objects)
    assert torch.equal(
        vocab_encoding.composition_triplets, channel_encoding.composition_triplets
    )
    assert torch.equal(vocab_encoding.atom_pair_ids, channel_encoding.atom_pair_ids)
    assert torch.equal(
        vocab_encoding.atom_pair_occurrence_i, channel_encoding.atom_pair_occurrence_i
    )
    assert torch.equal(
        vocab_encoding.atom_pair_occurrence_j, channel_encoding.atom_pair_occurrence_j
    )
    assert torch.equal(
        vocab_encoding.object_carrier_occurrence_ids,
        channel_encoding.object_carrier_occurrence_ids,
    )

    # Only the labelling differs: vocabulary mode has a single channel...
    assert bool((vocab_encoding.atom_channel_ids == 0).all())
    # ...and its predicate id is exactly the channel-mode (predicate, channel)
    # pair folded together via schema.relation_id.
    expected_predicate_ids = torch.tensor(
        [
            vocab_schema.relation_id(base_id, channel)[0]
            for base_id, channel in zip(
                channel_encoding.atom_predicate_ids.tolist(),
                channel_encoding.atom_channel_ids.tolist(),
            )
        ]
    )
    assert torch.equal(vocab_encoding.atom_predicate_ids, expected_predicate_ids)


# --------------------------------------------------------------------------
# Contract validator
# --------------------------------------------------------------------------


def test_validator_rejects_diagonal_pair() -> None:
    schema = _schema([("on", 2)])
    encoding = _encode(["a", "b"], [Atom("on", ("a", "b"))], None, schema)
    encoding.pair_objects[0, 1] = encoding.pair_objects[0, 0]
    with pytest.raises(ValueError, match="diagonal"):
        validate_sparse_atom_composition(encoding, predicate_arities=schema.arities)


def test_validator_rejects_arity_mismatch() -> None:
    schema = _schema([("on", 2)])
    encoding = _encode(["a", "b"], [Atom("on", ("a", "b"))], None, schema)
    bad_arities = list(schema.arities)
    bad_arities[schema.name_to_id["on"]] = 3
    with pytest.raises(ValueError, match="arity"):
        validate_sparse_atom_composition(encoding, predicate_arities=bad_arities)


def test_validator_rejects_misaligned_composition_triplet() -> None:
    schema = _schema([("between", 3)])
    encoding = _encode(
        ["a", "b", "c"], [Atom("between", ("a", "b", "c"))], None, schema
    )
    assert encoding.composition_triplets.size(0) > 0
    # swap two rows' left/right columns to break alignment
    encoding.composition_triplets[0, 1], encoding.composition_triplets[0, 2] = (
        encoding.composition_triplets[0, 2].item(),
        encoding.composition_triplets[0, 1].item(),
    )
    with pytest.raises(ValueError, match="witness object"):
        validate_sparse_atom_composition(encoding, predicate_arities=schema.arities)


def test_validator_rejects_carrier_out_of_order() -> None:
    schema = _schema([("clear", 1)])
    encoding = _encode(["a", "b"], [Atom("clear", ("a",))], None, schema)
    encoding.object_carrier_occurrence_ids = (
        encoding.object_carrier_occurrence_ids.flip(0)
    )
    with pytest.raises(ValueError, match="ordered one per object"):
        validate_sparse_atom_composition(encoding, predicate_arities=schema.arities)


def test_validator_accepts_a_clean_batch() -> None:
    schema = _schema([("on", 2), ("clear", 1)])
    first = _encode(
        ["a", "b"], [Atom("on", ("a", "b")), Atom("clear", ("a",))], None, schema
    )
    second = _encode(
        ["c", "d"], [Atom("on", ("c", "d"))], [Atom("on", ("d", "c"))], schema
    )
    batch = batch_sparse_atom_encodings([first, second])
    validate_sparse_atom_composition(batch, predicate_arities=schema.arities)


def test_validator_rejects_wrong_length_object_type_ids() -> None:
    schema = _schema([("on", 2)])
    encoding = _encode(["a", "b"], [Atom("on", ("a", "b"))], None, schema)
    encoding.object_type_ids = torch.tensor([0], dtype=torch.long)
    with pytest.raises(ValueError, match="object_type_ids"):
        validate_sparse_atom_composition(encoding, predicate_arities=schema.arities)


def test_validator_rejects_out_of_range_object_type_id() -> None:
    schema = _schema([("on", 2)])
    type_schema = build_type_schema(["truck"])
    encoding = _encode(
        ["a", "b"],
        [Atom("on", ("a", "b"))],
        None,
        schema,
        object_types=["truck", "truck"],
        type_schema=type_schema,
    )
    encoding.object_type_ids = encoding.object_type_ids.clone()
    encoding.object_type_ids[0] = len(type_schema.names)  # one past the end
    with pytest.raises(ValueError, match="outside the model schema"):
        validate_sparse_atom_composition(
            encoding,
            predicate_arities=schema.arities,
            num_object_types=len(type_schema.names),
        )
