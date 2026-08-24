"""Tests for the lifted planning-task encoder (`LiftedTaskEncoder`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

try:
    import pymimir
    from pypddl.formalism import ParserOptions
    from pyyggdrasil.execution import ExecutionContext
    from pytyr.formalism.planning import Parser
    from pytyr.planning.lifted import (
        AxiomEvaluatorFactory,
        StateRepositoryFactory,
        SuccessorGeneratorFactory,
        Task,
    )
except ImportError:  # pragma: no cover - optional backend dependencies
    pymimir = None

from mifrost import batch_encodings
from mifrost.encoders.custom import (
    ActionStructure,
    Atom,
    Effect,
    Literal,
    conformance_smoke,
)
from mifrost.encoders.lifted import LiftedTaskEncoder, LiftedTaskEncoderStream

ROOT = Path(__file__).resolve().parents[2]


def _pddl_paths(domain: str, problem: str) -> tuple[Path, Path]:
    directory = ROOT / "data" / "pddl" / domain
    return directory / "domain.pddl", directory / f"{problem}.pddl"


def _pymimir_problem(domain_path: Path, problem_path: Path):
    domain = pymimir.Domain(domain_path)
    return pymimir.Problem(domain, problem_path, mode="lifted")


@dataclass(frozen=True)
class PyTyrSearch:
    """Handles needed to walk a PyTyr state space."""

    generator: Any
    repository: Any
    evaluator: Any

    def initial_state(self) -> Any:
        node = self.generator.get_initial_node(self.repository, self.evaluator)
        return node.get_state()


def _pytyr_task(domain_path: Path, problem_path: Path):
    options = ParserOptions()
    parser = Parser(str(domain_path), options)
    planning_task = parser.parse_task(str(problem_path), options)
    task = Task(planning_task)
    context = ExecutionContext(1)
    evaluator = AxiomEvaluatorFactory().create(task, context)
    repository = StateRepositoryFactory().create(task)
    generator = SuccessorGeneratorFactory().create(task, context)
    return planning_task, PyTyrSearch(generator, repository, evaluator)


@pytest.fixture
def blocks_pair():
    if pymimir is None:  # pragma: no cover - optional backend dependencies
        pytest.skip("pymimir/pytyr planning stack not available")
    domain_path, problem_path = _pddl_paths("blocks", "small")
    pymimir_problem = _pymimir_problem(domain_path, problem_path)
    planning_task, search = _pytyr_task(domain_path, problem_path)
    return (
        pymimir_problem,
        pymimir_problem.get_initial_state(),
        planning_task,
        search.initial_state(),
    )


def _load_problem(domain: str, problem: str):
    domain_path, problem_path = _pddl_paths(domain, problem)
    return _pymimir_problem(domain_path, problem_path)


# --------------------------------------------------------------------------- #
# inspection helpers


def _pyg(encoder: LiftedTaskEncoder, state: Any = None, **kwargs: Any):
    return encoder.encode_pyg(state, **kwargs)


def _edge_tuples(data: Any) -> list[tuple[str, str, str, int, int]]:
    """(source name, target name, kind, pos_a, pos_b) for every edge."""
    kinds = list(data.vocab_edge_kinds)
    names = data.node_names
    return [
        (names[src], names[dst], kinds[int(attr[0])], int(attr[1]), int(attr[2]))
        for src, dst, attr in zip(
            data.edge_index[0].tolist(),
            data.edge_index[1].tolist(),
            data.edge_attr.tolist(),
            strict=True,
        )
    ]


def _kind_counts(data: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    kinds = list(data.vocab_edge_kinds)
    for attr in data.edge_attr.tolist():
        kind = kinds[int(attr[0])]
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _role_counts(data: Any) -> dict[int, int]:
    counts: dict[int, int] = {}
    for role in data.x_ids[:, 0].tolist():
        counts[int(role)] = counts.get(int(role), 0) + 1
    return counts


def _structure_edge_totals(view: Any) -> dict[str, int]:
    """Expected lifted-literal edge counts derived from action_structures()."""
    totals = {
        "pre": 0,
        "pre_arg": 0,
        "eff_add": 0,
        "eff_del": 0,
        "eff_add_arg": 0,
        "eff_del_arg": 0,
    }
    for structure in view.action_structures():
        for literal in structure.precondition:
            totals["pre"] += 1
            totals["pre_arg"] += sum(
                argument.startswith("?") for argument in literal.atom.args
            )
        for effect in structure.effects:
            for literal in effect.literals:
                polarity = "add" if literal.positive else "del"
                totals[f"eff_{polarity}"] += 1
                totals[f"eff_{polarity}_arg"] += sum(
                    argument.startswith("?") for argument in literal.atom.args
                )
    return totals


# --------------------------------------------------------------------------- #
# hand-checks


def test_blocks_small_hand_checks(blocks_pair) -> None:
    problem, state, _, _ = blocks_pair
    encoder = LiftedTaskEncoder(problem)
    data = _pyg(encoder, state)
    view = encoder.view

    structures = {structure.name: structure for structure in view.action_structures()}
    parameters_per_action = {
        info.name: len(structures[info.name].parameters) for info in view.action_schemas
    }
    expected_nodes = {
        0: len(view.objects),
        1: len(view.predicates),
        2: len(view.action_schemas),
        3: sum(parameters_per_action.values()),
    }

    assert tuple(data.x_ids.shape)[1] == 5
    assert _role_counts(data) == expected_nodes
    assert data.num_nodes == (
        len(view.objects)
        + len(view.predicates)
        + len(view.action_schemas)
        + sum(parameters_per_action.values())
    )

    totals = _structure_edge_totals(view)
    counts = _kind_counts(data)
    assert counts["param_of"] == sum(parameters_per_action.values())
    assert counts["pre"] == totals["pre"]
    assert counts["pre_arg"] == totals["pre_arg"]
    assert counts["eff_add"] == totals["eff_add"]
    assert counts["eff_del"] == totals["eff_del"]
    assert counts["eff_add_arg"] == totals["eff_add_arg"]
    assert counts["eff_del_arg"] == totals["eff_del_arg"]

    # blocks/small concrete numbers.
    assert expected_nodes == {0: 2, 1: 7, 2: 4, 3: 6}
    assert counts["pre"] == 15
    assert counts["param_of"] == 6
    assert counts["pre_arg"] == 14
    assert counts["eff_add"] == 9
    assert counts["eff_del"] == 9
    assert counts["eff_add_arg"] == 8
    assert counts["eff_del_arg"] == 8

    edges = _edge_tuples(data)
    assert ("clear", "pick-up", "pre", 0, 0) in edges
    assert ("holding", "stack", "eff_del", 0, 0) in edges
    assert ("stack?a0", "stack", "param_of", 0, 0) in edges
    assert ("stack?a1", "clear", "pre_arg", 0, 1) in edges
    assert ("unstack?a1", "on", "eff_del_arg", 1, 1) in edges


def test_goal_edges_match_goal_literals_with_polarity(blocks_pair) -> None:
    problem, state, _, _ = blocks_pair
    encoder = LiftedTaskEncoder(problem)
    data = _pyg(encoder, state)

    goals = encoder.view.goal_literals(state)
    edges = _edge_tuples(data)
    goal_edges = [edge for edge in edges if edge[2] in ("goal_pos", "goal_neg")]
    expected = []
    for literal in goals:
        kind = "goal_pos" if literal.positive else "goal_neg"
        for position, argument in enumerate(literal.atom.args):
            expected.append((literal.atom.predicate, argument, kind, position, 0))
    assert sorted(goal_edges) == sorted(expected)

    # blocks/small: single positive goal (on a b).
    assert sorted(goal_edges) == [
        ("on", "a", "goal_pos", 0, 0),
        ("on", "b", "goal_pos", 1, 0),
    ]


def test_negative_goal_literals_use_goal_neg_kind(blocks_pair, monkeypatch) -> None:
    problem, state, _, _ = blocks_pair
    encoder = LiftedTaskEncoder(problem)
    monkeypatch.setattr(
        encoder.view,
        "_goal_literals",
        (
            Literal(Atom("on", ("a", "b")), True),
            Literal(Atom("holding", ("a",)), False),
        ),
    )
    edges = _edge_tuples(_pyg(encoder, state))
    assert ("on", "a", "goal_pos", 0, 0) in edges
    assert ("on", "b", "goal_pos", 1, 0) in edges
    assert ("holding", "a", "goal_neg", 0, 0) in edges
    assert not any(edge[2] == "goal_neg" and edge[0] == "on" for edge in edges)


def test_conditional_effect_groups_get_sequential_indices(
    blocks_pair, monkeypatch
) -> None:
    problem, state, _, _ = blocks_pair
    encoder = LiftedTaskEncoder(problem)
    patched = tuple(
        ActionStructure(
            name="pick-up",
            arity=structure.arity,
            parameters=structure.parameters,
            precondition=structure.precondition,
            effects=(
                *structure.effects,
                Effect(
                    (
                        Literal(Atom("holding", ("?a0",)), True),
                        Literal(Atom("number", ("?a0",)), True),
                    ),
                    (
                        Literal(Atom("clear", ("?a0",)), True),
                        Literal(Atom("ontable", ("?a0",)), False),
                    ),
                ),
            ),
        )
        if structure.name == "pick-up"
        else structure
        for structure in encoder.view.action_structures()
    )
    monkeypatch.setattr(encoder.view, "_action_structures", patched)
    edges = _edge_tuples(_pyg(encoder, state))

    cond = sorted((edge[0], edge[3]) for edge in edges if edge[2] == "eff_cond")
    assert cond == [("holding", 1), ("number", 1)]
    group_one = {(edge[0], edge[2], edge[3]) for edge in edges if edge[1] == "pick-up"}
    assert ("ontable", "eff_del", 1) in group_one
    assert ("clear", "eff_add", 1) in group_one
    unconditional = {
        (edge[0], edge[2], edge[3])
        for edge in edges
        if edge[1] == "pick-up" and edge[2] != "eff_cond"
    }
    assert ("holding", "eff_add", 0) in unconditional
    assert all(kind != "eff_cond" for _, kind, _ in unconditional)


# --------------------------------------------------------------------------- #
# option flags


def test_include_parameters_false_keeps_literal_structure(blocks_pair) -> None:
    problem, state, _, _ = blocks_pair
    full = _kind_counts(_pyg(LiftedTaskEncoder(problem), state))
    reduced_encoder = LiftedTaskEncoder(problem, include_parameters=False)
    data = _pyg(reduced_encoder, state)
    counts = _kind_counts(data)

    for kind in ("param_of", "pre_arg", "eff_add_arg", "eff_del_arg"):
        assert kind not in counts
    assert counts["pre"] == full["pre"]
    assert counts["eff_add"] == full["eff_add"]
    assert counts["eff_del"] == full["eff_del"]
    assert _role_counts(data) == {0: 2, 1: 7, 2: 4}


def test_state_facts_flag_controls_fact_edges_and_state_dependence(
    blocks_pair,
) -> None:
    problem, state, _, _ = blocks_pair
    successor = next(iter(state.generate_applicable_actions())).apply(state)

    default = LiftedTaskEncoder(problem)
    assert "fact" not in _kind_counts(_pyg(default, state))
    assert default.encode(state).dumps() == default.encode(successor).dumps()

    factual = LiftedTaskEncoder(problem, include_state_facts=True)
    data = _pyg(factual, state)
    view = factual.view
    facts = (*view.static_facts, *view.state_facts(state))
    # Every argument contributes an edge; nullary facts contribute one
    # predicate self-loop each.
    expected = sum(len(atom.args) if atom.args else 1 for atom in facts)
    assert _kind_counts(data)["fact"] == expected

    fact_edges = [edge for edge in _edge_tuples(data) if edge[2] == "fact"]
    assert ("ontable", "a", "fact", 0, 0) in fact_edges
    assert ("handempty", "handempty", "fact", 0, 0) in fact_edges
    assert factual.encode(state).dumps() != factual.encode(successor).dumps()


def test_goal_flag_off_drops_goal_edges(blocks_pair) -> None:
    problem, state, _, _ = blocks_pair
    encoder = LiftedTaskEncoder(problem, include_goal=False)
    counts = _kind_counts(_pyg(encoder, state))
    assert "goal_pos" not in counts and "goal_neg" not in counts


# --------------------------------------------------------------------------- #
# channels and metadata


def test_channel_spot_checks(blocks_pair) -> None:
    problem, state, _, _ = blocks_pair
    encoder = LiftedTaskEncoder(problem)
    data = _pyg(encoder, state)

    names = data.node_names
    rows = {name: row for name, row in zip(names, data.x_ids.tolist())}

    assert rows["a"] == [0.0, 0.0, 0.0, 0.0, 0.0]

    predicates = list(data.vocab_predicates)
    on_row = rows["on"]
    assert on_row == [1.0, predicates.index("on") + 1, 2.0, 0.0, 1.0]
    number_row = rows["number"]
    assert number_row == [1.0, predicates.index("number") + 1, 1.0, 0.0, 0.0]

    actions = list(data.vocab_actions)
    stack_row = rows["stack"]
    assert stack_row == [2.0, actions.index("stack") + 1, 2.0, 0.0, 0.0]

    parameter_row = rows["stack?a1"]
    assert parameter_row == [3.0, actions.index("stack") + 1, 1.0, 0.0, 0.0]
    unstack_parameter_row = rows["unstack?a0"]
    assert unstack_parameter_row == [
        3.0,
        actions.index("unstack") + 1,
        0.0,
        0.0,
        0.0,
    ]

    attrs = encoder.encode(state).graph_attrs
    assert attrs["vocab_roles"] == ["object", "predicate", "action", "parameter"]
    assert attrs["vocab_categories"] == ["static", "fluent", "derived"]
    assert attrs["include_parameters"] == 1
    assert attrs["include_goal"] == 1
    assert attrs["include_state_facts"] == 0
    assert attrs["vocab_predicates"] == predicates
    assert attrs["vocab_actions"] == actions


def test_export_node_names_false_omits_names(blocks_pair) -> None:
    problem, state, _, _ = blocks_pair
    encoder = LiftedTaskEncoder(problem, export_node_names=False)
    data = _pyg(encoder, state)
    assert not getattr(data, "node_names", [])


# --------------------------------------------------------------------------- #
# backends and batching


@pytest.mark.parametrize(
    ("domain", "problem"),
    [
        ("blocks", "small"),
        ("delivery", "instance_2x2_p-2_0"),
        ("gripper", "gripper_b-5"),
    ],
)
def test_cross_backend_parity(domain: str, problem: str) -> None:
    if pymimir is None:  # pragma: no cover - optional backend dependencies
        pytest.skip("pymimir/pytyr planning stack not available")
    domain_path, problem_path = _pddl_paths(domain, problem)
    try:
        pymimir_problem = _pymimir_problem(domain_path, problem_path)
        pymimir_state = pymimir_problem.get_initial_state()
    except Exception as exc:  # pragma: no cover - depends on optional stack
        pytest.skip(f"pymimir cannot parse {domain}/{problem}: {exc}")
    try:
        planning_task, search = _pytyr_task(domain_path, problem_path)
        pytyr_state = search.initial_state()
    except Exception as exc:
        pytest.skip(f"pytyr cannot parse {domain}/{problem}: {exc}")

    pm_encoding = LiftedTaskEncoder(pymimir_problem).encode(pymimir_state)
    pt_encoding = LiftedTaskEncoder(planning_task).encode(pytyr_state)
    assert pm_encoding.dumps() == pt_encoding.dumps()


def test_batch_within_problem_matches_manual_collation(blocks_pair) -> None:
    problem, state, _, _ = blocks_pair
    encoder = LiftedTaskEncoder(problem)
    successor = next(iter(state.generate_applicable_actions())).apply(state)

    batched = encoder.encode_batch([state, successor])
    manual = batch_encodings([encoder.encode(state), encoder.encode(successor)])
    assert batched.dumps() == manual.dumps()
    assert batched.num_graphs == 2
    assert batched.num_nodes == 2 * encoder.encode(state).num_nodes


def test_same_domain_cross_problem_batching(blocks_pair) -> None:
    problem, state, _, _ = blocks_pair
    smedium_problem = _load_problem("blocks", "smedium")

    small_encoding = LiftedTaskEncoder(problem).encode(state)
    smedium_encoding = LiftedTaskEncoder(smedium_problem).encode(
        smedium_problem.get_initial_state()
    )
    combined = batch_encodings([small_encoding, smedium_encoding])
    assert combined.num_graphs == 2
    assert combined.num_nodes == small_encoding.num_nodes + smedium_encoding.num_nodes


def test_stream_variant_matches_direct_encoding(blocks_pair) -> None:
    problem, state, _, _ = blocks_pair
    successor = next(iter(state.generate_applicable_actions())).apply(state)
    encoder = LiftedTaskEncoder(problem)

    stream = encoder.stream()
    assert isinstance(stream, LiftedTaskEncoderStream)
    stream.append(state)
    stream.append(successor)
    flushed = stream.flush()
    direct = batch_encodings([encoder.encode(state), encoder.encode(successor)])
    assert flushed.dumps() == direct.dumps()


# --------------------------------------------------------------------------- #
# lanes and kwargs


def test_lane_rejection_names_the_encoder(blocks_pair) -> None:
    problem, state, _, _ = blocks_pair
    encoder = LiftedTaskEncoder(problem)
    native_goals = problem.get_goal_condition().get_literals()
    native_actions = list(state.generate_applicable_actions())

    with pytest.raises(ValueError, match="LiftedTaskEncoder.*goals"):
        encoder.encode(state, goals=native_goals)
    with pytest.raises(ValueError, match="LiftedTaskEncoder.*actions"):
        encoder.encode(state, actions=native_actions)
    with pytest.raises(ValueError, match="LiftedTaskEncoder.*subgoal_layers"):
        encoder.encode(state, subgoal_layers=[native_goals])
    with pytest.raises(ValueError, match="LiftedTaskEncoder.*history_subgoals"):
        encoder.encode(state, history_subgoals=[(1, native_goals)])
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        encoder.encode(state, mystery_lane=[1, 2, 3])


def test_empty_lanes_are_accepted(blocks_pair) -> None:
    problem, state, _, _ = blocks_pair
    encoder = LiftedTaskEncoder(problem)
    baseline = encoder.encode(state)
    assert encoder.encode(state, goals=None, actions=[]).dumps() == baseline.dumps()


# --------------------------------------------------------------------------- #
# GNN conformance


def test_conformance_smoke_on_encode_pyg(blocks_pair) -> None:
    pytest.importorskip("torch_geometric")
    problem, state, _, _ = blocks_pair
    encoder = LiftedTaskEncoder(problem)
    result = conformance_smoke(encoder.encode_pyg(state))
    assert "skipped" not in result
    assert set(result) == {"gcn", "gatv2"}


# --------------------------------------------------------------------------- #
# adversarial-review hardening

CONSTANTS_DOMAIN = (
    """
(define (domain const-lifted)
    (:requirements :typing)
    (:types
        rover - object
        loc - object
    )
    (:constants
        home - loc
    )
    (:predicates
        (at ?r - rover ?l - loc)
    )
    (:action move
        :parameters (?r - rover ?from ?to - loc)
        :precondition (at ?r ?from)
        :effect (and (not (at ?r ?from)) (at ?r ?to))
    )
)
""".strip()
    + "\n"
)

CONSTANTS_PROBLEM = (
    """
(define (problem const-lifted-p)
    (:domain const-lifted)
    (:objects r1 - rover)
    (:init (at r1 home))
    (:goal (at r1 home))
)
""".strip()
    + "\n"
)

NULLARY_DOMAIN = (
    """
(define (domain nullary-goal)
    (:requirements :strips :derived-predicates)
    (:predicates (step) (done))
    (:derived (done) (step))
    (:action act
        :parameters ()
        :precondition (step)
        :effect (and (not (step)))
    )
)
""".strip()
    + "\n"
)

NULLARY_PROBLEM = (
    """
(define (problem nullary-goal-p)
    (:domain nullary-goal)
    (:init (step))
    (:goal (done))
)
""".strip()
    + "\n"
)


def _scratch_problem(tmp_path: Path, domain_text: str, problem_text: str):
    if pymimir is None:  # pragma: no cover - optional backend dependencies
        pytest.skip("pymimir/pytyr planning stack not available")
    domain_path = tmp_path / "domain.pddl"
    problem_path = tmp_path / "problem.pddl"
    domain_path.write_text(domain_text, encoding="utf-8")
    problem_path.write_text(problem_text, encoding="utf-8")
    return _pymimir_problem(domain_path, problem_path)


def test_domain_constants_in_goal_become_object_nodes(tmp_path) -> None:
    """Goal literals referencing (:constants ...) names must not crash."""

    problem = _scratch_problem(tmp_path, CONSTANTS_DOMAIN, CONSTANTS_PROBLEM)
    encoder = LiftedTaskEncoder(problem)
    state = problem.get_initial_state()
    assert "home" not in encoder.view.objects  # precondition of this test

    data = _pyg(encoder, state)
    names = list(data.node_names)
    assert "home" in names  # constant appended as an object-role node
    home_row = names.index("home")
    assert int(data.x_ids[home_row][0]) == 0  # role object
    # Domain constants come after all problem objects.
    assert names.index("r1") < home_row

    edges = _edge_tuples(data)
    assert ("at", "r1", "goal_pos", 0, 0) in edges
    assert ("at", "home", "goal_pos", 1, 0) in edges


def test_domain_constants_in_facts_become_object_nodes(tmp_path) -> None:
    problem = _scratch_problem(tmp_path, CONSTANTS_DOMAIN, CONSTANTS_PROBLEM)
    encoder = LiftedTaskEncoder(problem, include_state_facts=True)
    data = _pyg(encoder, problem.get_initial_state())
    edges = _edge_tuples(data)
    assert ("at", "r1", "fact", 0, 0) in edges
    assert ("at", "home", "fact", 1, 0) in edges


def test_nullary_goal_literal_emits_predicate_self_loop(tmp_path) -> None:
    """Arity-0 goals anchor as self-loops instead of vanishing."""

    problem = _scratch_problem(tmp_path, NULLARY_DOMAIN, NULLARY_PROBLEM)
    encoder = LiftedTaskEncoder(problem)
    edges = _edge_tuples(_pyg(encoder, problem.get_initial_state()))
    assert ("done", "done", "goal_pos", 0, 0) in edges


def test_nullary_fact_emits_predicate_self_loop(tmp_path) -> None:
    problem = _scratch_problem(tmp_path, NULLARY_DOMAIN, NULLARY_PROBLEM)
    encoder = LiftedTaskEncoder(problem, include_state_facts=True)
    edges = _edge_tuples(_pyg(encoder, problem.get_initial_state()))
    # Both nullary facts hold initially ((done) is derived from (step)).
    assert ("done", "done", "fact", 0, 0) in edges
    assert ("step", "step", "fact", 0, 0) in edges


def test_explicit_goals_lane_is_rejected_even_when_empty(blocks_pair) -> None:
    """Goals come from the problem: ANY explicit goals lane violates that."""

    problem, state, _, _ = blocks_pair
    encoder = LiftedTaskEncoder(problem)
    with pytest.raises(ValueError, match="LiftedTaskEncoder.*goals"):
        encoder.encode(state, goals=[])
    with pytest.raises(ValueError, match="LiftedTaskEncoder.*goals"):
        encoder.encode_batch([state], goals=[[]])


def test_history_max_steps_is_rejected(blocks_pair) -> None:
    problem, state, _, _ = blocks_pair
    encoder = LiftedTaskEncoder(problem)
    with pytest.raises(ValueError, match="LiftedTaskEncoder.*history_max_steps"):
        encoder.encode(state, history_max_steps=3)


def test_constructing_from_pymimir_domain_names_the_fix(blocks_pair) -> None:
    del blocks_pair
    if pymimir is None:  # pragma: no cover - optional backend dependencies
        pytest.skip("pymimir/pytyr planning stack not available")
    domain_path, _ = _pddl_paths("blocks", "small")
    domain = pymimir.Domain(domain_path)
    with pytest.raises(TypeError, match="Problem"):
        LiftedTaskEncoder(domain)
    with pytest.raises(TypeError, match="Problem"):
        from mifrost.encoders.object_feature import ObjectFeatureEncoder

        ObjectFeatureEncoder(domain)
