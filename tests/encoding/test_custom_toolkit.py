"""Tests for the pure-Python custom-encoder toolkit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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

from mifrost.encoders.custom import (
    ActionInfo,
    Atom,
    CustomGraphEncoder,
    CustomStream,
    EdgeSink,
    GraphWriter,
    Literal,
    NodeTable,
    PredicateInfo,
    StateView,
    Vocabulary,
)

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


def _load_smedium_pair(pymimir_problem: Any) -> tuple[Any, Any]:
    """Return a pymimir (problem, state) pair whose initial state has binary facts."""

    del pymimir_problem
    domain_path, problem_path = _pddl_paths("blocks", "smedium")
    problem = _pymimir_problem(domain_path, problem_path)
    return problem, problem.get_initial_state()


class EchoEncoder(CustomGraphEncoder):
    """Toy encoder: object nodes, fact nodes with arg edges, goal nodes."""

    def encode_state(
        self,
        out: GraphWriter,
        state: Any,
        *,
        goals: Any = None,
        actions: Any = None,
        subgoal_layers: Any = None,
        history_subgoals: Any = None,
        history_max_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        view = out.view
        predicates = out.vocabulary("predicates")
        for obj in view.objects:
            out.add_node(("object", obj), role="object", channels=(1,), name=obj)
        for atom in view.state_facts(state):
            predicates.id_for(atom.predicate)
            fact_id = out.add_node(
                ("fact", atom.display), role="fact", channels=(2,), name=atom.display
            )
            for pos, arg in enumerate(atom.args):
                obj_id = out.add_node(("object", arg), role="object", name=arg)
                out.add_both(obj_id, fact_id, f"arg{pos}", f"arg{pos}_rev", pos, pos)
        literals = view.goal_literals(state) if goals is None else goals
        for literal in literals:
            predicates.id_for(literal.atom.predicate)
            goal_id = out.add_node(
                ("goal", literal.atom.display, literal.positive),
                role="goal",
                channels=(3,),
            )
            for pos, arg in enumerate(literal.atom.args):
                obj_id = out.add_node(("object", arg), role="object", name=arg)
                out.add_both(obj_id, goal_id, "goal_arg", "goal_arg_rev", pos, pos)
        out.set_vocab_attr("predicates")


# --------------------------------------------------------------------------- #
# tables


def test_vocabulary_ids_names_and_freeze() -> None:
    vocab = Vocabulary()
    assert len(vocab) == 0
    assert vocab.id_for("b") == 0
    assert vocab.id_for("a") == 1
    assert vocab.id_for("b") == 0
    assert len(vocab) == 2
    assert vocab.names() == ["b", "a"]
    assert vocab.name_for(1) == "a"
    vocab.freeze()
    vocab.freeze()
    assert vocab.id_for("b") == 0
    with pytest.raises(ValueError, match="frozen"):
        vocab.id_for("c")


def test_node_table_interning_padding_and_roles() -> None:
    table = NodeTable()
    first = table.id_for("k1", role="object", channels=(1,), name="obj")
    second = table.id_for("k1", role="fact", channels=(9, 9, 9), name="ignored")
    assert first == second == 0
    fact = table.id_for(("fact", "x"), role="fact", channels=(4, 5))
    goal = table.id_for("k2", role="goal")
    assert fact == 1 and goal == 2
    assert table.count == 3
    assert table.channel_dim == 2
    assert table.roles == ["object", "fact", "goal"]
    assert table.names() is None
    array = table.to_float_array()
    assert array.dtype == np.float32
    assert array.shape == (3, 2)
    np.testing.assert_array_equal(array, [[1.0, 0.0], [4.0, 5.0], [0.0, 0.0]])
    role_ids = table.role_ids()
    assert sorted(role_ids) == [0, 1, 2]
    assert table.roles_vocabulary.names() == ["object", "fact", "goal"]
    assert table.roles_vocabulary.name_for(0) == "object"


def test_node_table_names_requires_all_rows_named() -> None:
    table = NodeTable()
    table.id_for("a", role="object", name="A")
    assert table.names() == ["A"]
    table.id_for("b", role="object")
    assert table.names() is None


def test_edge_sink_arrays_kinds_and_add_both() -> None:
    sink = EdgeSink()
    edge_index, edge_attr = sink.to_arrays()
    assert edge_index.dtype == np.int64
    assert edge_attr.dtype == np.float32
    assert edge_index.shape == (2, 0)
    assert edge_attr.shape == (0, 3)
    sink.add_both(0, 1, "arg_fwd", "arg_bwd", pos_a=2, pos_b=3)
    sink.add(1, 2, "arg_fwd", pos_a=1)
    assert len(sink.kinds) == 2
    assert sink.kinds.name_for(0) == "arg_fwd"
    assert sink.kinds.id_for("arg_bwd") == 1
    edge_index, edge_attr = sink.to_arrays()
    assert edge_index.shape == (2, 3)
    np.testing.assert_array_equal(edge_index[0], [0, 1, 1])
    np.testing.assert_array_equal(edge_index[1], [1, 0, 2])
    np.testing.assert_array_equal(
        edge_attr,
        [
            [0.0, 2.0, 3.0],
            [1.0, 3.0, 2.0],
            [0.0, 1.0, 0.0],
        ],
    )


# --------------------------------------------------------------------------- #
# state views


def test_state_view_parity_across_backends(blocks_pair) -> None:
    pymimir_problem, pymimir_state, planning_task, pytyr_state = blocks_pair
    pm_view = StateView(pymimir_problem)
    pt_view = StateView(planning_task)

    assert pm_view.backend == "pymimir"
    assert pt_view.backend == "pytyr"
    assert pm_view.objects == pt_view.objects
    assert pm_view.object_types is None and pt_view.object_types is None
    assert pm_view.predicates == pt_view.predicates
    assert pm_view.action_schemas == pt_view.action_schemas
    assert pm_view.static_facts == pt_view.static_facts
    assert pm_view.state_facts(pymimir_state) == pt_view.state_facts(pytyr_state)
    assert pm_view.goal_literals(pymimir_state) == pt_view.goal_literals(pytyr_state)
    assert pm_view.problem_name == pt_view.problem_name


def test_action_structure_parity_across_backends(blocks_pair) -> None:
    pymimir_problem, _, planning_task, _ = blocks_pair
    pm_view = StateView(pymimir_problem)
    pt_view = StateView(planning_task)

    assert pm_view.has_action_structures and pt_view.has_action_structures
    pm_structures = pm_view.action_structures()
    pt_structures = pt_view.action_structures()
    assert pm_structures == pt_structures
    assert pm_view.action_structures() is pm_structures

    pick_up = next(s for s in pm_structures if s.name == "pick-up")
    assert pick_up.parameters == ("?a0",)
    precondition_displays = sorted(
        literal.atom.display for literal in pick_up.precondition
    )
    assert "(clear ?a0)" in precondition_displays
    assert "(handempty)" in precondition_displays
    assert all(literal.positive for literal in pick_up.precondition)
    assert pick_up.effects
    for effect in pick_up.effects:
        for literal in effect.literals:
            for argument in literal.atom.args:
                assert argument == "?a0" or not argument.startswith("?")


def _domain_names() -> list[str]:
    base = ROOT / "data" / "pddl"
    return sorted(path.name for path in base.iterdir() if path.is_dir())


def _smallest_problem(domain: str) -> Path:
    """The tiniest problem file of a domain (deterministic parse cost)."""

    directory = ROOT / "data" / "pddl" / domain
    problems = sorted(
        (path for path in directory.glob("*.pddl") if path.name != "domain.pddl"),
        key=lambda path: (path.stat().st_size, path.name),
    )
    assert problems, f"no problem files under {directory}"
    return problems[0]


@pytest.mark.parametrize("domain", _domain_names())
def test_action_structure_parity_all_domains(domain: str) -> None:
    if pymimir is None:  # pragma: no cover - optional backend dependencies
        pytest.skip("pymimir/pytyr planning stack not available")
    problem_path = _smallest_problem(domain)
    domain_path = ROOT / "data" / "pddl" / domain / "domain.pddl"
    try:
        pymimir_problem = _pymimir_problem(domain_path, problem_path)
    except Exception as exc:  # noqa: BLE001 - any parser failure must skip
        pytest.skip(
            f"pymimir cannot parse data/pddl/{domain}/{problem_path.name}: "
            f"{type(exc).__name__}: {exc}"
        )
    try:
        planning_task, _search = _pytyr_task(domain_path, problem_path)
    except Exception as exc:  # noqa: BLE001 - any parser failure must skip
        pytest.skip(
            f"pytyr cannot parse data/pddl/{domain}/{problem_path.name}: "
            f"{type(exc).__name__}: {exc}"
        )
    pm_structures = StateView(pymimir_problem).action_structures()
    pt_structures = StateView(planning_task).action_structures()
    assert pm_structures == pt_structures


def test_harness_assert_backend_parity_and_summary(blocks_pair) -> None:
    from mifrost.encoders.custom import assert_backend_parity, channel_summary

    pymimir_problem, pymimir_state, planning_task, pytyr_state = blocks_pair
    from mifrost.encoders.object_feature import ObjectFeatureEncoder

    assert_backend_parity(
        ObjectFeatureEncoder,
        [
            (pymimir_problem, [pymimir_state, pymimir_state]),
            (planning_task, [pytyr_state, pytyr_state]),
        ],
    )

    encoder = ObjectFeatureEncoder(pymimir_problem)
    summary = channel_summary(encoder.encode_pyg(pymimir_state))
    assert summary["nodes"] == len(encoder.view.objects)
    if summary["edges"] == 0:
        smedium_problem, smedium_state = _load_smedium_pair(pymimir_problem)
        encoder = ObjectFeatureEncoder(smedium_problem)
        summary = channel_summary(encoder.encode_pyg(smedium_state))
    assert summary["edges"] > 0 and "edge_kinds" in summary


def test_harness_assert_backend_parity_requires_distinct_backends(
    blocks_pair,
) -> None:
    from mifrost.encoders.custom import assert_backend_parity
    from mifrost.encoders.object_feature import ObjectFeatureEncoder

    pymimir_problem, pymimir_state, _, _ = blocks_pair
    with pytest.raises(ValueError, match="two distinct backends"):
        assert_backend_parity(
            ObjectFeatureEncoder,
            [
                (pymimir_problem, [pymimir_state]),
                (pymimir_problem, [pymimir_state]),
            ],
        )


def test_graph_writer_finish_guard_and_reset(blocks_pair) -> None:
    pymimir_problem, _state, _, _ = blocks_pair
    view = StateView(pymimir_problem)
    writer = GraphWriter(view)
    writer.add_node(("object", "a"), role="object", name="a")

    encoding = writer.finish()
    assert dict(encoding.schema_flags)["custom_encoder"] is True

    with pytest.raises(RuntimeError, match="finished"):
        writer.add_node(("object", "b"), role="object", name="b")
    with pytest.raises(RuntimeError, match="finished"):
        writer.add_edge(0, 0, "arg_fwd")
    with pytest.raises(RuntimeError, match="finished"):
        writer.add_both(0, 0, "arg_fwd", "arg_bwd")
    with pytest.raises(RuntimeError, match="finished"):
        writer.finish()

    writer.reset()
    revived_id = writer.add_node(("object", "c"), role="object", name="c")
    assert revived_id == 0
    assert writer.finish().num_nodes == 1


def test_harness_conformance_smoke(blocks_pair) -> None:
    pytest.importorskip("torch_geometric")
    from mifrost.encoders.custom import conformance_smoke
    from mifrost.encoders.object_feature import ObjectFeatureEncoder

    pymimir_problem, pymimir_state, _, _ = blocks_pair
    encoder = ObjectFeatureEncoder(pymimir_problem, include_goal_flags=True)
    data = encoder.encode_pyg(pymimir_state)
    result = conformance_smoke(data)
    assert "skipped" not in result
    assert set(result) == {"gcn", "gatv2"}


def test_state_view_explicit_backend_validation(blocks_pair) -> None:
    pymimir_problem, _, planning_task, _ = blocks_pair
    assert StateView(pymimir_problem, backend="pymimir").backend == "pymimir"
    assert StateView(planning_task, backend="pytyr").backend == "pytyr"
    with pytest.raises(ValueError, match="'pymimir' or 'pytyr'"):
        StateView(pymimir_problem, backend="mimir")


def test_state_view_neutral_actions_and_literals(blocks_pair) -> None:
    pymimir_problem, pymimir_state, _, _ = blocks_pair
    view = StateView(pymimir_problem)
    actions = list(pymimir_state.generate_applicable_actions())
    atoms = view.neutral_actions(pymimir_state, actions)
    assert atoms
    schema_names = {info.name for info in view.action_schemas}
    assert all(atom.predicate in schema_names for atom in atoms)
    assert all(atom.display.startswith("(") for atom in atoms)

    native_goals = pymimir_problem.get_goal_condition().get_literals()
    neutralized = view.neutral_literals(native_goals)
    assert neutralized == view.goal_literals(pymimir_state)
    with pytest.raises(TypeError, match="goals lane leaf.*pymimir"):
        view.neutral_literals(["not-a-literal"])


def test_atom_display_forms() -> None:
    assert Atom("handempty", ()).display == "(handempty)"
    assert Atom("on", ("a", "b")).display == "(on a b)"
    literal = Literal(Atom("on", ("a", "b")), positive=False)
    assert literal.atom.display == "(on a b)"


def test_predicate_info_category_validation() -> None:
    info = PredicateInfo("on", 2, "static")
    assert (info.name, info.arity, info.category) == ("on", 2, "static")
    assert ActionInfo("stack", 2).arity == 2
    with pytest.raises(ValueError, match="category"):
        PredicateInfo("on", 2, "derived-thing")


def test_echo_encoder_parity_across_backends(blocks_pair) -> None:
    pymimir_problem, pymimir_state, planning_task, pytyr_state = blocks_pair
    pm_encoder = EchoEncoder(pymimir_problem)
    pt_encoder = EchoEncoder(planning_task)
    pm_encoding = pm_encoder.encode(pymimir_state)
    pt_encoding = pt_encoder.encode(pytyr_state)
    assert pm_encoding.dumps() == pt_encoding.dumps()
    pm_batch = pm_encoder.encode_batch([pymimir_state])
    pt_batch = pt_encoder.encode_batch([pytyr_state])
    assert pm_batch.dumps() == pt_batch.dumps()
    assert dict(pm_encoding.schema_flags)["custom_encoder"] is True


# --------------------------------------------------------------------------- #
# encoder facade


def test_echo_encoder_single_graph(blocks_pair) -> None:
    problem, state, _, _ = blocks_pair
    encoder = EchoEncoder(problem)
    encoding = encoder.encode(state)

    view = encoder.view
    facts = view.state_facts(state)
    goals = view.goal_literals(state)
    expected_nodes = len(view.objects) + len(facts) + len(goals)
    expected_edges = 2 * sum(len(atom.args) for atom in facts)
    expected_edges += 2 * sum(len(literal.atom.args) for literal in goals)

    assert encoding.num_graphs == 1
    assert encoding.num_nodes == expected_nodes
    assert encoding.num_edges == expected_edges
    assert dict(encoding.schema_flags)["custom_encoder"] is True
    attrs = encoding.graph_attrs
    used_predicates = {atom.predicate for atom in facts}
    used_predicates |= {literal.atom.predicate for literal in goals}
    assert sorted(attrs["vocab_predicates"]) == sorted(used_predicates)
    pyg = encoding.as_pyg()
    assert tuple(pyg.x_ids.shape) == (expected_nodes, 1)
    # The homo PyG converter symmetrizes directed edges unless the
    # include_reverse_edges flag is set.
    assert tuple(pyg.edge_attr.shape) == (2 * expected_edges, 3)
    assert tuple(pyg.edge_index.shape) == (2, 2 * expected_edges)


def test_echo_encoder_goal_lane_override_matches_defaults(blocks_pair) -> None:
    problem, state, _, _ = blocks_pair
    encoder = EchoEncoder(problem)
    default = encoder.encode(state)
    override = encoder.encode(state, goals=problem.get_goal_condition().get_literals())
    assert override.dumps() == default.dumps()


def test_echo_encoder_rejects_unknown_kwargs(blocks_pair) -> None:
    problem, state, _, _ = blocks_pair
    encoder = EchoEncoder(problem)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        encoder.encode(state, mystery_lane=[1, 2, 3])


def test_echo_encoder_batch_matches_manual_collation(blocks_pair) -> None:
    problem, state, _, _ = blocks_pair
    encoder = EchoEncoder(problem)
    successor = next(iter(state.generate_applicable_actions())).apply(state)
    states = [state, successor]

    from mifrost import batch_encodings

    batched = encoder.encode_batch(states)
    manual = batch_encodings([encoder.encode(one) for one in states])
    assert batched.dumps() == manual.dumps()

    assert batched.num_graphs == 2
    assert batched.node_feature_dims["node"] == 1
    assert batched.num_nodes == sum(encoder.encode(one).num_nodes for one in states)
    assert batched.num_edges == sum(encoder.encode(one).num_edges for one in states)

    pyg = encoder.encode_batch_pyg(states)
    total_nodes = sum(encoder.encode(one).num_nodes for one in states)
    assert pyg.x_ids.shape[0] == total_nodes
    assert pyg.batch.max().item() == 1


def test_custom_stream_lifecycle_matches_direct_encoding(blocks_pair) -> None:
    problem, state, _, _ = blocks_pair
    encoder = EchoEncoder(problem)
    successor = next(iter(state.generate_applicable_actions())).apply(state)
    from mifrost import batch_encodings

    stream = CustomStream(encoder)
    assert isinstance(encoder.stream(), CustomStream)

    first = stream.append(state)
    second = stream.append(successor, goals=problem.get_goal_condition().get_literals())
    assert (first, second) == (0, 1)
    flushed = stream.flush()
    direct = batch_encodings([encoder.encode(state), encoder.encode(successor)])
    assert flushed.dumps() == direct.dumps()

    with pytest.raises(ValueError, match="empty"):
        stream.flush()

    only = stream.append(state)
    stream.update(only, successor)
    stream.remove(only)
    stream.set_reuse_removed(True)
    reused = stream.append(successor)
    assert reused == only
    assert stream.flush().dumps() == encoder.encode(successor).dumps()

    with pytest.raises(KeyError):
        stream.remove(42)
