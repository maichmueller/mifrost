"""Tests for the objects-only feature-channel showcase encoder."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

try:
    import pymimir
    import torch
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
    pytest.skip("pymimir/pytyr planning stack not available", allow_module_level=True)

from mifrost.encoders.custom import Atom, Literal
from mifrost.encoders.object_feature import (
    EXPANSIONS,
    ObjectFeatureEncoder,
    ObjectFeatureEncoderStream,
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


def _pytyr_problem(domain_path: Path, problem_path: Path):
    options = ParserOptions()
    parser = Parser(str(domain_path), options)
    planning_task = parser.parse_task(str(problem_path), options)
    task = Task(planning_task)
    context = ExecutionContext(1)
    evaluator = AxiomEvaluatorFactory().create(task, context)
    repository = StateRepositoryFactory().create(task)
    generator = SuccessorGeneratorFactory().create(task, context)
    return planning_task, PyTyrSearch(generator, repository, evaluator)


@pytest.fixture(scope="module")
def blocks():
    domain_path, problem_path = _pddl_paths("blocks", "small")
    return _pymimir_problem(domain_path, problem_path)


@pytest.fixture(scope="module")
def initial(blocks):
    return blocks.get_initial_state()


@pytest.fixture(scope="module")
def stacked(blocks, initial):
    """First reachable pymimir state containing an arity->=2 fact."""

    queue: deque[Any] = deque([initial])
    seen: list[Any] = [initial]
    probe = ObjectFeatureEncoder(blocks).view
    while queue:
        current = queue.popleft()
        for action in current.generate_applicable_actions():
            successor = action.apply(current)
            if any(item == successor for item in seen):
                continue
            seen.append(successor)
            if any(len(atom.args) >= 2 for atom in probe.state_facts(successor)):
                return successor
            queue.append(successor)
    raise AssertionError("no binary fact reachable in blocks/small")  # pragma: no cover


def _x(encoder: ObjectFeatureEncoder, state: Any):
    return encoder.encode_pyg(state).x_ids


# --------------------------------------------------------------------------- #
# shapes


@pytest.mark.parametrize(
    ("unary", "counts", "goal_flags", "expected_dim"),
    [
        (False, False, False, 1),
        (True, False, False, 6),
        (False, True, False, 3),
        (False, False, True, 2),
        (True, True, False, 8),
        (True, False, True, 7),
        (False, True, True, 4),
        (True, True, True, 9),
    ],
    ids=[
        "bare",
        "unary",
        "counts",
        "goals",
        "unary+counts",
        "unary+goals",
        "counts+goals",
        "all",
    ],
)
def test_shapes_nodes_and_channel_dim(
    blocks, stacked, unary, counts, goal_flags, expected_dim
) -> None:
    encoder = ObjectFeatureEncoder(
        blocks,
        include_unary_features=unary,
        include_participation_counts=counts,
        include_goal_flags=goal_flags,
    )
    data = encoder.encode_pyg(stacked)
    num_objects = len(encoder.view.objects)
    assert tuple(data.x_ids.shape) == (num_objects, expected_dim)
    assert data.num_nodes == num_objects
    assert data.edge_attr.shape[1] == 3


# --------------------------------------------------------------------------- #
# feature semantics


def test_unary_channels_match_state_facts_hand_checked(blocks, initial) -> None:
    encoder = ObjectFeatureEncoder(blocks)
    view = encoder.view
    predicates = view.predicates
    unary = [info for info in predicates if info.arity == 1]
    # number/object are pymimir's built-in static typing predicates.
    assert [info.name for info in unary] == [
        "clear",
        "holding",
        "ontable",
        "number",
        "object",
    ]

    holds = {
        (atom.predicate, atom.args[0])
        for atom in (*view.static_facts, *view.state_facts(initial))
        if len(atom.args) == 1
    }
    x = _x(encoder, initial)
    for row, name in enumerate(view.objects):
        expected = [0]
        expected.extend(
            predicates.index(info) + 1 if (info.name, name) in holds else 0
            for info in unary
        )
        # Hand check: (ontable a) and (clear a) hold initially, (holding a)
        # does not; the vocabulary order is clear=0, handempty=1, holding=2,
        # on=3, ontable=4, number=5, object=6, and the static (object a)
        # typing fact contributes id 6 + 1.
        if name == "a":
            assert expected == [0, 1, 0, 5, 0, 7]
        np_row = x[row].tolist()
        assert np_row == [float(value) for value in expected]


def test_static_unary_facts_are_encoded(blocks) -> None:
    encoder = ObjectFeatureEncoder(blocks)
    view = encoder.view
    static_unary = {
        (atom.predicate, atom.args[0])
        for atom in view.static_facts
        if len(atom.args) == 1
    }
    assert static_unary == {("object", "a"), ("object", "b")}
    x = _x(encoder, blocks.get_initial_state())
    unary_names = [info.name for info in view.predicates if info.arity == 1]
    for predicate, obj in static_unary:
        offset = 1 + unary_names.index(predicate)
        vocab_id = view.predicates.index(
            next(info for info in view.predicates if info.name == predicate)
        )
        assert x[view.objects.index(obj)][offset] == vocab_id + 1


def test_participation_counts_match_facts(blocks, stacked) -> None:
    encoder = ObjectFeatureEncoder(
        blocks,
        include_participation_counts=True,
        include_goal_flags=True,
    )
    view = encoder.view
    non_unary = [info for info in view.predicates if info.arity != 1]
    assert [info.name for info in non_unary] == ["handempty", "on"]

    facts = (*view.static_facts, *view.state_facts(stacked))
    counts: dict[tuple[str, str], int] = {}
    for atom in facts:
        for arg in atom.args:
            counts[(atom.predicate, arg)] = counts.get((atom.predicate, arg), 0) + 1

    x = _x(encoder, stacked)
    count_offset = x.shape[1] - len(non_unary) - 1
    for row, name in enumerate(view.objects):
        expected_counts = [
            float(counts.get((info.name, name), 0)) for info in non_unary
        ]
        # Hand check: after stacking a onto b the only arity-2 fact is
        # (on a b), so both a and b participate exactly once.
        if name in {"a", "b"}:
            assert expected_counts == [0.0, 1.0]
        assert x[row][count_offset:].tolist()[:-1] == expected_counts


def test_goal_flags_cover_positive_and_negative_literals(blocks, initial) -> None:
    encoder = ObjectFeatureEncoder(blocks, include_goal_flags=True)
    x = _x(encoder, initial)
    assert x[:, -1].tolist() == [1.0, 1.0]

    # The override replaces the default goals entirely: only b appears,
    # and it appears in a NEGATED literal, which must still set the flag.
    negated = (Literal(Atom("holding", ("b",)), positive=False),)
    override = encoder.encode_pyg(initial, goals=negated)
    assert override.x_ids[:, -1].tolist() == [0.0, 1.0]


# --------------------------------------------------------------------------- #
# edges


@pytest.mark.parametrize("expansion", EXPANSIONS)
def test_edge_semantics_per_expansion(blocks, stacked, expansion) -> None:
    encoder = ObjectFeatureEncoder(blocks, expansion=expansion)
    encoding = encoder.encode(stacked)
    data = encoder.encode_pyg(stacked)

    facts = [
        atom
        for atom in (*encoder.view.static_facts, *encoder.view.state_facts(stacked))
        if len(atom.args) >= 2
    ]
    expected = {"clique": 0, "chain": 0, "star_first": 0}[expansion]
    multiplier = {
        "clique": lambda k: k * (k - 1),
        "chain": lambda k: 2 * (k - 1),
        "star_first": lambda k: 2 * (k - 1),
    }[expansion]
    expected = sum(multiplier(len(atom.args)) for atom in facts)
    assert data.num_edges == encoding.num_edges == expected
    assert dict(encoding.schema_flags)["include_reverse_edges"] is True

    objects = encoder.view.objects
    kinds = encoding.graph_attrs["vocab_edge_kinds"]
    assert kinds == [f"{expansion}_fwd", f"{expansion}_bwd"]

    src, dst = data.edge_index.tolist()
    attrs = data.edge_attr.tolist()
    pairs = {}
    for s, d, (kind_id, pos_a, pos_b) in zip(src, dst, attrs, strict=True):
        kind = kinds[int(kind_id)]
        pairs[(s, d, kind)] = (pos_a, pos_b)

    for atom in facts:
        positions = [objects.index(arg) for arg in atom.args]
        for i, pi in enumerate(positions):
            for j, pj in enumerate(positions):
                if i >= j:
                    continue
                if expansion == "chain" and j != i + 1:
                    continue
                if expansion == "star_first" and i != 0:
                    continue
                fwd_kind = f"{expansion}_fwd"
                bwd_kind = f"{expansion}_bwd"
                assert pairs[(pi, pj, fwd_kind)] == (pi, pj)
                assert pairs[(pj, pi, bwd_kind)] == (pj, pi)


def test_nullary_facts_are_skipped_but_counted(blocks, initial) -> None:
    encoder = ObjectFeatureEncoder(blocks)
    view = encoder.view
    nullary = sum(
        1
        for atom in (*view.static_facts, *view.state_facts(initial))
        if len(atom.args) == 0
    )
    assert nullary >= 1
    assert encoder.last_nullary_count == 0
    encoder.encode(initial)
    assert encoder.last_nullary_count == nullary


# --------------------------------------------------------------------------- #
# the documented gap


def test_disabling_all_features_leaves_no_unary_trace(blocks, initial) -> None:
    bare = ObjectFeatureEncoder(
        blocks,
        include_unary_features=False,
        include_participation_counts=False,
        include_goal_flags=False,
    )
    x = _x(bare, initial)
    assert tuple(x.shape) == (len(bare.view.objects), 1)
    assert not x.any(), (
        "objects-only views lose unary facts entirely; this zero channel is "
        "exactly the gap ObjectFeatureEncoder exists to close"
    )


# --------------------------------------------------------------------------- #
# cross-backend parity


def _tensor_or_none(value: Any) -> Any:
    return None if value is None else value.tolist()


@pytest.mark.parametrize("expansion", EXPANSIONS)
def test_cross_backend_parity(blocks, expansion) -> None:
    domain_path, problem_path = _pddl_paths("blocks", "small")
    planning_task, pytyr_search = _pytyr_problem(domain_path, problem_path)

    pm_encoder = ObjectFeatureEncoder(blocks, expansion=expansion)
    pt_encoder = ObjectFeatureEncoder(planning_task, expansion=expansion)
    pm = pm_encoder.encode_pyg(blocks.get_initial_state())
    pt = pt_encoder.encode_pyg(pytyr_search.initial_state())
    assert torch.equal(pm.x_ids, pt.x_ids)
    # The blocks/small root holds no arity->=2 fact, so both conversions
    # omit the edge tensors entirely.
    assert _tensor_or_none(pm.edge_index) == _tensor_or_none(pt.edge_index)
    assert _tensor_or_none(pm.edge_attr) == _tensor_or_none(pt.edge_attr)


@pytest.mark.parametrize("expansion", EXPANSIONS)
def test_cross_backend_parity_with_edges(tmp_path, expansion) -> None:
    domain_path = tmp_path / "domain.pddl"
    problem_path = tmp_path / "problem.pddl"
    domain_path.write_text(TERNARY_DOMAIN, encoding="utf-8")
    problem_path.write_text(TERNARY_PROBLEM, encoding="utf-8")
    pymimir_problem = _pymimir_problem(domain_path, problem_path)
    planning_task, pytyr_search = _pytyr_problem(domain_path, problem_path)

    pm_encoder = ObjectFeatureEncoder(pymimir_problem, expansion=expansion)
    pt_encoder = ObjectFeatureEncoder(planning_task, expansion=expansion)
    pm = pm_encoder.encode_pyg(pymimir_problem.get_initial_state())
    pt = pt_encoder.encode_pyg(pytyr_search.initial_state())
    assert torch.equal(pm.x_ids, pt.x_ids)
    assert torch.equal(pm.edge_index, pt.edge_index)
    assert torch.equal(pm.edge_attr, pt.edge_attr)
    assert list(pm.vocab_predicates) == list(pt.vocab_predicates)
    assert list(pm.vocab_edge_kinds) == list(pt.vocab_edge_kinds)


# --------------------------------------------------------------------------- #
# streaming


def test_stream_lifecycle_matches_encode_batch(blocks, initial, stacked) -> None:
    from mifrost import batch_encodings

    encoder = ObjectFeatureEncoder(blocks, include_goal_flags=True)
    stream = encoder.stream()
    assert isinstance(stream, ObjectFeatureEncoderStream)

    first = stream.append(initial)
    second = stream.append(stacked)
    assert (first, second) == (0, 1)
    flushed = stream.flush()
    direct = batch_encodings([encoder.encode(initial), encoder.encode(stacked)])
    assert flushed.dumps() == direct.dumps()

    only = stream.append(initial)
    stream.update(only, stacked)
    stream.remove(only)
    stream.set_reuse_removed(True)
    reused = stream.append(stacked)
    assert reused == only
    assert stream.flush().dumps() == encoder.encode(stacked).dumps()

    with pytest.raises(ValueError, match="empty"):
        stream.flush()


# --------------------------------------------------------------------------- #
# GNN conformance


def test_mini_gnn_conformance_forward_backward(blocks, stacked) -> None:
    torch_geometric = pytest.importorskip("torch_geometric")
    from torch_geometric.nn import GCNConv, GATv2Conv

    del torch_geometric
    encoder = ObjectFeatureEncoder(blocks, include_goal_flags=True)
    data = encoder.encode_pyg(stacked)
    x = data.x_ids.float()
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    assert edge_attr.shape[1] == 3

    torch.manual_seed(0)
    gcn = GCNConv(x.size(1), 8)
    out = gcn(x, edge_index)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    grads = [p.grad for p in gcn.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)

    torch.manual_seed(0)
    gat = GATv2Conv(x.size(1), 8, edge_dim=3)
    out = gat(x, edge_index, edge_attr)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    grads = [p.grad for p in gat.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)


# --------------------------------------------------------------------------- #
# lane / kwarg validation


def test_unknown_kwarg_raises_type_error(blocks, initial) -> None:
    encoder = ObjectFeatureEncoder(blocks)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        encoder.encode(initial, mystery_lane=[1, 2, 3])


def test_non_empty_action_lane_raises_value_error(blocks, initial) -> None:
    encoder = ObjectFeatureEncoder(blocks)
    actions = list(initial.generate_applicable_actions())
    assert actions
    with pytest.raises(ValueError, match="ObjectFeatureEncoder.*actions"):
        encoder.encode(initial, actions=actions[:1])


def test_expansion_validation(blocks) -> None:
    with pytest.raises(ValueError, match="expansion"):
        ObjectFeatureEncoder(blocks, expansion="hypercube")


# --------------------------------------------------------------------------- #
# ternary arity differentiates the expansions


TERNARY_DOMAIN = (
    """
(define (domain ternary-feature)
  (:requirements :strips)
  (:predicates (r ?x ?y ?z))
  (:action act
    :parameters (?x ?y ?z)
    :precondition (r ?x ?y ?z)
    :effect (r ?x ?y ?z)))
""".strip()
    + "\n"
)

TERNARY_PROBLEM = (
    """
(define (problem ternary-feature-p)
  (:domain ternary-feature)
  (:objects a b c d)
  (:init (r a b c))
  (:goal (r a b c)))
""".strip()
    + "\n"
)


@pytest.mark.parametrize(
    ("expansion", "expected_edges"),
    [("clique", 6), ("chain", 4), ("star_first", 4)],
)
def test_ternary_fact_expansion_counts(tmp_path, expansion, expected_edges) -> None:
    domain_path = tmp_path / "domain.pddl"
    problem_path = tmp_path / "problem.pddl"
    domain_path.write_text(TERNARY_DOMAIN, encoding="utf-8")
    problem_path.write_text(TERNARY_PROBLEM, encoding="utf-8")
    domain = pymimir.Domain(domain_path)
    problem = pymimir.Problem(domain, problem_path, mode="lifted")

    encoder = ObjectFeatureEncoder(problem, expansion=expansion)
    data = encoder.encode_pyg(problem.get_initial_state())
    assert data.num_edges == expected_edges

    objects = encoder.view.objects
    attrs = data.edge_attr.tolist()
    src, dst = data.edge_index.tolist()
    observed = {
        (objects[s], objects[d], int(pos_a), int(pos_b))
        for s, d, (_k, pos_a, pos_b) in zip(src, dst, attrs, strict=True)
    }
    if expansion == "clique":
        assert observed == {
            ("a", "b", 0, 1),
            ("b", "a", 1, 0),
            ("a", "c", 0, 2),
            ("c", "a", 2, 0),
            ("b", "c", 1, 2),
            ("c", "b", 2, 1),
        }
    elif expansion == "chain":
        assert observed == {
            ("a", "b", 0, 1),
            ("b", "a", 1, 0),
            ("b", "c", 1, 2),
            ("c", "b", 2, 1),
        }
    else:
        assert observed == {
            ("a", "b", 0, 1),
            ("b", "a", 1, 0),
            ("a", "c", 0, 2),
            ("c", "a", 2, 0),
        }


# --------------------------------------------------------------------------- #
# adversarial-review hardening


def test_draw_unary_only_state_renders_nodes(blocks, initial) -> None:
    """Edge-less graphs must not be blanked by reverse-edge filtering."""

    pytest.importorskip("matplotlib")
    from mifrost.encoders._derived_visualization import (
        derived_to_networkx,
        draw_derived,
    )

    encoder = ObjectFeatureEncoder(blocks)
    data = encoder.encode_pyg(initial)
    # Precondition of this test: the blocks/small root has unary facts only.
    assert data.num_edges == 0
    graph = derived_to_networkx(data)
    assert graph.number_of_nodes() == len(encoder.view.objects)

    ax = draw_derived(graph)
    drawn = sum(len(collection.get_offsets()) for collection in ax.collections)
    assert drawn == graph.number_of_nodes()
    labels = [text.get_text() for text in ax.texts]
    assert set(encoder.view.objects) <= set(labels)


def test_batch_of_edgeless_and_edged_states_shares_edge_kind_vocab(
    blocks, initial, stacked
) -> None:
    """vocab_edge_kinds is state-independent, so batches cannot collide."""

    encoder = ObjectFeatureEncoder(blocks)
    initial_encoding = encoder.encode(initial)
    stacked_encoding = encoder.encode(stacked)
    assert dict(initial_encoding.graph_attrs)["vocab_edge_kinds"] == [
        "clique_fwd",
        "clique_bwd",
    ]
    assert (
        initial_encoding.graph_attrs["vocab_edge_kinds"]
        == stacked_encoding.graph_attrs["vocab_edge_kinds"]
    )
    batched = encoder.encode_batch([initial, stacked])
    assert batched.graph_attrs["vocab_edge_kinds"] == ["clique_fwd", "clique_bwd"]
