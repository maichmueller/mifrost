"""Cross-backend structural parity for the public derived-graph facades."""

from __future__ import annotations

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

import mifrost
from mifrost.encoders.derived import (
    ObjectGraphEncoder,
    TransformerBiasEncoder,
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

    def initial_node(self) -> Any:
        return self.generator.get_initial_node(self.repository, self.evaluator)


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


def _backend_pair(domain: str = "blocks", problem: str = "small"):
    domain_path, problem_path = _pddl_paths(domain, problem)
    pymimir_problem = _pymimir_problem(domain_path, problem_path)
    planning_task, pytyr_search = _pytyr_problem(domain_path, problem_path)
    return pymimir_problem, planning_task, pytyr_search


def _root_states(pymimir_problem, pytyr_search):
    return pymimir_problem.get_initial_state(), pytyr_search.initial_node().get_state()


def _assert_pyg_dicts_equal(left: dict, right: dict) -> None:
    assert set(left) == set(right)
    for key in sorted(left):
        left_value, right_value = left[key], right[key]
        if torch.is_tensor(left_value) or torch.is_tensor(right_value):
            assert torch.is_tensor(left_value) and torch.is_tensor(right_value)
            assert left_value.dtype == right_value.dtype
            assert torch.equal(left_value, right_value)
        else:
            assert left_value == right_value


def _assert_structural_equality(left, right) -> None:
    assert torch.equal(left.x_ids, right.x_ids)
    assert torch.equal(left.edge_index, right.edge_index)
    assert torch.equal(left.edge_attr.float(), right.edge_attr.float())
    assert list(left.node_names) == list(right.node_names)


def test_star_graph_encoder_structural_parity() -> None:
    pymimir_problem, planning_task, pytyr_search = _backend_pair()
    pymimir_state, pytyr_state = _root_states(pymimir_problem, pytyr_search)

    pymimir_data = mifrost.StarGraphEncoder(
        pymimir_problem, backend="pymimir"
    ).encode_pyg(pymimir_state)
    pytyr_data = mifrost.StarGraphEncoder(planning_task, backend="pytyr").encode_pyg(
        pytyr_state
    )

    _assert_structural_equality(pymimir_data, pytyr_data)


def test_object_clique_encoder_structural_parity() -> None:
    pymimir_problem, planning_task, pytyr_search = _backend_pair()
    pymimir_state, pytyr_state = _root_states(pymimir_problem, pytyr_search)

    pymimir_data = ObjectGraphEncoder(
        pymimir_problem, atom_expansion="clique", backend="pymimir"
    ).encode_pyg(pymimir_state)
    pytyr_data = ObjectGraphEncoder(
        planning_task, atom_expansion="clique", backend="pytyr"
    ).encode_pyg(pytyr_state)

    _assert_structural_equality(pymimir_data, pytyr_data)


def test_transformer_bias_encoder_spd_parity() -> None:
    pymimir_problem, planning_task, pytyr_search = _backend_pair()
    pymimir_state, pytyr_state = _root_states(pymimir_problem, pytyr_search)

    pymimir_data = TransformerBiasEncoder(
        pymimir_problem, backend="pymimir"
    ).encode_pyg(pymimir_state)
    pytyr_data = TransformerBiasEncoder(planning_task, backend="pytyr").encode_pyg(
        pytyr_state
    )

    assert torch.equal(pymimir_data.x_ids, pytyr_data.x_ids)
    assert torch.equal(pymimir_data.spd_src, pytyr_data.spd_src)
    assert torch.equal(pymimir_data.spd_dst, pytyr_data.spd_dst)
    assert torch.equal(pymimir_data.spd_dist, pytyr_data.spd_dist)


@pytest.mark.parametrize(
    ("facade_name", "kwargs"),
    (
        ("StarGraphEncoder", {}),
        ("ObjectGraphEncoder", {"atom_expansion": "clique"}),
        ("TransformerBiasEncoder", {}),
    ),
)
def test_encode_batch_pyg_backend_tensor_parity(
    facade_name: str, kwargs: dict[str, Any]
) -> None:
    """Batch outputs must agree key-by-key across backends.

    The initial state is encoded twice per backend: the two runtimes iterate
    fluent atoms of freshly reached states in different orders, so only
    shared-vocabulary-stable states carry comparable node ids.
    """

    def make_encoder(problem_or_task, backend: str):
        if facade_name == "StarGraphEncoder":
            return mifrost.StarGraphEncoder(problem_or_task, backend=backend, **kwargs)
        if facade_name == "ObjectGraphEncoder":
            return ObjectGraphEncoder(problem_or_task, backend=backend, **kwargs)
        return TransformerBiasEncoder(problem_or_task, backend=backend)

    pymimir_problem, planning_task, pytyr_search = _backend_pair()
    pymimir_state, pytyr_state = _root_states(pymimir_problem, pytyr_search)
    pymimir_batch = make_encoder(pymimir_problem, "pymimir").encode_batch_pyg(
        [pymimir_state, pymimir_state]
    )
    pytyr_batch = make_encoder(planning_task, "pytyr").encode_batch_pyg(
        [pytyr_state, pytyr_state]
    )

    assert pymimir_batch.num_graphs == pytyr_batch.num_graphs == 2
    _assert_pyg_dicts_equal(pymimir_batch.to_dict(), pytyr_batch.to_dict())


def test_stream_flush_matches_direct_encode_batch() -> None:
    pymimir_problem, _, _ = _backend_pair()
    root = pymimir_problem.get_initial_state()

    encoder = mifrost.StarGraphEncoder(pymimir_problem, backend="pymimir")
    stream = encoder.stream()
    stream.append(root)
    stream.append(root)
    streamed = stream.flush_pyg()

    direct = encoder.encode_batch_pyg([root, root])

    assert streamed.num_graphs == direct.num_graphs == 2
    _assert_pyg_dicts_equal(streamed.to_dict(), direct.to_dict())


def test_tuple_tensor_encoder_contract_and_localization() -> None:
    from mifrost.encoders.derived import TupleTensorEncoder

    pymimir_problem, _, _ = _backend_pair()
    state = pymimir_problem.get_initial_state()
    encoder = TupleTensorEncoder(pymimir_problem)

    data = encoder.encode_pyg(state).to_dict()
    assert data.keys().isdisjoint(
        {"tuple_slot_sizes", "tuple_slot_sizes_ptr", "tuple_counts"}
    )
    ptr = data["tuple_ptr"]
    assert int(ptr[0]) == 0
    assert int(ptr[-1]) == int(data["tuple_args"].numel())
    assert all(
        data[key].dtype == torch.long
        for key in ("tuple_args", "tuple_ptr", "tuple_rel_ids", "tuple_role_ids")
    )

    batch = encoder.encode_batch_pyg([state, state]).to_dict()
    node_offset = int(data["x_ids"].size(0)) if "x_ids" in data else 0
    num_tuples = int(data["tuple_ptr"][-1])
    for key in ("tuple_args", "tuple_rel_ids", "tuple_role_ids"):
        value = data[key]
        batch_value = batch[key]
        assert torch.equal(batch_value[: value.numel()], value)
        shifted = batch_value[value.numel() :] - (
            node_offset if key == "tuple_args" else num_tuples
        )
        assert torch.equal(shifted, value)
    batch_ptr = batch["tuple_ptr"]
    assert int(batch_ptr[0]) == 0
    assert int(batch_ptr[-1]) == int(batch["tuple_args"].numel())
    assert (batch_ptr.diff() >= 0).all()

    stream = TupleTensorEncoder(pymimir_problem).stream()
    stream.append(state)
    stream.append(state)
    assert torch.equal(stream.flush_pyg().to_dict()["tuple_ptr"], batch["tuple_ptr"])


def _collapse_consecutive_duplicates(tensor: torch.Tensor) -> torch.Tensor:
    """Drop consecutive duplicate entries from a 1-D tensor.

    Boundary-dedup for ``tuple_ptr``: manual re-batching concatenates
    per-graph CSRs, which duplicates the offsets where graph fragments meet;
    collapsing consecutive duplicates recovers the logical CSR shape.
    """
    if tensor.numel() <= 1:
        return tensor
    keep = torch.ones(tensor.numel(), dtype=torch.bool, device=tensor.device)
    keep[1:] = tensor[1:] != tensor[:-1]
    return tensor[keep]


@pytest.mark.parametrize(
    ("facade_name", "kwargs"),
    (
        ("StarGraphEncoder", {}),
        ("ObjectGraphEncoder", {"atom_expansion": "clique"}),
        ("AtomLineGraphEncoder", {}),
        ("HypergraphIncidenceEncoder", {}),
        ("TransformerBiasEncoder", {}),
        ("TupleTensorEncoder", {}),
    ),
)
def test_single_batch_conversion_consistency(
    facade_name: str, kwargs: dict[str, Any]
) -> None:
    """Manually re-batched singles must equal direct batch encoding."""
    from mifrost.encoders.derived import (
        AtomLineGraphEncoder,
        HypergraphIncidenceEncoder,
        StarGraphEncoder,
        TupleTensorEncoder,
    )

    pymimir_problem, _, _ = _backend_pair()
    state = pymimir_problem.get_initial_state()

    facade = {
        "StarGraphEncoder": StarGraphEncoder,
        "ObjectGraphEncoder": ObjectGraphEncoder,
        "AtomLineGraphEncoder": AtomLineGraphEncoder,
        "HypergraphIncidenceEncoder": HypergraphIncidenceEncoder,
        "TransformerBiasEncoder": TransformerBiasEncoder,
        "TupleTensorEncoder": TupleTensorEncoder,
    }[facade_name]
    encoder = facade(pymimir_problem, **kwargs)

    single = encoder.encode_pyg(state)
    batch = encoder.encode_batch_pyg([state, state])
    rebatched = type(batch).from_data_list([single, single])

    assert rebatched.num_graphs == batch.num_graphs == 2
    assert torch.equal(rebatched.x_ids, batch.x_ids)
    assert torch.equal(rebatched.edge_index, batch.edge_index)
    assert torch.equal(rebatched.edge_attr.float(), batch.edge_attr.float())
    if facade_name == "HypergraphIncidenceEncoder":
        assert torch.equal(rebatched.hyperedge_index, batch.hyperedge_index)
        assert torch.equal(rebatched.hyperedge_attr_ids, batch.hyperedge_attr_ids)
    if facade_name == "TupleTensorEncoder":
        assert torch.equal(rebatched.tuple_args, batch.tuple_args)
        assert torch.equal(
            _collapse_consecutive_duplicates(rebatched.tuple_ptr),
            _collapse_consecutive_duplicates(batch.tuple_ptr),
        )
        assert torch.equal(rebatched.tuple_rel_ids, batch.tuple_rel_ids)
        assert torch.equal(rebatched.tuple_role_ids, batch.tuple_role_ids)
