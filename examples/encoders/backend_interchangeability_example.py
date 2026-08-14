"""Encode both planners, batch their outputs, and run one learning step."""

from __future__ import annotations

from pathlib import Path

import torch

import mifrost


ROOT = Path(__file__).resolve().parents[2]
DOMAIN = ROOT / "data" / "pddl" / "blocks" / "domain.pddl"
PROBLEM = ROOT / "data" / "pddl" / "blocks" / "small.pddl"


def _pymimir_input():
    import pymimir

    domain = pymimir.Domain(DOMAIN)
    problem = pymimir.Problem(domain, PROBLEM, mode="lifted")
    return domain, problem.get_initial_state()


def _pytyr_input():
    from pypddl.formalism import ParserOptions
    from pyyggdrasil.execution import ExecutionContext
    from pytyr.formalism.planning import Parser
    from pytyr.planning.lifted import (
        AxiomEvaluatorFactory,
        StateRepositoryFactory,
        SuccessorGeneratorFactory,
        Task,
    )

    options = ParserOptions()
    planning_task = Parser(str(DOMAIN), options).parse_task(str(PROBLEM), options)
    task = Task(planning_task)
    context = ExecutionContext(1)
    evaluator = AxiomEvaluatorFactory().create(task, context)
    repository = StateRepositoryFactory().create(task)
    generator = SuccessorGeneratorFactory().create(task, context)
    return planning_task, generator.get_initial_node(repository, evaluator).get_state()


def main() -> None:
    pymimir_domain, pymimir_state = _pymimir_input()
    pytyr_task, pytyr_state = _pytyr_input()
    pymimir_encoder = mifrost.HGraphEncoder(pymimir_domain)
    pytyr_encoder = mifrost.HGraphEncoder(pytyr_task)

    pymimir_encoding = pymimir_encoder.encode(pymimir_state)
    pytyr_encoding = pytyr_encoder.encode(pytyr_state)
    assert pymimir_encoding.schema_fingerprint() == (
        pytyr_encoding.schema_fingerprint()
    )
    mixed = mifrost.batch_encodings([pymimir_encoding, pytyr_encoding], fast_path=True)
    data = mixed.as_pyg(as_batch=True)

    object_x = data["object"].x.float()
    object_batch = data["object"].batch
    model = torch.nn.Linear(object_x.shape[1], 1)
    node_scores = model(object_x).squeeze(-1)
    graph_scores = torch.zeros(mixed.num_graphs, dtype=node_scores.dtype)
    graph_scores.index_add_(0, object_batch, node_scores)
    graph_scores.square().mean().backward()

    print(
        "batched",
        mixed.num_graphs,
        "graphs from",
        pymimir_encoder.backend,
        "and",
        pytyr_encoder.backend,
    )
    print("schema fingerprint:", mixed.schema_fingerprint())
    print("learning step gradients:", model.weight.grad is not None)


if __name__ == "__main__":
    main()
