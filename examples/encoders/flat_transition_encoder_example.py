"""Flat transition encoding example for full and delta successor wrappers."""

from __future__ import annotations

from mifrost.encoders import FlatTransitionEffectsEncoder, FlatTransitionEncoder

from _helpers import begin_plot, first_transition, load_space, save_plot


def _print_summary(label: str, data) -> None:
    print(f"== {label} ==")
    print("schema:", data.schema)
    print("target entity names:", data.graph_target_entity_names(0))
    print("target positions:", data.graph_target_positions(0).tolist())
    print("target names:", data.graph_target_names(0))
    if hasattr(data, "target_depths"):
        print("target depths:", data.graph_target_depths(0).tolist())
    print("relation counts:", data.relation_instance_counts_total().tolist())


def main() -> None:
    space, domain, problem = load_space(domain="blocks", problem="smedium")
    state = problem.get_initial_state()
    _action, successor = first_transition(space, state)
    goals = list(problem.get_goal_condition().get_literals())

    full = FlatTransitionEncoder(domain)
    delta = FlatTransitionEffectsEncoder(domain)

    full_data = full.encode_pyg(state=state, successor=successor, goals=goals)
    delta_data = delta.encode_pyg(state=state, successor=successor, goals=goals)

    _print_summary("FlatTransitionEncoder", full_data)
    _print_summary("FlatTransitionEffectsEncoder", delta_data)

    try:
        from matplotlib import pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not installed; skipped plot export")
        return

    fig = begin_plot(figsize=(20, 10))
    ax_full, ax_delta = fig.subplots(1, 2)
    full.draw(full_data, ax=ax_full, with_labels=True, edge_labels=True)
    ax_full.set_title("FlatTransitionEncoder")
    delta.draw(delta_data, ax=ax_delta, with_labels=True, edge_labels=True)
    ax_delta.set_title("FlatTransitionEffectsEncoder")
    out_path = save_plot("flat_transition_encoder.png")
    print(f"Saved flat transition plot to {out_path}")


if __name__ == "__main__":
    main()
