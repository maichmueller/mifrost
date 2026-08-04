from __future__ import annotations

import mifrost


def test_external_flat_mode_contract_is_stable() -> None:
    contracts = mifrost._neutral_core.flat_external_mode_contracts()

    assert [contract.name for contract in contracts] == [
        "concurrent_internal",
        "concurrent_internal_tree",
        "concurrent_internal_tree_rooted",
        "concurrent_internal_comparison_tree",
        "concurrent_internal_action_tree",
        "concurrent_internal_action_hybrid_tree",
    ]
    rooted = mifrost._neutral_core.flat_external_mode_contract(
        mifrost._neutral_core.FlatExternalMode.concurrent_internal_tree_rooted
    )
    assert (
        rooted.required_components
        & mifrost._neutral_core.FlatExternalComponent.root_action_nodes.value
    )


def test_external_flat_mode_capability_validation() -> None:
    components = (
        mifrost._neutral_core.FlatExternalComponent.state_facts.value
        | mifrost._neutral_core.FlatExternalComponent.goal_facts.value
        | mifrost._neutral_core.FlatExternalComponent.transition_effects.value
    )
    assert mifrost._neutral_core.flat_external_mode_satisfied(
        mifrost._neutral_core.FlatExternalMode.concurrent_internal,
        components,
    )
    assert not mifrost._neutral_core.flat_external_mode_satisfied(
        mifrost._neutral_core.FlatExternalMode.concurrent_internal_tree,
        components,
    )
    assert (
        mifrost._neutral_core.flat_external_mode_missing_components(
            mifrost._neutral_core.FlatExternalMode.concurrent_internal_tree,
            components,
        )
        == mifrost._neutral_core.FlatExternalComponent.parent_relations.value
    )
