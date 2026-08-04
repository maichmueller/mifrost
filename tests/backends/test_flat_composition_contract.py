from __future__ import annotations

import mifrost


def test_flat_composition_capabilities_are_generic() -> None:
    required = (
        mifrost._neutral_core.FlatCompositionCapability.state_facts.value
        | mifrost._neutral_core.FlatCompositionCapability.goal_facts.value
        | mifrost._neutral_core.FlatCompositionCapability.parent_relations.value
    )
    available = (
        mifrost._neutral_core.FlatCompositionCapability.state_facts.value
        | mifrost._neutral_core.FlatCompositionCapability.goal_facts.value
    )

    assert not mifrost._neutral_core.flat_composition_capabilities_satisfied(
        required,
        available,
    )
    assert (
        mifrost._neutral_core.flat_composition_missing_capabilities(required, available)
        == mifrost._neutral_core.FlatCompositionCapability.parent_relations.value
    )
    assert mifrost._neutral_core.flat_composition_capabilities_satisfied(
        required, required
    )
