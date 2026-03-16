"""Shared target-source labels used by encoder `target_sources` options.

Interpret the sources as:

- `action`: explicit grounded actions passed through `actions=...`
- `goal`: literals from the root `goals=...` input
- `subgoal`: literals from `subgoal_layers=...`
- `state`: candidate states on horizon or transition encoders
- `history`: time-indexed literals from `history_subgoals=...`
"""

from __future__ import annotations

from typing import Iterable

from .. import _core

TargetSource = getattr(_core, "TargetSource", None)


_TARGET_SOURCE_ALIASES = (
    {
        "action": TargetSource.actions,
        "actions": TargetSource.actions,
        "goal": TargetSource.goals,
        "goals": TargetSource.goals,
        "subgoal": TargetSource.subgoals,
        "subgoals": TargetSource.subgoals,
        "state": TargetSource.states,
        "states": TargetSource.states,
        "history": TargetSource.history,
    }
    if TargetSource is not None
    else {}
)


def normalize_target_sources(
    target_sources: Iterable[TargetSource | str] | None,
) -> set[TargetSource] | None:
    """Normalize target-source names to native enum values.

    This helper is shared by `target_sources` and `lgan_anchor_sources`.
    The meaning of each source stays the same across encoders:

    - `action`: explicit grounded actions
    - `goal`: root-goal literals
    - `subgoal`: layered subgoal literals
    - `state`: horizon or transition candidate states
    - `history`: time-indexed history literals
    """
    if target_sources is None:
        return None
    if TargetSource is None:
        raise RuntimeError("target_sources requires mifrost._core.TargetSource support")
    if isinstance(target_sources, (TargetSource, str)):
        target_sources = [target_sources]
    out: set[TargetSource] = set()
    for source in target_sources:
        if isinstance(source, TargetSource):
            out.add(source)
            continue
        if isinstance(source, str):
            key = source.strip().lower()
            if key not in _TARGET_SOURCE_ALIASES:
                raise ValueError(
                    f"Unknown target source '{source}'. "
                    "Expected one of: action, goal, subgoal, state, history."
                )
            out.add(_TARGET_SOURCE_ALIASES[key])
            continue
        raise TypeError(
            "target_sources entries must be mifrost.TargetSource or str, "
            f"got {type(source)!r}"
        )
    return out


__all__ = ["TargetSource", "normalize_target_sources"]
