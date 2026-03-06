from __future__ import annotations

from typing import Iterable

from .. import _core

TargetSource = getattr(_core, "TargetSource", None)


_TARGET_SOURCE_ALIASES = (
    {
        "action": TargetSource.Actions,
        "actions": TargetSource.Actions,
        "goal": TargetSource.Goals,
        "goals": TargetSource.Goals,
        "subgoal": TargetSource.Subgoals,
        "subgoals": TargetSource.Subgoals,
        "state": TargetSource.States,
        "states": TargetSource.States,
        "history": TargetSource.History,
    }
    if TargetSource is not None
    else {}
)


def normalize_target_sources(
    target_sources: Iterable[TargetSource | str] | None,
) -> set[TargetSource] | None:
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
