"""Shared root-policy handling for horizon-style encoders.

Interpret the policies as:

- `include`: keep the root in encoding and in target metadata
- `encode_only`: keep the root in encoding, but omit it from target metadata
- `exclude`: omit the root from target metadata and keep root facts as base facts
"""

from __future__ import annotations

from .. import _core

RootPolicy = getattr(_core, "RootPolicy", None)


def normalize_root_policy(policy: RootPolicy | str | None) -> RootPolicy | None:
    """Normalize one horizon root policy to the native enum."""
    if policy is None:
        return None
    if RootPolicy is None:
        raise RuntimeError("root_policy requires mifrost._core.RootPolicy support")
    if isinstance(policy, RootPolicy):
        return policy
    if isinstance(policy, str):
        key = policy.strip().lower()
        if key == "include":
            return RootPolicy.include
        if key in {"encode_only", "encode-only", "encoded_only", "encoded-only"}:
            return RootPolicy.encode_only
        if key == "exclude":
            return RootPolicy.exclude
        raise ValueError(
            f"Unknown root_policy '{policy}'. "
            "Expected one of: include, encode_only, exclude."
        )
    raise TypeError(
        f"root_policy must be mifrost.RootPolicy or str, got {type(policy)!r}"
    )


__all__ = ["RootPolicy", "normalize_root_policy"]
