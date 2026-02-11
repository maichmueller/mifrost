from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Mode(str, Enum):
    STACK = "stack"
    CAT = "cat"
    RAGGED_CAT = "ragged_cat"
    CONST = "const"


class DType(str, Enum):
    F32 = "f32"
    I64 = "i64"


@dataclass(frozen=True)
class Inc:
    kind: str
    node_type: str = ""

    @staticmethod
    def none() -> "Inc":
        return Inc(kind="none")

    @staticmethod
    def node_offset(node_type: str) -> "Inc":
        if not node_type:
            raise ValueError("Inc.node_offset requires a non-empty node_type")
        return Inc(kind="node_offset", node_type=node_type)

    def to_core_dict(self) -> dict[str, str]:
        payload = {"kind": self.kind}
        if self.kind == "node_offset":
            if not self.node_type:
                raise ValueError("Inc.node_offset requires a non-empty node_type")
            payload["node_type"] = self.node_type
        return payload


@dataclass(frozen=True)
class GraphFieldSpec:
    mode: Mode
    dtype: DType
    dim: int = 1
    cat_dim: int = 0
    inc: Inc = field(default_factory=Inc.none)

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("GraphFieldSpec.dim must be > 0")
        normalized_cat_dim = 1 if self.cat_dim == -1 else int(self.cat_dim)
        if self.mode in (Mode.CAT, Mode.RAGGED_CAT):
            if normalized_cat_dim not in (0, 1):
                raise ValueError(
                    "GraphFieldSpec.cat_dim must be 0, 1, or -1 for CAT/RAGGED_CAT"
                )
        elif normalized_cat_dim != 0:
            raise ValueError("GraphFieldSpec.cat_dim must be 0 for STACK/CONST")
        object.__setattr__(self, "cat_dim", normalized_cat_dim)
        if self.inc.kind == "node_offset" and self.dtype is not DType.I64:
            raise ValueError("NODE_OFFSET increment requires dtype=DType.I64")

    def to_core_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "dtype": self.dtype.value,
            "dim": int(self.dim),
            "cat_dim": int(self.cat_dim),
            "inc": self.inc.to_core_dict(),
        }


__all__ = ["Mode", "DType", "Inc", "GraphFieldSpec"]
