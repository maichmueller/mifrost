from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class Mode(str, Enum):
    STACK = "stack"
    CAT = "cat"
    RAGGED_CAT = "ragged_cat"
    CONST = "const"


class DType(str, Enum):
    F32 = "f32"
    I64 = "i64"
    STR = "str"
    PYOBJ = "pyobj"


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
        if self.dtype in (DType.PYOBJ, DType.STR):
            raise ValueError(
                "GraphFieldSpec dtype is only supported for Python-side "
                "batch_encodings collate_spec, not native HGraph dynamic fields"
            )
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

    def to_core_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "dtype": self.dtype.value,
            "dim": int(self.dim),
            "cat_dim": int(self.cat_dim),
            "inc": self.inc.to_core_dict(),
        }

    @classmethod
    def from_spec(cls, spec: "GraphFieldSpec | Mapping[str, Any]") -> "GraphFieldSpec":
        if isinstance(spec, cls):
            return spec
        if not isinstance(spec, Mapping):
            raise TypeError(
                f"Graph field spec must be GraphFieldSpec or mapping, got {type(spec)!r}"
            )

        mode_raw = spec.get("mode")
        dtype_raw = spec.get("dtype")
        if mode_raw is None or dtype_raw is None:
            raise ValueError("Graph field spec mapping requires 'mode' and 'dtype'")

        mode = _normalize_mode(mode_raw)
        dtype = _normalize_dtype(dtype_raw)
        dim = int(spec.get("dim", 1))
        cat_dim = int(spec.get("cat_dim", 0))
        inc = _normalize_inc(spec.get("inc", {"kind": "none"}))
        return cls(mode=mode, dtype=dtype, dim=dim, cat_dim=cat_dim, inc=inc)


@dataclass(frozen=True)
class CollateSpec:
    mode: Mode
    dtype: DType
    dim: int = 1
    cat_dim: int = 0
    inc: Inc = field(default_factory=Inc.none)

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("CollateSpec.dim must be > 0")
        normalized_cat_dim = 1 if self.cat_dim == -1 else int(self.cat_dim)
        if self.dtype in (DType.F32, DType.I64):
            if self.mode in (Mode.CAT, Mode.RAGGED_CAT):
                if normalized_cat_dim not in (0, 1):
                    raise ValueError(
                        "CollateSpec.cat_dim must be 0, 1, or -1 for CAT/RAGGED_CAT"
                    )
            elif normalized_cat_dim != 0:
                raise ValueError("CollateSpec.cat_dim must be 0 for STACK/CONST")
            if self.inc.kind == "node_offset" and self.dtype is not DType.I64:
                raise ValueError("NODE_OFFSET increment requires dtype=DType.I64")
        else:
            if self.mode is Mode.CAT:
                raise ValueError("CollateSpec CAT mode requires numeric dtype")
            if normalized_cat_dim != 0 or self.dim != 1 or self.inc.kind != "none":
                raise ValueError(
                    "Non-numeric CollateSpec only supports dim=1, cat_dim=0, inc=none"
                )
        object.__setattr__(self, "cat_dim", normalized_cat_dim)

    def to_core_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"mode": self.mode.value, "dtype": self.dtype.value}
        if self.dtype in (DType.F32, DType.I64):
            out["dim"] = int(self.dim)
            out["cat_dim"] = int(self.cat_dim)
            out["inc"] = self.inc.to_core_dict()
        return out

    @classmethod
    def from_spec(cls, spec: "CollateSpec | Mapping[str, Any]") -> "CollateSpec":
        if isinstance(spec, cls):
            return spec
        if not isinstance(spec, Mapping):
            raise TypeError(
                f"Collate spec must be CollateSpec or mapping, got {type(spec)!r}"
            )
        mode_raw = spec.get("mode", Mode.STACK)
        dtype_raw = spec.get("dtype", DType.PYOBJ)
        mode = _normalize_mode(mode_raw)
        dtype = _normalize_dtype(dtype_raw)
        dim = int(spec.get("dim", 1))
        cat_dim = int(spec.get("cat_dim", 0))
        inc = _normalize_inc(spec.get("inc", {"kind": "none"}))
        return cls(mode=mode, dtype=dtype, dim=dim, cat_dim=cat_dim, inc=inc)


def _normalize_mode(mode: Mode | str) -> Mode:
    if isinstance(mode, Mode):
        return mode
    if isinstance(mode, str):
        return Mode(mode.lower())
    raise TypeError(f"Graph field mode must be Mode or str, got {type(mode)!r}")


def _normalize_dtype(dtype: DType | str | type) -> DType:
    if isinstance(dtype, DType):
        return dtype
    if isinstance(dtype, type):
        if dtype is str:
            return DType.STR
        if dtype is float:
            return DType.F32
        if dtype is int:
            return DType.I64
    if isinstance(dtype, str):
        return DType(dtype.lower())
    raise TypeError(f"Graph field dtype must be DType or str, got {type(dtype)!r}")


def _normalize_inc(inc: Inc | Mapping[str, Any]) -> Inc:
    if isinstance(inc, Inc):
        return inc
    if not isinstance(inc, Mapping):
        raise TypeError(f"Graph field inc must be Inc or mapping, got {type(inc)!r}")
    kind = str(inc.get("kind", "none")).lower()
    if kind == "node_offset":
        if "node_type" not in inc:
            raise ValueError("Graph field inc kind='node_offset' requires 'node_type'")
        return Inc.node_offset(str(inc["node_type"]))
    if kind == "none":
        return Inc.none()
    raise ValueError(f"Unsupported graph field inc kind: {kind!r}")


__all__ = ["Mode", "DType", "Inc", "GraphFieldSpec", "CollateSpec"]
