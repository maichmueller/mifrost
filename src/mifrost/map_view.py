from __future__ import annotations

from abc import ABCMeta
from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView
from typing import Any, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class _MapViewMeta(ABCMeta):
    _typed_cache: dict[tuple[type[Any], type[Any]], type["MapView[Any, Any]"]] = {}

    def __getitem__(cls, params: object) -> type["MapView[Any, Any]"]:
        if cls is not MapView:
            raise TypeError("Nested MapView[...] specializations are not supported")

        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError("MapView[...] expects exactly two type arguments")

        key_type, value_type = params
        if not isinstance(key_type, type) or not isinstance(value_type, type):
            raise TypeError("MapView[...] runtime checks require concrete Python types")

        pair = (key_type, value_type)
        cached = _MapViewMeta._typed_cache.get(pair)
        if cached is not None:
            return cached

        typed_cls = _MapViewMeta(
            f"MapView[{key_type.__name__},{value_type.__name__}]",
            (MapView,),
            {
                "__module__": MapView.__module__,
                "_typed": True,
                "_expected_key_type": key_type,
                "_expected_value_type": value_type,
            },
        )
        _MapViewMeta._typed_cache[pair] = typed_cls
        return typed_cls

    def __instancecheck__(cls, instance: object) -> bool:
        if cls is MapView:
            return type.__instancecheck__(cls, instance)

        if not type.__instancecheck__(MapView, instance):
            return False

        if not getattr(cls, "_typed", False):
            return True

        return getattr(instance, "key_type", None) is getattr(
            cls, "_expected_key_type"
        ) and getattr(instance, "value_type", None) is getattr(
            cls, "_expected_value_type"
        )


class MapView(Mapping[K, V], Generic[K, V], metaclass=_MapViewMeta):
    """Public read-only map wrapper around private nanobind map-view implementations."""

    __slots__ = ("_impl", "key_type", "value_type")

    _typed = False

    def __init__(self, impl: object) -> None:
        key_type, value_type = _resolve_impl_types(impl)
        if not isinstance(key_type, type) or not isinstance(value_type, type):
            raise TypeError(
                "Concrete map view implementation is missing key_type/value_type metadata"
            )
        self._impl = impl
        self.key_type = key_type
        self.value_type = value_type

    @classmethod
    def from_impl(cls, impl: object) -> "MapView[Any, Any]":
        if isinstance(impl, MapView):
            return impl
        return cls(impl)

    def __len__(self) -> int:
        return len(self._impl)

    def __iter__(self) -> Iterator[K]:
        return iter(self._impl)

    def __contains__(self, key: object) -> bool:
        return key in self._impl

    def __getitem__(self, key: K) -> V:
        return self._impl.at(key)

    def __bool__(self) -> bool:
        return bool(self._impl)

    def keys(self) -> KeysView[K] | object:
        return self._impl.keys()

    def values(self) -> ValuesView[V] | object:
        return self._impl.values()

    def items(self) -> ItemsView[K, V] | object:
        return self._impl.items()

    def as_dict(self) -> dict[K, V]:
        return self._impl.as_dict()

    def __repr__(self) -> str:
        return (
            f"MapView[{self.key_type.__name__}, {self.value_type.__name__}]"
            f"(len={len(self)})"
        )


def _wrap_map_view_method(owner_cls: type[object], name: str) -> None:
    original = getattr(owner_cls, name)
    if getattr(original, "__mifrost_map_view_wrapped__", False):
        return

    def wrapped(self: object, *args: object, **kwargs: object) -> MapView[Any, Any]:
        return MapView.from_impl(original(self, *args, **kwargs))

    wrapped.__name__ = getattr(original, "__name__", name)
    wrapped.__qualname__ = getattr(
        original, "__qualname__", f"{owner_cls.__name__}.{wrapped.__name__}"
    )
    wrapped.__doc__ = getattr(original, "__doc__", None)
    wrapped.__mifrost_map_view__ = getattr(original, "__mifrost_map_view__", False)
    wrapped.__mifrost_map_view_wrapped__ = True
    setattr(owner_cls, name, wrapped)


def _iter_marked_map_view_methods(
    core_module: object,
) -> Iterator[tuple[type[object], str]]:
    for class_name in dir(core_module):
        owner_cls = getattr(core_module, class_name, None)
        if not isinstance(owner_cls, type):
            continue

        marker = getattr(owner_cls, "__mifrost_map_view_methods__", ())
        if isinstance(marker, str):
            marker = (marker,)

        for method_name in marker:
            if not isinstance(method_name, str):
                continue
            method = getattr(owner_cls, method_name, None)
            if callable(method):
                yield owner_cls, method_name


def _infer_types_from_impl(impl: object) -> tuple[type[Any], type[Any]] | None:
    items = impl.items()
    for key, value in items:
        return type(key), type(value)
    return None


def _resolve_impl_types(impl: object) -> tuple[type[Any], type[Any]]:
    impl_cls = impl.__class__
    key_type = getattr(impl_cls, "key_type", None)
    value_type = getattr(impl_cls, "value_type", None)
    if isinstance(key_type, type) and isinstance(value_type, type):
        return key_type, value_type

    inferred = _infer_types_from_impl(impl)
    if inferred is None:
        raise TypeError(
            "Cannot infer key_type/value_type from empty map view implementation"
        )

    key_type, value_type = inferred
    setattr(impl_cls, "key_type", key_type)
    setattr(impl_cls, "value_type", value_type)
    return key_type, value_type


def install_map_view_wrappers(core_module: object) -> None:
    sentinel = "__mifrost_map_view_wrappers_installed__"
    if getattr(core_module, sentinel, False):
        return

    for cls, name in _iter_marked_map_view_methods(core_module):
        _wrap_map_view_method(cls, name)

    setattr(core_module, sentinel, True)
