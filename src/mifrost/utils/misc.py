from __future__ import annotations

import inspect
from contextlib import contextmanager
from functools import singledispatch
from typing import Any, List

import torch
from torch_geometric.data import Batch


@singledispatch
def tolist(input_, **kwargs) -> List:
    return list(input_)


@tolist.register(list)
def _(input_: list, *, ensure_copy: bool = False, **kwargs) -> List:
    if ensure_copy:
        return input_.copy()
    return input_


@tolist.register(torch.Tensor)
def _(input_: torch.Tensor, **kwargs) -> List:
    return input_.tolist()


@tolist.register(Batch)
def _(input_: Batch, **kwargs) -> List:
    return input_.to_data_list()


def forward_kwargs(fn, kwargs):
    sig = inspect.signature(fn)
    # prefilter to avoid TypeError on unknown keys
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    bound = sig.bind_partial(**allowed)
    # prop positional-only and *args entries just in case
    result = {
        k: v
        for k, v in bound.arguments.items()
        if k in sig.parameters
        and sig.parameters[k].kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_KEYWORD,
        )
    }
    return result


@contextmanager
def monkeypatch(obj, attribute_name, new_value):
    """
    Like pytest's monkeypatch, but as a context manager.
    """
    old_value = getattr(obj, attribute_name)
    try:
        setattr(obj, attribute_name, new_value)
        yield
    finally:
        setattr(obj, attribute_name, old_value)
