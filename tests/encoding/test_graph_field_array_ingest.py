"""Graph fields accept contiguous arrays without being walked in Python.

``BatchBuilder.set_field`` takes an untyped Python object, unlike its siblings
``add_node_features``/``add_edges`` which take ``nb::ndarray``. Its coercion
helper therefore used to treat *every* array-like as a generic iterable and step
it one element at a time. For a framework whose elements are themselves array
scalars -- torch, whose ``t[i]`` is a fresh 0-dim tensor -- that meant a heap
allocation, three attribute lookups and an ``.item()`` call **per element** on
data that was already a flat contiguous ``int64_t*``.

The observable symptom was an inversion: the same numbers ingested faster as a
plain Python list than as a contiguous tensor, because list elements are already
Python ints and skip the array-scalar branch entirely. That inversion is what
these tests pin -- a caller doing the native thing must not be punished for it.

The fast path is deliberately narrow (host-resident, contiguous, rank 1 or 2, a
supported dtype); everything else keeps the original element-wise behaviour, so
the correctness cases below cover both sides of that fork.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

import mifrost

_SPEC = {
    "dtype": "i64",
    "mode": "cat",
    "dim": 1,
    "cat_dim": 0,
    "inc": {"kind": "none"},
}


def _roundtrip(value, *, dim: int = 1, dtype: str = "i64") -> torch.Tensor:
    """Push one value through a single-graph builder and read it back."""

    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("flat")
    builder.register_field("f", {**_SPEC, "dim": dim, "dtype": dtype})
    builder.add_nodes("entity", 4)
    builder.set_field("f", value)
    return builder.build().f


_SEQUENCE = [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    ("label", "value", "expected"),
    [
        ("torch int64", torch.arange(5, dtype=torch.int64), _SEQUENCE),
        ("torch int32", torch.arange(5, dtype=torch.int32), _SEQUENCE),
        ("torch int16", torch.arange(5, dtype=torch.int16), _SEQUENCE),
        ("torch uint8", torch.arange(5, dtype=torch.uint8), _SEQUENCE),
        ("torch float32", torch.arange(5, dtype=torch.float32), _SEQUENCE),
        ("numpy int64", np.arange(5, dtype=np.int64), _SEQUENCE),
        ("numpy int32", np.arange(5, dtype=np.int32), _SEQUENCE),
        ("numpy float64", np.arange(5, dtype=np.float64), _SEQUENCE),
        ("python list", _SEQUENCE, _SEQUENCE),
        ("python tuple", (0, 1, 2, 3, 4), _SEQUENCE),
        ("generator", iter(_SEQUENCE), _SEQUENCE),
        # Not contiguous, so it must fall back to the element-wise walk rather
        # than reading the underlying buffer straight through, which would
        # silently return the elements it strides over.
        (
            "strided view",
            torch.arange(10, dtype=torch.int64)[::2],
            [0, 2, 4, 6, 8],
        ),
    ],
)
def test_every_input_form_yields_the_expected_values(
    label: str, value, expected: list[int]
) -> None:
    assert _roundtrip(value).tolist() == expected


def test_scalars_are_untouched_by_the_array_path() -> None:
    """0-dim inputs resolve before the array fork and must stay scalars."""

    assert _roundtrip(7).tolist() == [7]
    assert _roundtrip(torch.tensor(7)).tolist() == [7]
    assert _roundtrip(np.int64(7)).tolist() == [7]


def test_two_dimensional_arrays_keep_row_major_order() -> None:
    """A 2D array flattens the same way nested iteration used to."""

    rows = torch.arange(6, dtype=torch.int64).reshape(2, 3)
    assert _roundtrip(rows, dim=3).tolist() == [[0, 1, 2], [3, 4, 5]]
    assert _roundtrip(rows.numpy(), dim=3).tolist() == [[0, 1, 2], [3, 4, 5]]
    # Nested lists take the element-wise path and must agree with it.
    assert _roundtrip([[0, 1, 2], [3, 4, 5]], dim=3).tolist() == [[0, 1, 2], [3, 4, 5]]


def test_bool_arrays_widen_to_the_field_dtype() -> None:
    assert _roundtrip(torch.tensor([True, False, True])).tolist() == [1, 0, 1]


def test_float_fields_accept_integer_arrays() -> None:
    values = _roundtrip(torch.arange(4, dtype=torch.int64), dtype="f32")
    assert values.tolist() == [0.0, 1.0, 2.0, 3.0]


def _ingest_nanoseconds_per_element(value, count: int) -> float:
    """Cost of one `set_field`, with the surrounding builder work subtracted."""

    def run(assign: bool) -> float:
        builder = mifrost.BatchBuilder()
        builder.set_graph_kind("flat")
        builder.register_field("f", _SPEC)
        start = time.perf_counter()
        for index in range(200):
            if index:
                builder.next_graph()
            builder.add_nodes("entity", 4)
            if assign:
                builder.set_field("f", value)
        return (time.perf_counter() - start) / 200

    baseline = min(run(False) for _ in range(3))
    loaded = min(run(True) for _ in range(3))
    return max(loaded - baseline, 0.0) / count * 1e9


def test_a_contiguous_tensor_is_not_slower_than_the_same_numbers_in_a_list() -> None:
    """The inversion that gave the defect away.

    Before the array fast path a contiguous int64 tensor cost ~7x more per
    element than a Python list of the same integers. The threshold is loose on
    purpose -- this is a timing test guarding an order of magnitude, not a
    benchmark, and it must not go red on a loaded machine.
    """

    count = 4096
    tensor_ns = _ingest_nanoseconds_per_element(
        torch.arange(count, dtype=torch.int64), count
    )
    list_ns = _ingest_nanoseconds_per_element(list(range(count)), count)

    assert tensor_ns < list_ns * 2.0, (
        f"ingesting a contiguous tensor ({tensor_ns:.1f} ns/element) should not "
        f"cost more than a plain Python list ({list_ns:.1f} ns/element); the "
        "array fast path is probably not being taken"
    )
