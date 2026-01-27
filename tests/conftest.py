"""Test helpers to avoid editable-import hooks during local runs."""

import sys


def _strip_scikit_build_editable() -> None:
    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if finder.__class__.__module__ != "_mifrost_editable"
    ]


_strip_scikit_build_editable()
