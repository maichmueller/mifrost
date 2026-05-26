from __future__ import annotations

from typing import Any

from .._core import DEFAULT_PARENT_RELATION
from .flat_horizon import FlatHorizonEncoder
from .types import DomainInput


class FlatRootedHorizonEncoder(FlatHorizonEncoder):
    """First-class rooted flat horizon encoder.

    This wrapper exposes the rooted horizon lane as a concrete encoder instead
    of requiring callers to remember the generic FlatHorizonEncoder knobs.

    Defaults:
    - encode the root but do not score it as a target (`root_policy="encode_only"`)
    - emit grounded action relations on candidate states (`ignore_actions=False`)
    - emit parent topology edges (`enable_parent_relation=True`)
    """

    def __init__(
        self,
        domain: DomainInput,
        *,
        root_policy: str = "encode_only",
        enable_parent_relation: bool = True,
        ignore_actions: bool = False,
        parent_relation: str = DEFAULT_PARENT_RELATION,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            domain,
            root_policy=root_policy,
            enable_parent_relation=enable_parent_relation,
            ignore_actions=ignore_actions,
            parent_relation=parent_relation,
            **kwargs,
        )
