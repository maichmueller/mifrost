"""Rendering helpers for the derived-graph encoder family.

Mirrors the hgraph/color visualization idioms: ``derived_to_networkx``
converts one PyG :class:`DerivedGraphData` into a NetworkX multigraph with
stable display names and decoded role/kind attributes, and
``draw_derived`` renders it with role-shaped, kind-styled matplotlib passes.

Two vocabularies matter here and they are *not* the same thing:

``CORE_ROLE_NAMES`` / ``CORE_EDGE_KIND_NAMES``
    Positional mirrors of the encoder's ``vocab_roles`` / ``vocab_edge_kinds``
    graph attrs. Index == id. These are the only names used to *decode*
    integer channels, and the payload's own vocabularies win over them
    whenever present.
``ROLE_NAMES`` / ``EDGE_KIND_NAMES``
    The *display* vocabularies: the core vocabulary followed by the
    display-only names this module synthesizes (hyperedge/tuple instance
    pseudo-nodes and their membership/slot/instance/spd edges). The core
    prefix is index-stable, so ``ROLE_NAMES[:len(CORE_ROLE_NAMES)]`` is still
    id-indexed; the appended entries live beyond the core id space and are
    never produced by decoding a tensor.

The distinction matters because the core gained role ``anchor`` = 6, which
used to be where this module kept its synthetic ``"hyperedge"`` role.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Literal, Sequence

import torch

_LineStyle = Literal["-", "--", "-.", ":"]

#: Core node roles, index == ``vocab_roles`` id (``anchor`` = 6 is a real
#: node emitted by the encoder, not a rendering artefact).
CORE_ROLE_NAMES: tuple[str, ...] = (
    "object",
    "fact",
    "goal",
    "subgoal",
    "history",
    "action",
    "anchor",
)

#: Display-only roles for the auxiliary instance nodes this module
#: synthesizes. They are appended *after* the core ids and never decoded
#: from ``x_ids``.
AUX_ROLE_NAMES: tuple[str, ...] = ("hyperedge", "tuple")

ROLE_NAMES: tuple[str, ...] = CORE_ROLE_NAMES + AUX_ROLE_NAMES

#: Core edge kinds, index == ``vocab_edge_kinds`` id.
CORE_EDGE_KIND_NAMES: tuple[str, ...] = (
    "arg_fwd",
    "arg_bwd",
    "clique_fwd",
    "clique_bwd",
    "chain_fwd",
    "chain_bwd",
    "star_first_fwd",
    "star_first_bwd",
    "nullary_self",
    "action_fwd",
    "action_bwd",
    "line_share",
    "unary_self",
)

#: Display-only edge kinds for synthesized overlays.
AUX_EDGE_KIND_NAMES: tuple[str, ...] = ("membership", "slot", "instance", "spd")

EDGE_KIND_NAMES: tuple[str, ...] = CORE_EDGE_KIND_NAMES + AUX_EDGE_KIND_NAMES

CATEGORY_NAMES: tuple[str, ...] = ("static", "fluent", "derived", "action")

REVERSE_KINDS: frozenset[str] = frozenset(
    {"arg_bwd", "clique_bwd", "chain_bwd", "star_first_bwd", "action_bwd"}
)

SELF_LOOP_KINDS: frozenset[str] = frozenset({"nullary_self", "unary_self"})

ROLE_COLORS: dict[str, str] = {
    "object": "#4C72B0",
    "fact": "#DD8452",
    "goal": "#55A868",
    "subgoal": "#C44E52",
    "history": "#8172B3",
    "action": "#937860",
    "anchor": "#CCB974",
    "hyperedge": "#DA8BC3",
    "tuple": "#64B5CD",
}

KIND_STYLES: dict[str, tuple[str, float, float, _LineStyle]] = {
    "arg_fwd": ("#666666", 1.0, 0.9, "-"),
    "arg_bwd": ("#BBBBBB", 0.5, 0.35, ":"),
    "clique_fwd": ("#1F77B4", 1.4, 0.9, "-"),
    "clique_bwd": ("#1F77B4", 0.6, 0.3, ":"),
    "chain_fwd": ("#2CA02C", 1.4, 0.9, "-"),
    "chain_bwd": ("#2CA02C", 0.6, 0.3, ":"),
    "star_first_fwd": ("#FF7F0E", 1.4, 0.9, "-"),
    "star_first_bwd": ("#FF7F0E", 0.6, 0.3, ":"),
    "nullary_self": ("#CCB974", 1.2, 0.9, "-"),
    "unary_self": ("#B07AA1", 1.2, 0.9, "-"),
    "action_fwd": ("#8C564B", 1.4, 0.9, "-"),
    "action_bwd": ("#8C564B", 0.6, 0.3, ":"),
    "line_share": ("#17BECF", 0.5, 0.16, "-"),
    "membership": ("#DA8BC3", 1.0, 0.6, ":"),
    "slot": ("#64B5CD", 1.1, 0.7, ":"),
    "instance": ("#7F7F7F", 1.1, 0.75, "-."),
    "spd": ("#BCBD22", 1.6, 0.9, "--"),
}

#: Arc curvature per edge kind. Overlay kinds are bent away from the
#: structural edges they run parallel to -- an spd pair and a clique edge
#: connect exactly the same two objects, so a straight spd edge is invisible.
#: Paint order. ``line_share`` is a fact-to-fact shortcut relation that can
#: outnumber the structural edges ten to one; it belongs *under* them as a
#: density wash, not on top as a hairball.
KIND_ZORDER: dict[str, float] = {
    "line_share": 0.4,
    "membership": 1.2,
    "slot": 1.2,
    "spd": 1.4,
    "instance": 1.5,
}

KIND_CURVATURE: dict[str, float] = {
    "spd": 0.3,
    "line_share": 0.16,
    "instance": 0.1,
    "membership": 0.08,
    "slot": 0.08,
}

ROLE_SHAPES: dict[str, str] = {
    "object": "o",
    "fact": "o",
    "goal": "D",
    "subgoal": "D",
    "history": "s",
    "action": "s",
    "anchor": "h",
    "hyperedge": "P",
    "tuple": "X",
}


# --------------------------------------------------------------------------- #
# payload helpers


def _str_vocab(data: Any, name: str) -> tuple[str, ...]:
    """Return one string vocabulary graph attr as a tuple (possibly empty)."""
    value = getattr(data, name, None)
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(item) for item in value)
    except TypeError:
        return ()


def _scalar_int(value: Any) -> int | None:
    """Coerce a graph attr that may be an int, a 0-d or a 1-element tensor."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return int(value.reshape(-1)[0].item())
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_list(value: Any) -> list[int] | None:
    """Return a flat python int list for a 1-d tensor-ish field."""
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return None
    return [int(item) for item in value.reshape(-1).long().tolist()]


def _decode(names: Sequence[str], index: int, fallback: Sequence[str] = ()) -> str:
    """Index a vocabulary, falling back and then failing loudly, not silently."""
    if 0 <= index < len(names):
        return str(names[index])
    if 0 <= index < len(fallback):
        return str(fallback[index])
    return str(index)


class _Decoder:
    """Decodes the six-channel id space of one encoded graph."""

    def __init__(self, data: Any) -> None:
        self.roles = _str_vocab(data, "vocab_roles") or CORE_ROLE_NAMES
        self.categories = _str_vocab(data, "vocab_categories") or CATEGORY_NAMES
        self.edge_kinds = _str_vocab(data, "vocab_edge_kinds") or CORE_EDGE_KIND_NAMES
        self.predicates = _str_vocab(data, "vocab_predicates")
        self.actions = _str_vocab(data, "vocab_actions")
        self.relations = _str_vocab(data, "vocab_relations")
        num_predicates = _scalar_int(getattr(data, "num_predicates", None))
        if num_predicates is None and self.relations and self.actions:
            num_predicates = len(self.relations) - len(self.actions)
        self.num_predicates = num_predicates

    def role(self, role_id: int) -> str:
        return _decode(self.roles, role_id, CORE_ROLE_NAMES)

    def category(self, category_id: int) -> str:
        return _decode(self.categories, category_id, CATEGORY_NAMES)

    def kind(self, kind_id: int) -> str:
        return _decode(self.edge_kinds, kind_id, CORE_EDGE_KIND_NAMES)

    def relation(self, rel_plus_one: int, role_id: int) -> tuple[str, str]:
        """Decode ``relation_id + 1`` into ``(name, "predicate"|"action"|"")``.

        The unified relation id space is ``vocab_predicates ++ vocab_actions``
        split at ``num_predicates``. Older payloads carry only
        ``vocab_predicates``; there an action-role node's id is *not* a
        predicate id, so it renders as ``<action#k>`` rather than silently
        borrowing an unrelated predicate name (the D3 mislabelling).
        """
        rel = int(rel_plus_one) - 1
        if rel < 0:
            return "", ""
        if self.relations:
            name = _decode(self.relations, rel)
            if self.num_predicates is not None:
                kind = "action" if rel >= self.num_predicates else "predicate"
            else:
                kind = "action" if int(role_id) == 5 else "predicate"
            return name, kind
        if int(role_id) == 5:
            if self.actions and rel < len(self.actions):
                return str(self.actions[rel]), "action"
            return f"<action#{rel}>", "action"
        if self.predicates and rel < len(self.predicates):
            return str(self.predicates[rel]), "predicate"
        return f"<relation#{rel}>", ""


def _instance_detail(
    *,
    role: str,
    relation: str,
    relation_kind: str,
    sign: int,
    goal_level: int,
    history_dt: int,
    category: str,
) -> str:
    """Render the six decoded channels as one compact human string."""
    head = relation or "-"
    if relation_kind == "action":
        head = f"@{head}"
    if int(sign):
        head = f"not {head}"
    parts = [role, head]
    if role in {"goal", "subgoal"} or int(goal_level):
        parts.append(f"lvl{int(goal_level)}")
    if int(history_dt):
        parts.append(f"dt{int(history_dt):+d}")
    if category:
        parts.append(category)
    return " ".join(parts)


def _instance_label(
    index: int,
    *,
    prefix: str,
    node_names: Sequence[str] | None,
    instance_node: int | None,
    relation: str,
    relation_kind: str,
    sign: int,
    argument_labels: Sequence[str],
) -> str:
    """Name one synthesized instance node.

    Prefers the encoder's own canonical node name for the reified instance
    (``instance_node_indices``); rebuilds ``rel(a b c)`` from the decoded
    relation and its members when the instance was never reified (the
    objects-only universe).
    """
    canonical = ""
    if (
        instance_node is not None
        and node_names is not None
        and 0 <= instance_node < len(node_names)
    ):
        canonical = str(node_names[instance_node])
    if not canonical:
        head = relation or "?"
        if relation_kind == "action":
            head = f"@{head}"
        body = " ".join(argument_labels)
        canonical = f"({head} {body})" if body else f"({head})"
        if int(sign):
            canonical = f"not {canonical}"
    return f"{prefix}{index} {canonical}"


# --------------------------------------------------------------------------- #
# conversion


def derived_to_networkx(
    data: Any,
    *,
    include_hyperedges: bool = True,
    include_reverse_edges: bool = True,
    include_line_shares: bool = False,
    include_self_loops: bool = False,
    include_instances: bool | None = None,
    include_spd: bool | None = None,
) -> Any:
    """Convert derived-graph PyG data into a labeled NetworkX multigraph.

    Node attributes carry the decoded integer channels (``role``,
    ``relation``, ``relation_kind``, ``predicate``, ``sign``, ``goal_level``,
    ``history_dt``, ``category``) plus the raw ``channel_ids`` row; edge
    attributes carry the decoded kind name, both argument positions and --
    when the payload uses the nine-column ``edge_attr`` -- the originating
    instance's own labels.

    ``include_hyperedges`` is the master switch for every synthesized
    overlay; ``include_instances`` (hyperedge / tuple instance pseudo-nodes)
    and ``include_spd`` (shortest-path-distance object pairs) default to it
    and can be overridden individually.

    Self-loops (``nullary_self`` / ``unary_self``) are the *only* trace a
    nullary or unary instance leaves in the objects-only projection. When
    ``include_self_loops`` is false they are still recorded on their node as
    ``self_loops`` / ``self_loop_labels`` so no encoded node can silently
    lose its reason to exist.
    """
    import networkx as nx

    if include_instances is None:
        include_instances = include_hyperedges
    if include_spd is None:
        include_spd = include_hyperedges

    decoder = _Decoder(data)
    graph = nx.MultiDiGraph()
    node_names = getattr(data, "node_names", None)
    count = int(getattr(data, "num_nodes", 0))
    x_ids = getattr(data, "x_ids", None)
    has_channels = (
        isinstance(x_ids, torch.Tensor) and x_ids.dim() == 2 and x_ids.size(1) >= 6
    )
    rows = x_ids.long().tolist() if isinstance(x_ids, torch.Tensor) else []

    labels: list[str] = []
    for index in range(count):
        name = (
            str(node_names[index])
            if node_names is not None and index < len(node_names)
            else str(index)
        )
        labels.append(name)
        attrs: dict[str, Any] = {
            "role": "object",
            "role_id": 0,
            "relation": "",
            "relation_kind": "",
            "predicate": "",
            "action": "",
            "sign": 0,
            "goal_level": 0,
            "history_dt": 0,
            "category": "",
            "category_id": 0,
            "synthetic": False,
            "self_loops": (),
            "self_loop_labels": (),
        }
        if has_channels and index < len(rows):
            row = rows[index]
            role_id = int(row[0])
            role = decoder.role(role_id)
            relation, relation_kind = decoder.relation(int(row[1]), role_id)
            category_id = int(row[5])
            category = decoder.category(category_id)
            attrs.update(
                role=role,
                role_id=role_id,
                relation=relation,
                relation_kind=relation_kind,
                predicate=relation if relation_kind == "predicate" else "",
                action=relation if relation_kind == "action" else "",
                sign=int(row[2]),
                goal_level=int(row[3]),
                history_dt=int(row[4]),
                category=category,
                category_id=category_id,
                channel_ids=row,
                detail=_instance_detail(
                    role=role,
                    relation=relation,
                    relation_kind=relation_kind,
                    sign=int(row[2]),
                    goal_level=int(row[3]),
                    history_dt=int(row[4]),
                    category=category,
                ),
            )
        graph.add_node(index, label=name, **attrs)

    _add_core_edges(
        graph,
        data,
        decoder,
        include_reverse_edges=include_reverse_edges,
        include_line_shares=include_line_shares,
        include_self_loops=include_self_loops,
    )

    added_instances = False
    if include_instances:
        added_instances = _add_hyperedge_nodes(graph, data, decoder, labels)
        if not added_instances:
            _add_tuple_nodes(graph, data, decoder, labels)
    if include_spd:
        _add_spd_edges(graph, data)
    return graph


def _add_core_edges(
    graph: Any,
    data: Any,
    decoder: _Decoder,
    *,
    include_reverse_edges: bool,
    include_line_shares: bool,
    include_self_loops: bool,
) -> None:
    """Add one edge per ``edge_index`` column, decoding all nine channels."""
    edge_index = getattr(data, "edge_index", None)
    if not isinstance(edge_index, torch.Tensor) or edge_index.numel() == 0:
        return
    edge_attr = getattr(data, "edge_attr", None)
    attrs = edge_attr.long().tolist() if isinstance(edge_attr, torch.Tensor) else None
    width = int(edge_attr.size(1)) if isinstance(edge_attr, torch.Tensor) else 0
    ends = edge_index.long().tolist()

    for column in range(edge_index.size(1)):
        row = attrs[column] if attrs is not None and column < len(attrs) else None
        kind_id = int(row[0]) if row else 0
        kind = decoder.kind(kind_id)
        source = int(ends[0][column])
        target = int(ends[1][column])
        pos_a = int(row[1]) if row and width > 1 else -1
        pos_b = int(row[2]) if row and width > 2 else -1

        relation = relation_kind = ""
        role = category = ""
        sign = goal_level = history_dt = 0
        if row and width >= 9:
            role_id = int(row[4])
            role = decoder.role(role_id)
            relation, relation_kind = decoder.relation(int(row[3]), role_id)
            sign = int(row[5])
            goal_level = int(row[6])
            history_dt = int(row[7])
            category = decoder.category(int(row[8]))

        position = f"[{pos_a},{pos_b}]" if pos_a >= 0 or pos_b >= 0 else ""
        head = relation
        if relation_kind == "action" and head:
            head = f"@{head}"
        if head and sign:
            head = f"not {head}"
        label = " ".join(part for part in (kind, position, head) if part)
        detail = (
            _instance_detail(
                role=role,
                relation=relation,
                relation_kind=relation_kind,
                sign=sign,
                goal_level=goal_level,
                history_dt=history_dt,
                category=category,
            )
            if role
            else kind
        )

        is_self_loop = source == target or kind in SELF_LOOP_KINDS
        if is_self_loop and source in graph.nodes:
            node = graph.nodes[source]
            node["self_loops"] = tuple(node.get("self_loops", ())) + (kind,)
            node["self_loop_labels"] = tuple(node.get("self_loop_labels", ())) + (
                label,
            )

        if kind in REVERSE_KINDS and not include_reverse_edges:
            continue
        if kind == "line_share" and not include_line_shares:
            continue
        if is_self_loop and not include_self_loops:
            continue
        graph.add_edge(
            source,
            target,
            key=column,
            kind=kind,
            kind_id=kind_id,
            position=(pos_a, pos_b),
            relation=relation,
            relation_kind=relation_kind,
            role=role,
            sign=sign,
            goal_level=goal_level,
            history_dt=history_dt,
            category=category,
            label=label,
            detail=detail,
        )


def _hyperedge_attr_rows(data: Any) -> list[list[int]] | None:
    """Return the ``[M, 6]`` hyperedge attribute table, whatever it is named.

    Accepts the contract name ``hyperedge_attr_ids``, the raw native field
    ``hyperedge_attr_rows``, and the pre-contract role-only 1-d vector.
    """
    for name in ("hyperedge_attr_ids", "hyperedge_attr_rows", "hyperedge_role_ids"):
        value = getattr(data, name, None)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            continue
        table = value.long()
        if table.dim() == 1:
            table = table.reshape(-1, 1)
        return table.tolist()
    return None


def _anchor_index(data: Any) -> int | None:
    """Return the anchor node's index when the encoding emitted one.

    The ``anchor_index`` field is present exactly when an anchor was emitted
    (``has_anchor`` is its batch-invariant metadata mirror), so its absence
    is the "no anchor" signal -- there is no ``-1`` sentinel to filter.
    """
    return _scalar_int(getattr(data, "anchor_index", None))


def _instance_nodes(data: Any, count: int) -> list[int | None]:
    """Return ``instance_node_indices`` as validity-masked node ids.

    ``-1`` marks an instance the chosen node universe never reified; it is
    also *not* batch-safe (the node offset turns it into ``offset - 1``), so
    anything outside the node range is treated as absent.
    """
    values = _int_list(getattr(data, "instance_node_indices", None))
    if values is None:
        return [None] * count
    total = int(getattr(data, "num_nodes", 0))
    out: list[int | None] = []
    for index in range(count):
        if index >= len(values):
            out.append(None)
            continue
        node = values[index]
        out.append(node if 0 <= node < total else None)
    return out


def _pad_row(row: Sequence[int]) -> list[int]:
    """Pad a possibly role-only attribute row out to the six channels."""
    out = list(int(value) for value in row)
    return out + [0] * (6 - len(out)) if len(out) < 6 else out[:6]


def _add_hyperedge_nodes(
    graph: Any, data: Any, decoder: _Decoder, labels: Sequence[str]
) -> bool:
    """Insert one labeled pseudo-node per hyperedge; return whether any was."""
    membership = getattr(data, "hyperedge_index", None)
    if not isinstance(membership, torch.Tensor) or membership.numel() == 0:
        return False
    table = _hyperedge_attr_rows(data)
    pairs = membership.long().tolist()

    members: dict[int, list[int]] = {}
    order: list[int] = []
    for member, hyperedge in zip(pairs[0], pairs[1], strict=True):
        key = int(hyperedge)
        if key not in members:
            members[key] = []
            order.append(key)
        members[key].append(int(member))

    instance_nodes = _instance_nodes(data, (len(table) if table else max(order) + 1))
    anchor = _anchor_index(data)
    node_names = getattr(data, "node_names", None)
    for key in order:
        row = _pad_row(table[key]) if table and key < len(table) else [0] * 6
        _add_instance_node(
            graph,
            data,
            decoder,
            node_id=("h", key),
            index=key,
            prefix="h",
            role="hyperedge",
            row=row,
            arguments=members[key],
            labels=labels,
            node_names=node_names,
            instance_node=(instance_nodes[key] if key < len(instance_nodes) else None),
            anchor_node=anchor,
            member_kind="membership",
            member_from_argument=True,
        )
    return True


def _tuple_boundaries(data: Any, count: int) -> list[int] | None:
    """Return the ``M + 1`` CSR boundaries over ``tuple_args``.

    Handles the materialized ``tuple_ptr`` as well as the per-tuple slot
    sizes it is derived from, under either the contract name (``tuple_sizes``)
    or the raw native one (``tuple_slot_sizes``) -- facades that carry the
    tuple channels only implicitly (every objects-only view, since the
    contract forces ``include_tuple_tensors``) may expose either.
    """
    ptr = _int_list(getattr(data, "tuple_ptr", None))
    if ptr is not None and len(ptr) >= count + 1:
        return ptr[: count + 1]
    for name in ("tuple_sizes", "tuple_slot_sizes"):
        sizes = _int_list(getattr(data, name, None))
        if sizes is None:
            continue
        bounds = [0]
        for size in sizes[:count]:
            bounds.append(bounds[-1] + max(0, size))
        return bounds
    return None


def _tuple_attr_rows(data: Any, count: int) -> list[list[int]] | None:
    """Return the ``[T, 6]`` tuple label table in ``x_ids`` column order.

    Prefers the ``tuple_attr_ids`` convenience channel (already shifted to
    ``relation_id + 1``); falls back to the raw per-channel vectors, where
    ``tuple_rel_ids`` holds the **unshifted** relation id.
    """
    stacked = getattr(data, "tuple_attr_ids", None)
    if (
        isinstance(stacked, torch.Tensor)
        and stacked.dim() == 2
        and stacked.size(1) >= 6
    ):
        return stacked.long()[:count].tolist()
    rel_ids = _int_list(getattr(data, "tuple_rel_ids", None))
    if rel_ids is None:
        return None
    channels = [
        _int_list(getattr(data, name, None)) or [0] * count
        for name in (
            "tuple_role_ids",
            "tuple_sign_ids",
            "tuple_level_ids",
            "tuple_dt_ids",
            "tuple_category_ids",
        )
    ]

    def _at(values: list[int], index: int) -> int:
        return int(values[index]) if index < len(values) else 0

    return [
        [
            _at(channels[0], index),
            int(rel_ids[index]) + 1,
            _at(channels[1], index),
            _at(channels[2], index),
            _at(channels[3], index),
            _at(channels[4], index),
        ]
        for index in range(count)
    ]


def _add_tuple_nodes(
    graph: Any, data: Any, decoder: _Decoder, labels: Sequence[str]
) -> bool:
    """Insert one labeled pseudo-node per tuple instance with ordered slots."""
    rel_ids = _int_list(getattr(data, "tuple_rel_ids", None))
    if rel_ids is None:
        return False
    count = len(rel_ids)
    args = _int_list(getattr(data, "tuple_args", None)) or []
    bounds = _tuple_boundaries(data, count)
    table = _tuple_attr_rows(data, count)
    if bounds is None or table is None:
        return False
    instance_nodes = _instance_nodes(data, count)
    anchor = _anchor_index(data)
    node_names = getattr(data, "node_names", None)

    for index in range(count):
        start, stop = bounds[index], bounds[index + 1]
        arguments = [int(value) for value in args[start:stop]]
        row = _pad_row(table[index])
        _add_instance_node(
            graph,
            data,
            decoder,
            node_id=("t", index),
            index=index,
            prefix="t",
            role="tuple",
            row=row,
            arguments=arguments,
            labels=labels,
            node_names=node_names,
            instance_node=instance_nodes[index],
            anchor_node=anchor,
            member_kind="slot",
            member_from_argument=False,
        )
    return True


def _add_instance_node(
    graph: Any,
    data: Any,
    decoder: _Decoder,
    *,
    node_id: Any,
    index: int,
    prefix: str,
    role: str,
    row: Sequence[int],
    arguments: Sequence[int],
    labels: Sequence[str],
    node_names: Any,
    instance_node: int | None,
    anchor_node: int | None,
    member_kind: str,
    member_from_argument: bool,
) -> None:
    """Add one synthesized instance pseudo-node and wire its arguments."""
    del data
    instance_role = decoder.role(int(row[0]))
    relation, relation_kind = decoder.relation(int(row[1]), int(row[0]))
    sign = int(row[2])
    category = decoder.category(int(row[5]))
    argument_labels = [
        labels[node] if 0 <= node < len(labels) else str(node) for node in arguments
    ]
    label = _instance_label(
        index,
        prefix=prefix,
        node_names=node_names,
        # An arity-0 instance "reifies" onto the shared anchor, whose name is
        # "<nullary>" for every such instance -- rebuild the literal instead.
        instance_node=(None if instance_node == anchor_node else instance_node),
        relation=relation,
        relation_kind=relation_kind,
        sign=sign,
        argument_labels=argument_labels,
    )
    detail = _instance_detail(
        role=instance_role,
        relation=relation,
        relation_kind=relation_kind,
        sign=sign,
        goal_level=int(row[3]),
        history_dt=int(row[4]),
        category=category,
    )
    graph.add_node(
        node_id,
        label=label,
        short_label=f"{prefix}{index}",
        instance_label=label.split(" ", 1)[-1],
        role=role,
        role_id=-1,
        instance_role=instance_role,
        relation=relation,
        relation_kind=relation_kind,
        predicate=relation if relation_kind == "predicate" else "",
        action=relation if relation_kind == "action" else "",
        sign=sign,
        goal_level=int(row[3]),
        history_dt=int(row[4]),
        category=category,
        category_id=int(row[5]),
        channel_ids=list(int(value) for value in row),
        arity=len(arguments),
        instance_node=instance_node,
        synthetic=True,
        self_loops=(),
        self_loop_labels=(),
        detail=detail,
    )
    for slot, member in enumerate(arguments):
        if member not in graph.nodes:
            continue
        source, target = (
            (member, node_id) if member_from_argument else (node_id, member)
        )
        graph.add_edge(
            source,
            target,
            key=(member_kind, index, slot),
            kind=member_kind,
            kind_id=-1,
            position=(slot, -1),
            relation=relation,
            relation_kind=relation_kind,
            role=instance_role,
            sign=sign,
            goal_level=int(row[3]),
            history_dt=int(row[4]),
            category=category,
            label=f"{member_kind} [{slot}]",
            detail=f"{detail} arg{slot}",
        )
    if instance_node is not None and instance_node in graph.nodes:
        graph.add_edge(
            node_id,
            instance_node,
            key=("instance", index),
            kind="instance",
            kind_id=-1,
            position=(-1, -1),
            relation=relation,
            relation_kind=relation_kind,
            role=instance_role,
            sign=sign,
            goal_level=int(row[3]),
            history_dt=int(row[4]),
            category=category,
            label="instance",
            detail=f"{detail} reified",
        )


def _add_spd_edges(graph: Any, data: Any) -> bool:
    """Overlay the shortest-path-distance object pairs as labeled edges."""
    src = _int_list(getattr(data, "spd_src", None))
    dst = _int_list(getattr(data, "spd_dst", None))
    dist = _int_list(getattr(data, "spd_dist", None))
    if src is None or dst is None or dist is None:
        return False
    added = False
    for index, (source, target) in enumerate(zip(src, dst, strict=False)):
        if source not in graph.nodes or target not in graph.nodes:
            continue
        distance = dist[index] if index < len(dist) else -1
        graph.add_edge(
            source,
            target,
            key=("spd", index),
            kind="spd",
            kind_id=-1,
            position=(-1, -1),
            relation="",
            relation_kind="",
            role="",
            sign=0,
            goal_level=0,
            history_dt=0,
            category="",
            distance=distance,
            label=f"spd {distance}",
            detail=f"shortest-path distance {distance}",
        )
        added = True
    return added


# --------------------------------------------------------------------------- #
# rendering


def _layout_span(positions: dict[Any, Any]) -> float:
    """Return the larger extent of a layout, for offsetting labels."""
    if len(positions) < 2:
        return 1.0
    xs = [float(point[0]) for point in positions.values()]
    ys = [float(point[1]) for point in positions.values()]
    return max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)


def _normalized(
    positions: dict[Any, Any], *, fill: float = 1.0
) -> dict[Any, tuple[float, float]]:
    """Fit coordinates into the unit square centred on 0.

    ``fill`` allows a bounded anisotropic stretch: a spring layout is rarely
    square, and scaling it uniformly leaves one axis of the panel unused. A
    stretch of at most ``fill`` times the uniform scale fills the canvas
    without distorting the layout into something unrecognisable.
    """
    if not positions:
        return {}
    xs = [float(point[0]) for point in positions.values()]
    ys = [float(point[1]) for point in positions.values()]
    width = max(max(xs) - min(xs), 1e-9)
    height = max(max(ys) - min(ys), 1e-9)
    uniform = 1.0 / max(width, height)
    scale_x = min(1.0 / width, uniform * fill)
    scale_y = min(1.0 / height, uniform * fill)
    mid_x = (max(xs) + min(xs)) / 2.0
    mid_y = (max(ys) + min(ys)) / 2.0
    return {
        node: (
            (float(point[0]) - mid_x) * scale_x,
            (float(point[1]) - mid_y) * scale_y,
        )
        for node, point in positions.items()
    }


def _separated(
    positions: dict[Any, tuple[float, float]], *, iterations: int = 90
) -> dict[Any, tuple[float, float]]:
    """Push apart nodes that a spring layout stacked on top of each other.

    Literals over identical arguments -- ``(on c b)`` as goal, as subgoal and
    at two history steps -- attach to exactly the same objects, so the spring
    forces on them are identical and they land in one illegible pile. A short
    deterministic repulsion relaxation separates them without disturbing the
    overall shape.
    """
    nodes = sorted(positions, key=repr)
    if len(nodes) < 2:
        return dict(positions)
    minimum = min(0.32, max(0.11, 1.15 / (len(nodes) ** 0.5)))
    points = {node: [float(x), float(y)] for node, (x, y) in positions.items()}
    for _ in range(iterations):
        moved = False
        for i, left in enumerate(nodes):
            for right in nodes[i + 1 :]:
                dx = points[right][0] - points[left][0]
                dy = points[right][1] - points[left][1]
                distance = (dx * dx + dy * dy) ** 0.5
                if distance >= minimum:
                    continue
                if distance < 1e-9:
                    # Deterministic tie-break for exactly coincident nodes.
                    angle = 2.399963 * i
                    dx, dy, distance = math.cos(angle), math.sin(angle), 1.0
                shift = (minimum - distance) / 2.0
                unit_x, unit_y = dx / distance, dy / distance
                points[left][0] -= unit_x * shift
                points[left][1] -= unit_y * shift
                points[right][0] += unit_x * shift
                points[right][1] += unit_y * shift
                moved = True
        if not moved:
            break
    return {node: (points[node][0], points[node][1]) for node in nodes}


def _layout_positions(graph: Any) -> dict[Any, tuple[float, float]]:
    """Lay out a derived multigraph readably, component by component.

    Three corrections over a plain ``spring_layout`` of the multigraph:

    * Springs are taken from the **simple** projection. Otherwise every
      forward/reverse pair and every ``line_share`` shortcut multiplies the
      force between the same two nodes and collapses co-argument facts into
      an unreadable pile.
    * Connected components are laid out separately, each into a square whose
      **area is proportional to its node count**, so a lone node does not get
      the same real estate as the main graph.
    * The small components are then packed into columns immediately to the
      right of the main one. A plain spring layout flings them to the far
      corners -- and every objects-only view has one, because the nullary
      anchor's only graph edge is a self-loop -- leaving a panel that is
      mostly whitespace with a dot in the corner.
    """
    import networkx as nx

    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes)
    simple.add_edges_from((u, v) for u, v in graph.edges() if u != v)
    components = sorted(nx.connected_components(simple), key=len, reverse=True)
    if len(components) <= 1:
        return _normalized(
            _separated(nx.spring_layout(simple, seed=0, k=1.6, iterations=400)),
            fill=1.45,
        )

    blocks: list[tuple[float, dict[Any, tuple[float, float]]]] = []
    for component in components:
        nodes = sorted(component, key=repr)
        if len(nodes) == 1:
            local = {nodes[0]: (0.0, 0.0)}
        else:
            local = _normalized(
                _separated(
                    nx.spring_layout(
                        simple.subgraph(nodes), seed=0, k=1.6, iterations=400
                    )
                )
            )
        # side ~ sqrt(n) => area ~ n.
        blocks.append((max(0.9, float(len(nodes)) ** 0.5), local))

    main_side, main_local = blocks[0]
    positions: dict[Any, tuple[float, float]] = {
        node: (x * main_side, y * main_side) for node, (x, y) in main_local.items()
    }
    pad = 0.10 * main_side
    cursor_x = max(x for x, _ in positions.values()) + pad
    column: list[tuple[float, dict[Any, tuple[float, float]]]] = []
    column_height = 0.0

    def _extent(local, side):
        """Actual drawn size of a component, not the area it was allotted."""
        xs = [x * side for x, _ in local.values()]
        ys = [y * side for _, y in local.values()]
        return max(xs) - min(xs), max(ys) - min(ys)

    def _flush(column, column_height, cursor_x):
        """Place one column of small components, vertically centred."""
        width = max((_extent(local, side)[0] for side, local in column), default=0.0)
        cursor_y = column_height / 2.0
        for side, local in column:
            height = _extent(local, side)[1]
            centre_y = cursor_y - height / 2.0
            centre_x = cursor_x + width / 2.0
            for node, (x, y) in local.items():
                positions[node] = (centre_x + x * side, centre_y + y * side)
            cursor_y -= height + pad
        return cursor_x + width + pad

    main_height = max(y for _, y in positions.values()) - min(
        y for _, y in positions.values()
    )
    for side, local in blocks[1:]:
        height = _extent(local, side)[1]
        if column and column_height + pad + height > max(main_height, height):
            cursor_x = _flush(column, column_height, cursor_x)
            column, column_height = [], 0.0
        column_height += height + (pad if column else 0.0)
        column.append((side, local))
    if column:
        _flush(column, column_height, cursor_x)
    return _normalized(positions, fill=1.45)


def legend_handles(
    graphs: Any,
    *,
    hide_reverse_edges: bool = True,
    ringed_kinds: Iterable[str] | None = None,
) -> list[Any]:
    """Build legend handles for the roles and edge kinds present in ``graphs``.

    Accepts one graph or an iterable of them, so a multi-panel figure can
    carry a single shared legend instead of repeating the same vocabulary in
    every panel. The role/kind vocabularies are identical across the whole
    encoder family, so per-panel legends are pure noise.
    """
    import matplotlib.pyplot as plt

    if hasattr(graphs, "nodes"):
        graphs = [graphs]
    roles: set[str] = set()
    kinds: set[str] = set()
    rings: set[str] = set(ringed_kinds or ())
    for graph in graphs:
        for node, attrs in graph.nodes(data=True):
            roles.add(str(attrs.get("role", "object")))
            loops = tuple(attrs.get("self_loops", ()) or ())
            if ringed_kinds is None and loops and not graph.has_edge(node, node):
                rings.add(loops[0])
        for _source, _target, attrs in graph.edges(data=True):
            kind = str(attrs.get("kind", "arg_fwd"))
            if hide_reverse_edges and kind in REVERSE_KINDS:
                continue
            kinds.add(kind)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=ROLE_SHAPES.get(role, "o"),
            linestyle="",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=8,
            label=role,
        )
        for role, color in ROLE_COLORS.items()
        if role in roles
    ]
    handles.extend(
        plt.Line2D(
            [0],
            [0],
            color=color,
            linewidth=max(1.2, width),
            alpha=min(1.0, alpha + 0.4),
            linestyle=style,
            label=kind,
        )
        for kind in KIND_STYLES
        if kind in kinds
        for color, width, alpha, style in [KIND_STYLES[kind]]
    )
    handles.extend(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="none",
            markeredgecolor=KIND_STYLES.get(kind, ("#888888", 1.0, 0.9, "-"))[0],
            markersize=9,
            label=f"{kind} (loop)",
        )
        for kind in sorted(rings)
    )
    return handles


def _emptiest_corner(positions: dict[Any, Any]) -> tuple[float, float, str, str]:
    """Pick the axes corner holding the fewest nodes, for an overlay box."""
    if not positions:
        return 0.99, 0.99, "right", "top"
    xs = [float(point[0]) for point in positions.values()]
    ys = [float(point[1]) for point in positions.values()]
    mid_x = (max(xs) + min(xs)) / 2.0
    mid_y = (max(ys) + min(ys)) / 2.0
    corners = {
        (0.01, 0.99, "left", "top"): sum(
            1 for x, y in zip(xs, ys) if x <= mid_x and y >= mid_y
        ),
        (0.99, 0.99, "right", "top"): sum(
            1 for x, y in zip(xs, ys) if x >= mid_x and y >= mid_y
        ),
        (0.01, 0.01, "left", "bottom"): sum(
            1 for x, y in zip(xs, ys) if x <= mid_x and y <= mid_y
        ),
        (0.99, 0.01, "right", "bottom"): sum(
            1 for x, y in zip(xs, ys) if x >= mid_x and y <= mid_y
        ),
    }
    return min(corners, key=lambda corner: corners[corner])


def _draw_instance_key(
    ax: Any,
    graph: Any,
    overlay_nodes: list[Any],
    positions: dict[Any, Any],
    font_size: int,
) -> None:
    """Print the ``id -> decoded literal`` key for the instance pseudo-nodes.

    Short ids keep the graph readable; the key keeps the encoding legible.
    Dropping the literals entirely would be exactly the silent omission this
    module exists to prevent.
    """

    def _order(node: Any) -> tuple[str, int, str]:
        """Sort h2 before h10: the ids are numbered, not lexicographic."""
        short = str(graph.nodes[node].get("short_label", node))
        digits = "".join(character for character in short if character.isdigit())
        return short.rstrip("0123456789"), int(digits) if digits else 0, short

    entries = [
        (
            str(graph.nodes[node].get("short_label", node)),
            str(graph.nodes[node].get("instance_label", "")),
        )
        for node in sorted(overlay_nodes, key=_order)
    ]
    if not entries:
        return
    width = max(len(short) for short, _ in entries)
    columns = 2 if len(entries) > 12 else 1
    rows = -(-len(entries) // columns)
    lines = []
    for row in range(rows):
        cells = []
        for column in range(columns):
            index = column * rows + row
            if index < len(entries):
                short, text = entries[index]
                cells.append(f"{short:<{width}}  {text}")
        lines.append("   ".join(cells))
    x, y, ha, va = _emptiest_corner(positions)
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=max(5, font_size - 2),
        family="monospace",
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#CCCCCC",
            "alpha": 0.9,
        },
        zorder=5,
    )


def draw_derived(
    graph: Any,
    *,
    ax: Any | None = None,
    with_labels: bool = True,
    edge_labels: bool = False,
    hide_reverse_edges: bool = True,
    layout: Any | None = None,
    node_size: int | None = None,
    font_size: int = 7,
    legend: bool = True,
    instance_labels: Literal["auto", "full", "id", "none"] = "auto",
    node_kwargs: dict | None = None,
    edge_kwargs: dict | None = None,
) -> Any:
    """Render a derived-graph NetworkX multigraph and return the axis.

    Nodes are drawn per role with distinct colors and marker shapes (objects
    largest, the nullary anchor as a hexagon, synthesized instance nodes as
    pentagons/crosses); edges are styled per kind, with reverse-direction
    kinds hidden by default since forward kinds already carry position
    labels.

    ``instance_labels`` controls how synthesized instance pseudo-nodes are
    named. ``"full"`` prints the decoded literal at the node; ``"id"`` prints
    the short id (``h4``, ``t9``) and lists the id -> literal mapping once, in
    the emptiest corner of the panel; ``"auto"`` (the default) switches to
    ``"id"`` past seven instance nodes, where a dozen literals printed inside
    the graph stop being readable.

    Edge filtering never removes nodes: **every node of the graph is drawn**.
    A node whose only incident edge is a self-loop (the nullary/unary
    instances of the objects-only projection) would otherwise be dropped by
    an ``edge_subgraph`` filter and vanish from a picture of an encoding that
    contains it. Nodes carrying a filtered-out self-loop are ringed instead.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    node_kwargs = node_kwargs or {}
    edge_kwargs = edge_kwargs or {}
    if ax is None:
        _, ax = plt.subplots()

    render_graph = graph
    if hide_reverse_edges:
        dropped = [
            (u, v, k)
            for u, v, k, attrs in graph.edges(keys=True, data=True)
            if attrs.get("kind") in REVERSE_KINDS
        ]
        if dropped:
            # Node-preserving filter: edge_subgraph would delete every node
            # that becomes isolated, silently shrinking the picture below the
            # node count of the encoding it depicts.
            render_graph = graph.copy()
            render_graph.remove_edges_from(dropped)

    positions = layout or _layout_positions(render_graph)

    role_groups: dict[str, list[Any]] = {}
    for node, attrs in render_graph.nodes(data=True):
        role_groups.setdefault(str(attrs.get("role", "object")), []).append(node)

    base_size = node_size or 420
    size_by_role: dict[str, int] = {
        "object": base_size,
        "fact": max(160, base_size // 2),
        "anchor": max(180, int(base_size * 0.55)),
        "hyperedge": max(140, base_size // 3),
        "tuple": max(140, base_size // 3),
        "goal": max(200, int(base_size * 0.6)),
        "subgoal": max(200, int(base_size * 0.6)),
        "history": max(180, int(base_size * 0.55)),
        "action": max(180, int(base_size * 0.55)),
    }
    for role, nodes in role_groups.items():
        nx.draw_networkx_nodes(
            render_graph,
            positions,
            nodelist=nodes,
            node_color=ROLE_COLORS.get(role, "#999999"),
            node_shape=ROLE_SHAPES.get(role, "o"),
            node_size=size_by_role.get(role, base_size // 2),
            ax=ax,
            **node_kwargs,
        )

    # Self-loops that the conversion filtered out are re-stated as rings, so
    # a nullary fact never renders as an unexplained isolated dot. Drawn as
    # point markers (sized in points, hence round at any axis aspect ratio,
    # and living in ``ax.lines`` so node counts over ``ax.collections`` stay
    # exact).
    span = _layout_span(positions)
    ringed_kinds: set[str] = set()
    for node, attrs in render_graph.nodes(data=True):
        kinds = tuple(attrs.get("self_loops", ()) or ())
        if not kinds or render_graph.has_edge(node, node):
            continue
        point = positions.get(node)
        if point is None:
            continue
        color = KIND_STYLES.get(kinds[0], ("#888888", 1.0, 0.9, "-"))[0]
        ringed_kinds.add(kinds[0])
        marker_area = size_by_role.get(str(attrs.get("role", "object")), base_size // 2)
        ax.add_line(
            Line2D(
                [float(point[0])],
                [float(point[1])],
                marker="o",
                linestyle="none",
                markersize=1.15 * (marker_area**0.5) + 5.0,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.3,
                alpha=0.95,
                zorder=1,
            )
        )

    kind_groups: dict[str, list[tuple[Any, Any, Any]]] = {}
    for source, target, key, attrs in render_graph.edges(keys=True, data=True):
        kind_groups.setdefault(str(attrs.get("kind", "arg_fwd")), []).append(
            (source, target, key)
        )
    for kind in sorted(kind_groups, key=lambda name: KIND_ZORDER.get(name, 2.0)):
        edges = kind_groups[kind]
        color, width, alpha, style = KIND_STYLES.get(kind, ("#CCCCCC", 1.0, 0.8, "-"))
        curvature = KIND_CURVATURE.get(kind)
        extra = {"connectionstyle": f"arc3,rad={curvature}"} if curvature else {}
        artists = nx.draw_networkx_edges(
            render_graph,
            positions,
            edgelist=[(u, v, k) for u, v, k in edges],
            edge_color=color,
            width=width,
            alpha=alpha,
            style=style,
            arrows=True,
            arrowsize=9,
            ax=ax,
            **extra,
            **edge_kwargs,
        )
        zorder = KIND_ZORDER.get(kind, 2.0)
        for artist in artists if isinstance(artists, list) else [artists]:
            if artist is not None:
                artist.set_zorder(zorder)

    # The spd triplets ARE the payload of the transformer-bias view; without
    # their distance printed the overlay says nothing the clique edges do not.
    spd_labels = {
        (source, target): f"d={attrs.get('distance', '?')}"
        for source, target, attrs in render_graph.edges(data=True)
        if attrs.get("kind") == "spd"
    }
    if spd_labels and not edge_labels:
        nx.draw_networkx_edge_labels(
            render_graph,
            positions,
            edge_labels=spd_labels,
            font_size=max(5, font_size - 1),
            font_color=KIND_STYLES["spd"][0],
            label_pos=0.35,
            bbox={
                "boxstyle": "round,pad=0.1",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.7,
            },
            ax=ax,
        )

    if with_labels:
        labels = {
            node: str(attrs.get("label", node))
            for node, attrs in render_graph.nodes(data=True)
        }
        object_nodes: list[Any] = []
        core_other: list[Any] = []
        overlay_nodes: list[Any] = []
        for node, attrs in render_graph.nodes(data=True):
            if attrs.get("synthetic", False):
                overlay_nodes.append(node)
            elif attrs.get("role") == "object":
                object_nodes.append(node)
            else:
                core_other.append(node)
        centre_x = sum(float(point[0]) for point in positions.values()) / max(
            1, len(positions)
        )
        centre_y = sum(float(point[1]) for point in positions.values()) / max(
            1, len(positions)
        )

        def _pushed(nodes: list[Any], push: float) -> dict[Any, tuple[float, float]]:
            """Move labels radially away from the layout centroid."""
            out: dict[Any, tuple[float, float]] = {}
            for node in nodes:
                if node not in positions:
                    continue
                x, y = float(positions[node][0]), float(positions[node][1])
                dx, dy = x - centre_x, y - centre_y
                norm = (dx * dx + dy * dy) ** 0.5
                if norm < 1e-9:
                    dx, dy, norm = 0.0, -1.0, 1.0
                out[node] = (x + push * dx / norm, y + push * dy / norm)
            return out

        if core_other:
            nx.draw_networkx_labels(
                render_graph,
                _pushed(core_other, 0.038 * span),
                labels={node: labels[node] for node in core_other},
                font_size=font_size,
                ax=ax,
            )
        if object_nodes:
            nx.draw_networkx_labels(
                render_graph,
                positions,
                labels={node: labels[node] for node in object_nodes},
                font_size=font_size + 1,
                font_weight="bold",
                ax=ax,
            )
        mode = instance_labels
        if mode == "auto":
            mode = "id" if len(overlay_nodes) > 7 else "full"
        if mode == "none":
            overlay_nodes = []
        if overlay_nodes:
            # Synthesized instance labels are long (a whole decoded literal).
            # Push each one radially outward from the layout's centroid so it
            # lands in the empty margin rather than across the core cluster it
            # annotates, and back it with a light box.
            push = 0.055 if mode == "full" else 0.03
            offset = _pushed(overlay_nodes, push * span)
            if mode == "id":
                labels = labels | {
                    node: str(render_graph.nodes[node].get("short_label", labels[node]))
                    for node in overlay_nodes
                }
            nx.draw_networkx_labels(
                render_graph,
                offset,
                labels={node: labels[node] for node in offset},
                font_size=max(5, font_size - 1),
                ax=ax,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.65,
                },
            )
            if mode == "id":
                _draw_instance_key(
                    ax, render_graph, overlay_nodes, positions, font_size
                )

    if edge_labels:
        edge_label_map = {
            (source, target): str(attrs.get("label", attrs.get("kind", "")))
            for source, target, _, attrs in render_graph.edges(keys=True, data=True)
        }
        nx.draw_networkx_edge_labels(
            render_graph,
            positions,
            edge_labels=edge_label_map,
            font_size=max(6, font_size - 1),
            ax=ax,
        )

    if legend:
        ax.legend(
            handles=legend_handles(
                render_graph,
                hide_reverse_edges=hide_reverse_edges,
                ringed_kinds=ringed_kinds,
            ),
            loc="best",
            fontsize=max(6, font_size - 1),
            framealpha=0.85,
        )

    # Node labels are Text artists and do not enter matplotlib's autoscale,
    # so the default data limits clip the outermost ones. Reserve a margin.
    if positions:
        xs = [float(point[0]) for point in positions.values()]
        ys = [float(point[1]) for point in positions.values()]
        width = max(max(xs) - min(xs), 1e-6)
        height = max(max(ys) - min(ys), 1e-6)
        ax.set_xlim(min(xs) - 0.16 * width, max(xs) + 0.16 * width)
        ax.set_ylim(min(ys) - 0.10 * height, max(ys) + 0.10 * height)

    ax.set_axis_off()
    return ax


def summarize_derived(data: Any) -> str:
    """Return a short human-readable channel histogram for one graph."""
    decoder = _Decoder(data)
    lines: list[str] = []
    x_ids = getattr(data, "x_ids", None)
    if isinstance(x_ids, torch.Tensor) and x_ids.numel() > 0:
        counts: dict[str, int] = {}
        for value in x_ids[:, 0].long().tolist():
            role = decoder.role(int(value))
            counts[role] = counts.get(role, 0) + 1
        lines.append(
            "roles: "
            + ", ".join(f"{role}={count}" for role, count in sorted(counts.items()))
        )
    edge_attr = getattr(data, "edge_attr", None)
    if isinstance(edge_attr, torch.Tensor) and edge_attr.numel() > 0:
        kinds: dict[str, int] = {}
        for value in edge_attr[:, 0].long().tolist():
            kind = decoder.kind(int(value))
            kinds[kind] = kinds.get(kind, 0) + 1
        lines.append(
            "kinds: "
            + ", ".join(f"{kind}={count}" for kind, count in sorted(kinds.items()))
        )
    if isinstance(x_ids, torch.Tensor) and x_ids.dim() == 2 and x_ids.size(1) >= 6:
        categories: dict[str, int] = {}
        for value in x_ids[:, 5].long().tolist():
            name = decoder.category(int(value))
            categories[name] = categories.get(name, 0) + 1
        lines.append(
            "categories: "
            + ", ".join(f"{name}={count}" for name, count in sorted(categories.items()))
        )
        dts = x_ids[:, 4].long()
        if int(dts.min().item()) != 0 or int(dts.max().item()) != 0:
            lines.append(
                f"history_dt: [{int(dts.min().item())}, {int(dts.max().item())}] "
                f"(signed; offset={_scalar_int(getattr(data, 'history_dt_offset', 0))})"
            )
    extras: list[str] = []
    table = _hyperedge_attr_rows(data)
    if table is not None:
        extras.append(f"hyperedges={len(table)}")
    rel_ids = _int_list(getattr(data, "tuple_rel_ids", None))
    if rel_ids is not None:
        extras.append(f"tuples={len(rel_ids)}")
    spd = _int_list(getattr(data, "spd_dist", None))
    if spd is not None:
        extras.append(f"spd_pairs={len(spd)}")
    if extras:
        lines.append("instances: " + ", ".join(extras))
    return "; ".join(lines)


def filter_nodes_by_roles(graph: Any, roles: Iterable[str]) -> list[Any]:
    """Return nodes whose role attribute is one of ``roles``."""
    wanted = set(roles)
    return [
        node for node, attrs in graph.nodes(data=True) if attrs.get("role") in wanted
    ]


def core_nodes(graph: Any) -> list[Any]:
    """Return the nodes that came from ``x_ids`` rows, excluding overlays.

    ``len(core_nodes(derived_to_networkx(data))) == data.num_nodes`` is the
    invariant that keeps a rendered figure honest about the encoding it
    depicts.
    """
    return [
        node
        for node, attrs in graph.nodes(data=True)
        if not attrs.get("synthetic", False)
    ]
