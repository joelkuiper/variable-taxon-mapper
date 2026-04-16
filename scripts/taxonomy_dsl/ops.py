"""Tree-transformation operations for the taxonomy DSL.

Five operations: rename, promote, merge, dissolve, add_node.
"""
from __future__ import annotations

import re

from .model import Node, Payload, Tree
from .selectors import select


class OpError(Exception):
    pass


# ═══════════════════════════════════════════════════════════════════════
# rename
# ═══════════════════════════════════════════════════════════════════════

def op_rename(tree: Tree, rule_id: str, params: dict) -> None:
    sel = params.get("select")
    to = params.get("to")
    if sel is None or to is None:
        raise OpError(f"rename requires 'select' and 'to': {params!r}")
    targets = select(tree, sel)
    if not targets:
        return
    pattern = None
    if isinstance(sel, dict) and "re" in sel:
        pattern = re.compile(sel["re"])
    for node in list(targets):
        if pattern is not None and ("$" in to):
            py_template = re.sub(r"\$(\d+)", r"\\\1", to)
            new_name = pattern.sub(py_template, node.name)
        else:
            new_name = to
        _rename_single(tree, node.name, new_name, rule_id)


def _rename_single(tree: Tree, old: str, new: str, rule_id: str) -> None:
    if old == new:
        return
    if new in tree.nodes:
        raise OpError(f"rename collision: '{old}' → '{new}' but '{new}' already exists")
    node = tree.nodes.pop(old)
    node.name = new
    tree.nodes[new] = node
    if node.parent and node.parent in tree.nodes:
        pnode = tree.nodes[node.parent]
        pnode.children = [new if c == old else c for c in pnode.children]
        pnode.children.sort()
    else:
        tree.roots = [new if r == old else r for r in tree.roots]
        tree.roots.sort()
    for child_name in node.children:
        tree.nodes[child_name].parent = new
    for p in tree.provenance:
        if p.current_name == old:
            p.current_name = new
            p.op = "rename"
            p.op_id = rule_id
            p.notes = f"renamed from '{old}'"
        if p.current_parent == old:
            p.current_parent = new


# ═══════════════════════════════════════════════════════════════════════
# promote — move a node (with subtree) under a new parent
# ═══════════════════════════════════════════════════════════════════════

def op_promote(tree: Tree, rule_id: str, params: dict) -> None:
    sel = params.get("select")
    to = params.get("to")
    if sel is None or to is None:
        raise OpError(f"promote requires 'select' and 'to': {params!r}")
    if to not in tree.nodes:
        raise OpError(f"promote destination '{to}' does not exist")
    targets = select(tree, sel)
    for node in list(targets):
        if node.name not in tree.nodes:
            continue
        if node.name == to:
            raise OpError(f"cannot promote '{node.name}' to itself")
        if to in _descendants(tree, node.name):
            raise OpError(f"promote '{node.name}' → '{to}' would create a cycle")
        if node.parent == to:
            continue
        old_parent = node.parent
        if old_parent and old_parent in tree.nodes:
            pnode = tree.nodes[old_parent]
            pnode.children = [c for c in pnode.children if c != node.name]
        else:
            tree.roots = [r for r in tree.roots if r != node.name]
        node.parent = to
        dest = tree.nodes[to]
        if node.name not in dest.children:
            dest.children.append(node.name)
        dest.children.sort()
        for p in tree.provenance:
            if p.current_name == node.name:
                p.current_parent = to
                p.op = "promote"
                p.op_id = rule_id
                p.notes = f"promoted to '{to}' (from '{old_parent or ''}')"


# ═══════════════════════════════════════════════════════════════════════
# dissolve — remove a wrapper, reparent children to parent
# ═══════════════════════════════════════════════════════════════════════

def op_dissolve(tree: Tree, rule_id: str, params: dict) -> None:
    sel = params.get("select")
    policy = params.get("on_collision", "crash")
    keep_kw = bool(params.get("keep_as_keyword", False))
    if sel is None:
        raise OpError(f"dissolve requires 'select': {params!r}")
    targets = select(tree, sel)
    for node in list(targets):
        if node.name not in tree.nodes:
            continue
        _dissolve_single(tree, node, rule_id, policy, keep_kw)


def _dissolve_single(
    tree: Tree, node: Node, rule_id: str, policy: str = "crash",
    keep_kw: bool = False,
) -> None:
    old_name = node.name
    parent_name = node.parent
    if parent_name is None:
        raise OpError(f"dissolve on root node '{old_name}' is not supported")
    parent = tree.nodes[parent_name]

    _assert_no_code_collision(node, parent, old_name, parent_name, policy)

    parent.children = [c for c in parent.children if c != old_name]

    parent.payload.definition = _concat_def(
        parent.payload.definition, node.payload.definition, old_name
    )
    parent.payload.definition_summary = _concat_def(
        parent.payload.definition_summary, node.payload.definition_summary, old_name
    )
    if node.payload.children_col.strip():
        parent.payload.children_col = _concat_children(
            parent.payload.children_col, node.payload.children_col
        )
    if keep_kw:
        parent.payload.children_col = _concat_children(
            parent.payload.children_col, old_name
        )

    for child_name in list(node.children):
        child = tree.nodes[child_name]
        child.parent = parent_name
        if child_name not in parent.children:
            parent.children.append(child_name)
    parent.children.sort()

    del tree.nodes[old_name]

    for p in tree.provenance:
        if p.current_name == old_name:
            p.current_name = parent_name
            p.current_parent = parent.parent or ""
            p.op = "dissolve"
            p.op_id = rule_id
            p.notes = f"dissolved into '{parent_name}'"
        elif p.current_parent == old_name:
            p.current_parent = parent_name


# ═══════════════════════════════════════════════════════════════════════
# merge — fold targets into one node
#
# Two modes:
#   as: "Name"   → classic merge: targets collapse into a node named "Name"
#                   (may be one of the targets or a fresh node)
#   into: "Name" → absorb mode: targets are absorbed INTO the existing node
#                   "Name", which keeps its own payload and children.
#                   Minimum 1 target (replaces the old reroute/flatten ops).
# ═══════════════════════════════════════════════════════════════════════

def op_merge(tree: Tree, rule_id: str, params: dict) -> None:
    sel = params.get("select")
    as_name = params.get("as")
    into_name = params.get("into")
    into_parent = params.get("into_parent")
    policy = params.get("on_collision", "crash")
    keep_kw = bool(params.get("keep_as_keyword", False))

    if sel is None:
        raise OpError(f"merge requires 'select': {params!r}")
    if as_name and into_name:
        raise OpError("merge: 'as' and 'into' are mutually exclusive")
    if not as_name and not into_name:
        raise OpError("merge requires either 'as' or 'into'")

    targets = select(tree, sel)

    if into_name:
        _merge_into(tree, targets, into_name, rule_id, policy, keep_kw)
    else:
        _merge_as(tree, targets, as_name, into_parent, rule_id, policy, keep_kw)


def _merge_into(
    tree: Tree, targets: list[Node], into_name: str,
    rule_id: str, policy: str, keep_kw: bool,
) -> None:
    """Absorb targets INTO an existing node. The destination keeps its
    payload and children; targets are deleted after their payload and
    children migrate over."""
    if not targets:
        return
    if into_name not in tree.nodes:
        raise OpError(f"merge into: destination '{into_name}' does not exist")
    dest = tree.nodes[into_name]
    target_names = {n.name for n in targets}
    if into_name in target_names:
        raise OpError(f"merge into: destination '{into_name}' is itself a target")

    for node in list(targets):
        if node.name not in tree.nodes:
            continue
        old_name = node.name

        _merge_payload_into(dest.payload, node.payload, old_name, policy)

        if keep_kw:
            dest.payload.children_col = _concat_children(
                dest.payload.children_col, old_name
            )

        # Detach from old parent
        if node.parent and node.parent in tree.nodes:
            pnode = tree.nodes[node.parent]
            pnode.children = [c for c in pnode.children if c != old_name]
        elif node.parent is None:
            tree.roots = [r for r in tree.roots if r != old_name]

        # Reparent target's children to destination
        for child_name in list(node.children):
            child = tree.nodes[child_name]
            child.parent = into_name
            if child_name not in dest.children:
                dest.children.append(child_name)

        del tree.nodes[old_name]

        for p in tree.provenance:
            if p.current_name == old_name:
                p.current_name = into_name
                p.current_parent = dest.parent or ""
                p.op = "merge"
                p.op_id = rule_id
                suffix = " (kept as keyword)" if keep_kw else ""
                p.notes = f"merged into '{into_name}'" + suffix
            elif p.current_parent == old_name:
                p.current_parent = into_name

    dest.children.sort()


def _merge_as(
    tree: Tree, targets: list[Node], new_name: str,
    into_parent: str | None, rule_id: str, policy: str, keep_kw: bool,
) -> None:
    """Classic merge: N targets → 1 node named new_name."""
    if len(targets) < 2:
        raise OpError(
            f"merge (as) needs at least 2 matching nodes, got {len(targets)}"
        )

    target_names = {n.name for n in targets}
    for n in targets:
        for other in targets:
            if other.name == n.name:
                continue
            if other.name in _descendants(tree, n.name):
                raise OpError(
                    f"merge: '{n.name}' is an ancestor of '{other.name}'; "
                    f"not supported"
                )

    parents = {n.parent for n in targets}
    if into_parent is not None:
        if into_parent not in tree.nodes:
            raise OpError(f"merge: into_parent '{into_parent}' does not exist")
        new_parent = into_parent
    elif len(parents) == 1:
        only_parent = next(iter(parents))
        if only_parent is None:
            raise OpError("merge: targets are roots; specify 'into_parent'")
        new_parent = only_parent
    else:
        raise OpError(
            f"merge: targets have {len(parents)} different parents {parents}; "
            f"specify 'into_parent'"
        )

    if new_parent in target_names:
        raise OpError(f"merge: into_parent '{new_parent}' is itself a merge target")

    if new_name in tree.nodes and new_name not in target_names:
        raise OpError(
            f"merge: destination name '{new_name}' already exists "
            f"and is not one of the merged targets"
        )

    if new_name in target_names:
        base_node = next(n for n in targets if n.name == new_name)
        base_payload = base_node.payload
        others = [n for n in targets if n.name != new_name]
    else:
        base_node = None
        base_payload = Payload()
        others = list(targets)

    for n in others:
        _merge_payload_into(base_payload, n.payload, n.name, policy)
        if keep_kw:
            base_payload.children_col = _concat_children(
                base_payload.children_col, n.name
            )

    merged_children: list[str] = []
    for n in targets:
        for c in n.children:
            if c in target_names:
                continue
            if c not in merged_children:
                merged_children.append(c)

    for n in targets:
        if n.parent and n.parent in tree.nodes and n.parent not in target_names:
            pnode = tree.nodes[n.parent]
            pnode.children = [c for c in pnode.children if c != n.name]
        elif n.parent is None:
            tree.roots = [r for r in tree.roots if r != n.name]

    for n in targets:
        if n.name != new_name:
            del tree.nodes[n.name]

    if new_name in tree.nodes:
        merged_node = tree.nodes[new_name]
        merged_node.payload = base_payload
        merged_node.children = []
    else:
        merged_node = Node(name=new_name, parent=None, payload=base_payload)
        tree.nodes[new_name] = merged_node

    merged_node.parent = new_parent
    for c in merged_children:
        if c not in tree.nodes:
            continue
        tree.nodes[c].parent = new_name
        if c not in merged_node.children:
            merged_node.children.append(c)
    merged_node.children.sort()

    new_parent_node = tree.nodes[new_parent]
    if new_name not in new_parent_node.children:
        new_parent_node.children.append(new_name)
    new_parent_node.children.sort()

    for p in tree.provenance:
        if p.current_name in target_names:
            p.current_name = new_name
            p.current_parent = new_parent
            p.op = "merge"
            p.op_id = rule_id
            p.notes = f"merged into '{new_name}'"
        elif p.current_parent in target_names:
            p.current_parent = new_name


# ═══════════════════════════════════════════════════════════════════════
# add_node — create a node from scratch (scaffolding)
# ═══════════════════════════════════════════════════════════════════════

def op_add_node(tree: Tree, rule_id: str, params: dict) -> None:
    name = params.get("name")
    parent = params.get("parent")
    definition = params.get("definition", "") or ""
    if not name or not parent:
        raise OpError(f"add_node requires 'name' and 'parent': {params!r}")
    if name in tree.nodes:
        raise OpError(f"add_node: '{name}' already exists")
    if parent not in tree.nodes:
        raise OpError(f"add_node: parent '{parent}' does not exist")
    payload = Payload(definition=definition)
    new_node = Node(name=name, parent=parent, payload=payload)
    tree.nodes[name] = new_node
    pnode = tree.nodes[parent]
    pnode.children.append(name)
    pnode.children.sort()


# ═══════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════

def _descendants(tree: Tree, name: str) -> set[str]:
    out: set[str] = set()
    stack = list(tree.nodes[name].children)
    while stack:
        cur = stack.pop()
        if cur in out:
            continue
        out.add(cur)
        stack.extend(tree.nodes[cur].children)
    return out


def _assert_no_code_collision(
    node: Node, dest: Node, old_name: str, to: str, policy: str = "crash",
) -> None:
    if policy == "keep_destination":
        return
    a, b = node.payload, dest.payload
    for field in ("code", "codesystem", "ontology_uri"):
        av = getattr(a, field).strip()
        bv = getattr(b, field).strip()
        if av and bv and av != bv:
            raise OpError(
                f"'{old_name}' → '{to}': {field} collision "
                f"('{av}' vs '{bv}'); set 'on_collision: keep_destination' "
                f"to force or resolve in DSL"
            )


def _merge_payload_into(
    base: Payload, incoming: Payload, origin: str, policy: str = "crash"
) -> None:
    for field in ("code", "codesystem", "ontology_uri"):
        av = getattr(base, field).strip()
        bv = getattr(incoming, field).strip()
        if av and bv and av != bv:
            if policy == "keep_destination":
                continue
            raise OpError(
                f"merge: {field} collision on '{origin}' ('{av}' vs '{bv}'); "
                f"set 'on_collision: keep_destination' to force or resolve in DSL"
            )
        if bv and not av:
            setattr(base, field, bv)
    for field in ("label", "tags", "mg_draft"):
        av = getattr(base, field).strip()
        bv = getattr(incoming, field).strip()
        if bv and bv != av:
            merged = f"{av}; {bv}" if av else bv
            setattr(base, field, merged)
    base.definition = _concat_def(base.definition, incoming.definition, origin)
    base.definition_summary = _concat_def(
        base.definition_summary, incoming.definition_summary, origin
    )
    if incoming.children_col.strip():
        base.children_col = _concat_children(base.children_col, incoming.children_col)
    bo = base.order.strip()
    io = incoming.order.strip()
    if io and (not bo or (io.isdigit() and bo.isdigit() and int(io) < int(bo))):
        base.order = io


def _concat_def(existing: str, incoming: str, origin: str) -> str:
    """Concatenate source definitions so that a downstream summariser has all the
    material to work from. No audit tags — origin information lives in provenance.csv.
    Plain space-join is enough; the LLM tolerates duplicates and ordering noise."""
    incoming = (incoming or "").strip()
    if not incoming:
        return existing
    if not existing.strip():
        return incoming
    return existing.rstrip() + " " + incoming


def _concat_children(existing: str, incoming: str) -> str:
    existing_items = [s.strip() for s in existing.split(",") if s.strip()]
    incoming_items = [s.strip() for s in incoming.split(",") if s.strip()]
    seen: set[str] = set()
    merged: list[str] = []
    for item in existing_items + incoming_items:
        norm = item.lower()
        if norm not in seen:
            seen.add(norm)
            merged.append(item)
    return ",".join(merged)
