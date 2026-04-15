from __future__ import annotations

import re

from .model import Node, Payload, Tree
from .selectors import select


class OpError(Exception):
    pass


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


def op_reroute(tree: Tree, rule_id: str, params: dict) -> None:
    sel = params.get("select")
    to = params.get("to")
    keep_as_keyword = bool(params.get("keep_as_keyword", False))
    policy = params.get("on_collision", "crash")
    if sel is None or to is None:
        raise OpError(f"reroute requires 'select' and 'to': {params!r}")
    if to not in tree.nodes:
        raise OpError(f"reroute destination '{to}' does not exist")
    targets = select(tree, sel)
    for node in list(targets):
        if node.name not in tree.nodes:
            continue
        if node.name == to:
            raise OpError(f"cannot reroute node '{node.name}' to itself")
        if to in _descendants(tree, node.name):
            raise OpError(f"reroute '{node.name}' → '{to}' would create a cycle")
        _reroute_single(tree, node, to, keep_as_keyword, rule_id, policy)


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


def _reroute_single(
    tree: Tree,
    node: Node,
    to: str,
    keep_as_keyword: bool,
    rule_id: str,
    policy: str = "crash",
) -> None:
    old_name = node.name
    old_parent = node.parent
    dest = tree.nodes[to]

    _assert_no_code_collision(node, dest, old_name, to, policy)

    if old_parent and old_parent in tree.nodes:
        pnode = tree.nodes[old_parent]
        pnode.children = [c for c in pnode.children if c != old_name]
    else:
        tree.roots = [r for r in tree.roots if r != old_name]

    dest.payload.definition = _concat_def(
        dest.payload.definition, node.payload.definition, old_name
    )
    dest.payload.definition_summary = _concat_def(
        dest.payload.definition_summary, node.payload.definition_summary, old_name
    )
    if node.payload.children_col.strip():
        dest.payload.children_col = _concat_children(
            dest.payload.children_col, node.payload.children_col
        )
    if keep_as_keyword:
        dest.payload.children_col = _concat_children(
            dest.payload.children_col, old_name
        )

    for child_name in list(node.children):
        child = tree.nodes[child_name]
        child.parent = to
        if child_name not in dest.children:
            dest.children.append(child_name)
    dest.children.sort()

    del tree.nodes[old_name]

    for p in tree.provenance:
        if p.current_name == old_name:
            p.current_name = to
            p.current_parent = dest.parent or ""
            p.op = "reroute"
            p.op_id = rule_id
            suffix = " (kept as keyword)" if keep_as_keyword else ""
            p.notes = f"rerouted to '{to}'" + suffix
        elif p.current_parent == old_name:
            p.current_parent = to


def _assert_no_code_collision(
    node: Node,
    dest: Node,
    old_name: str,
    to: str,
    policy: str = "crash",
) -> None:
    """policy: 'crash' (default) raises on any code/codesystem/uri collision.
    'keep_destination' silently resolves collisions by keeping `dest`'s values."""
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


def op_add_node(tree: Tree, rule_id: str, params: dict) -> None:
    """Create a new node with no source-row provenance (it isn't in the source CSV).
    Used to add scaffolding (e.g., new ICD subchapters, missing categories)."""
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
        _promote_single(tree, node, to, rule_id)


def _promote_single(tree: Tree, node: Node, to: str, rule_id: str) -> None:
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


def op_dissolve(tree: Tree, rule_id: str, params: dict) -> None:
    sel = params.get("select")
    policy = params.get("on_collision", "crash")
    if sel is None:
        raise OpError(f"dissolve requires 'select': {params!r}")
    targets = select(tree, sel)
    for node in list(targets):
        if node.name not in tree.nodes:
            continue
        _dissolve_single(tree, node, rule_id, policy)


def _dissolve_single(
    tree: Tree, node: Node, rule_id: str, policy: str = "crash"
) -> None:
    old_name = node.name
    parent_name = node.parent
    if parent_name is None:
        raise OpError(
            f"dissolve on root node '{old_name}' is not supported; use reroute"
        )
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


def op_merge(tree: Tree, rule_id: str, params: dict) -> None:
    sel = params.get("select")
    new_name = params.get("as")
    into_parent = params.get("into_parent")
    policy = params.get("on_collision", "crash")
    if sel is None or new_name is None:
        raise OpError(f"merge requires 'select' and 'as': {params!r}")
    if "facets" in params:
        print(f"[warn] merge id={rule_id}: 'facets' is accepted but unused in v1")
    targets = select(tree, sel)
    if len(targets) < 2:
        raise OpError(
            f"merge needs at least 2 matching nodes, got {len(targets)} "
            f"(selector: {sel!r})"
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
        raise OpError(
            f"merge: into_parent '{new_parent}' is itself a merge target"
        )

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


def op_flatten(tree: Tree, rule_id: str, params: dict) -> None:
    sel = params.get("select")
    absorb = params.get("absorb") or {}
    definition_suffix = params.get("definition_suffix")
    policy = params.get("on_collision", "crash")
    if sel is None:
        raise OpError(f"flatten requires 'select': {params!r}")
    match_re = re.compile(absorb["match"]) if "match" in absorb else None
    parents = select(tree, sel)
    for parent in list(parents):
        if parent.name not in tree.nodes:
            continue
        _flatten_parent(tree, parent, match_re, definition_suffix, rule_id, policy)


def _flatten_parent(
    tree: Tree,
    parent: Node,
    match_re: re.Pattern | None,
    definition_suffix: str | None,
    rule_id: str,
    policy: str = "crash",
) -> None:
    absorbed: list[tuple[Node, re.Match | None]] = []
    for child_name in list(parent.children):
        if child_name not in tree.nodes:
            continue
        child = tree.nodes[child_name]
        if match_re is not None:
            m = match_re.search(child_name)
            if m is None:
                continue
            absorbed.append((child, m))
        else:
            absorbed.append((child, None))
    if not absorbed:
        return

    group_values: dict[str, list[str]] = {}
    for _, m in absorbed:
        if m is None:
            continue
        for k, v in m.groupdict().items():
            if v is None:
                continue
            group_values.setdefault(k, [])
            if v not in group_values[k]:
                group_values[k].append(v)

    for child, _ in absorbed:
        _absorb_child_into_parent(
            tree, parent, child, rule_id, op_label="flatten", policy=policy
        )

    if definition_suffix:
        suffix = _apply_template(definition_suffix, group_values)
        existing = parent.payload.definition
        if existing.strip():
            parent.payload.definition = existing.rstrip() + " " + suffix
        else:
            parent.payload.definition = suffix


def op_extract_facet(tree: Tree, rule_id: str, params: dict) -> None:
    sel = params.get("select")
    pattern_str = params.get("pattern")
    group_by = params.get("group_by")
    name_template = params.get("name_template")
    definition_suffix = params.get("definition_suffix")
    policy = params.get("on_collision", "crash")
    if not (sel and pattern_str and group_by and name_template):
        raise OpError(
            "extract_facet requires 'select', 'pattern', 'group_by', 'name_template'"
        )
    pattern = re.compile(pattern_str)
    targets = select(tree, sel)
    if not targets:
        return

    groups: dict[tuple[str, str], list[tuple[Node, re.Match]]] = {}
    for node in targets:
        m = pattern.search(node.name)
        if m is None:
            continue
        gd = m.groupdict()
        if group_by not in gd or gd[group_by] is None:
            continue
        parent_name = node.parent or ""
        key = (parent_name, gd[group_by])
        groups.setdefault(key, []).append((node, m))

    for (parent_name, _group_val), members in groups.items():
        if not parent_name:
            raise OpError("extract_facet: cannot group root nodes")
        _extract_group(
            tree, parent_name, members, name_template, definition_suffix, rule_id, policy
        )


def _extract_group(
    tree: Tree,
    parent_name: str,
    members: list[tuple[Node, re.Match]],
    name_template: str,
    definition_suffix: str | None,
    rule_id: str,
    policy: str = "crash",
) -> None:
    group_values: dict[str, list[str]] = {}
    for _, m in members:
        for k, v in m.groupdict().items():
            if v is None:
                continue
            group_values.setdefault(k, [])
            if v not in group_values[k]:
                group_values[k].append(v)

    new_name = _apply_template(name_template, group_values)
    if not new_name:
        raise OpError(
            f"extract_facet: name_template produced empty name "
            f"for {[n.name for n, _ in members]}"
        )

    parent = tree.nodes[parent_name]
    member_names = {n.name for n, _ in members}

    if new_name in tree.nodes and new_name not in member_names:
        raise OpError(
            f"extract_facet: new name '{new_name}' collides with existing node"
        )

    if new_name in tree.nodes:
        base = tree.nodes[new_name]
        others = [(n, m) for n, m in members if n.name != new_name]
    else:
        base = Node(name=new_name, parent=parent_name, payload=Payload())
        tree.nodes[new_name] = base
        if new_name not in parent.children:
            parent.children.append(new_name)
        parent.children.sort()
        others = list(members)

    for child, _m in others:
        _absorb_child_into_parent(
            tree, base, child, rule_id, op_label="extract_facet", policy=policy
        )

    base.children.sort()

    if definition_suffix:
        suffix = _apply_template(definition_suffix, group_values)
        existing = base.payload.definition
        if existing.strip():
            base.payload.definition = existing.rstrip() + " " + suffix
        else:
            base.payload.definition = suffix


def _absorb_child_into_parent(
    tree: Tree,
    parent: Node,
    child: Node,
    rule_id: str,
    op_label: str,
    policy: str = "crash",
) -> None:
    """Absorb `child` into `parent`: merge payload, reparent grandchildren,
    keep child's name as a keyword on parent, delete child, update provenance."""
    old_name = child.name
    parent.children = [c for c in parent.children if c != old_name]
    _merge_payload_into(parent.payload, child.payload, old_name, policy)
    parent.payload.children_col = _concat_children(parent.payload.children_col, old_name)
    for gc_name in list(child.children):
        gc = tree.nodes[gc_name]
        gc.parent = parent.name
        if gc_name not in parent.children:
            parent.children.append(gc_name)
    parent.children.sort()
    del tree.nodes[old_name]
    for p in tree.provenance:
        if p.current_name == old_name:
            p.current_name = parent.name
            p.current_parent = parent.parent or ""
            p.op = op_label
            p.op_id = rule_id
            p.notes = f"{op_label} into '{parent.name}'"
        elif p.current_parent == old_name:
            p.current_parent = parent.name


_TEMPLATE_RE = re.compile(r"\{([^}]+)\}")


def _apply_template(template: str, group_values: dict[str, list[str]]) -> str:
    def repl(m: re.Match) -> str:
        expr = m.group(1)
        parts = expr.split("|", 1)
        key = parts[0].strip()
        if key not in group_values:
            return m.group(0)
        vals = group_values[key]
        if len(parts) == 2:
            filt = parts[1].strip()
            if filt.startswith("join:"):
                sep = filt[len("join:"):].strip().strip("'\"")
                return sep.join(vals)
        return ", ".join(vals)

    return _TEMPLATE_RE.sub(repl, template)


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
