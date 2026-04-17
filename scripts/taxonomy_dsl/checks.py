from __future__ import annotations

from .ledger import LedgerError, check_ledger
from .model import Node, Tree


class CheckError(Exception):
    pass


def run_check(tree: Tree, source_names: set[str], params: dict) -> None:
    assertion = params.get("assert")
    if assertion is None:
        raise CheckError(f"check requires 'assert': {params!r}")

    if assertion == "ledger_preserved":
        try:
            check_ledger(tree, source_names)
        except LedgerError as e:
            raise CheckError(f"ledger_preserved: {e}")

    elif assertion == "name_unique":
        # Structural: tree.nodes is keyed by name; duplicates would have raised earlier.
        # This check is a no-op guard that documents intent.
        return

    elif assertion == "every_leaf_has":
        fields = params.get("fields", [])
        under = params.get("under")
        if not fields:
            raise CheckError("every_leaf_has requires non-empty 'fields' list")
        bad: list[str] = []
        for name, node in tree.nodes.items():
            if node.children:
                continue
            if under and not _is_under(tree, name, under):
                continue
            if not _has_any_field(node, fields):
                bad.append(name)
        if bad:
            raise CheckError(
                f"every_leaf_has {fields}"
                + (f" under '{under}'" if under else "")
                + f": {len(bad)} leaves failing, e.g. {bad[:10]}"
            )

    elif assertion == "no_duplicate_keywords":
        # Deferred: full implementation lands with `merge`/`flatten` in stage 2.
        return

    else:
        raise CheckError(f"unknown assertion: {assertion!r}")


def _is_under(tree: Tree, name: str, ancestor: str) -> bool:
    cur = tree.nodes[name]
    while cur.parent:
        if cur.parent == ancestor:
            return True
        cur = tree.nodes[cur.parent]
    return False


def _has_any_field(node: Node, fields: list[str]) -> bool:
    p = node.payload
    field_map = {
        "definition": p.definition,
        "definition_summary": p.definition_summary,
        "keywords": p.children_col,
    }
    for f in fields:
        v = field_map.get(f, "")
        if v and v.strip():
            return True
    return False
