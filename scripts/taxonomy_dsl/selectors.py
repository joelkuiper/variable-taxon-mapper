from __future__ import annotations

import re
from typing import Callable

from .model import Node, Tree


class SelectorError(Exception):
    pass


Predicate = Callable[[Tree, Node], bool]


_NUM_CMP = re.compile(r"^(>=|<=|>|<|=)?\s*(\d+)$")


def _parse_num_cmp(val: object) -> tuple[str, int]:
    s = str(val).strip()
    m = _NUM_CMP.match(s)
    if not m:
        raise SelectorError(f"invalid numeric comparison: {val!r}")
    return (m.group(1) or "="), int(m.group(2))


def _cmp(actual: int, op: str, n: int) -> bool:
    if op == "=":
        return actual == n
    if op == ">":
        return actual > n
    if op == "<":
        return actual < n
    if op == ">=":
        return actual >= n
    if op == "<=":
        return actual <= n
    return False


def _subtree_size(tree: Tree, node: Node) -> int:
    count = 0
    stack = list(node.children)
    while stack:
        c = stack.pop()
        count += 1
        stack.extend(tree.nodes[c].children)
    return count


def parse_selector(spec: object) -> Predicate:
    if not isinstance(spec, dict) or len(spec) != 1:
        raise SelectorError(f"selector must be a single-key dict: {spec!r}")
    (key, val), = spec.items()
    if key == "name":
        target = str(val)
        return lambda tree, node: node.name == target
    if key == "re":
        pattern = re.compile(str(val))
        return lambda tree, node: bool(pattern.search(node.name))
    if key == "parent":
        target = str(val)
        return lambda tree, node: node.parent == target
    if key == "children":
        op_, n = _parse_num_cmp(val)
        return lambda tree, node: _cmp(len(node.children), op_, n)
    if key == "descendants":
        op_, n = _parse_num_cmp(val)
        return lambda tree, node: _cmp(_subtree_size(tree, node), op_, n)
    if key == "all":
        subs = [parse_selector(s) for s in val]
        return lambda tree, node: all(s(tree, node) for s in subs)
    if key == "any":
        subs = [parse_selector(s) for s in val]
        return lambda tree, node: any(s(tree, node) for s in subs)
    if key == "not":
        sub = parse_selector(val)
        return lambda tree, node: not sub(tree, node)
    raise SelectorError(f"unknown selector key: {key!r}")


def select(tree: Tree, spec: dict) -> list[Node]:
    pred = parse_selector(spec)
    return [n for n in tree.nodes.values() if pred(tree, n)]
