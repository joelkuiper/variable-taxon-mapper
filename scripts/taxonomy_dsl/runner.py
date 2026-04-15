from __future__ import annotations

from pathlib import Path

import yaml

from .checks import run_check
from .ledger import LedgerError, check_ledger
from .model import Tree
from .ops import (
    op_add_node,
    op_dissolve,
    op_extract_facet,
    op_flatten,
    op_merge,
    op_promote,
    op_rename,
    op_reroute,
)

OPS = {
    "rename": op_rename,
    "reroute": op_reroute,
    "promote": op_promote,
    "dissolve": op_dissolve,
    "merge": op_merge,
    "flatten": op_flatten,
    "extract_facet": op_extract_facet,
    "add_node": op_add_node,
}


def load_script(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"DSL script must be a mapping at top level: {path}")
    return data


def run_script(
    tree: Tree,
    script: dict,
    source_names: set[str],
    verbose: bool = False,
) -> None:
    rules = script.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError("'rules' must be a list")
    for i, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"rule {i} is not a mapping: {rule!r}")
        rule_id = rule.get("id") or f"rule_{i}"
        op_keys = [k for k in rule.keys() if k in OPS or k == "check"]
        if len(op_keys) != 1:
            raise ValueError(
                f"rule {i} ({rule_id}) must have exactly one op key "
                f"(one of {sorted(list(OPS) + ['check'])}); got {op_keys}"
            )
        op_key = op_keys[0]
        params = rule[op_key] or {}
        if not isinstance(params, dict):
            raise ValueError(f"rule {i} ({rule_id}) {op_key} params must be a mapping")

        if verbose:
            print(f"[rule {i}] {op_key} id={rule_id}")

        if op_key == "check":
            run_check(tree, source_names, params)
        else:
            OPS[op_key](tree, rule_id, params)
            try:
                check_ledger(tree, source_names)
            except LedgerError as e:
                raise LedgerError(
                    f"rule {i} ({op_key} id={rule_id}) broke ledger: {e}"
                )
