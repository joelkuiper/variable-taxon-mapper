from __future__ import annotations

from .model import Tree


class LedgerError(Exception):
    pass


def check_ledger(tree: Tree, source_names: set[str]) -> None:
    """Invariant: every source row is represented in provenance, and every
    provenance entry's current_name resolves to an existing node."""
    prov_sources = {p.source_name for p in tree.provenance}
    missing = source_names - prov_sources
    if missing:
        raise LedgerError(
            f"{len(missing)} source rows missing from provenance: {sorted(missing)[:5]}"
        )
    extra = prov_sources - source_names
    if extra:
        raise LedgerError(
            f"{len(extra)} spurious provenance sources: {sorted(extra)[:5]}"
        )
    dangling = [
        (p.source_name, p.current_name)
        for p in tree.provenance
        if p.current_name not in tree.nodes
    ]
    if dangling:
        raise LedgerError(
            f"{len(dangling)} provenance entries point to missing nodes: {dangling[:5]}"
        )
