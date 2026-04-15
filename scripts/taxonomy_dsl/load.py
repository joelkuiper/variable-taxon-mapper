from __future__ import annotations

from pathlib import Path

import pandas as pd

from .model import COLUMNS, Node, Payload, ProvenanceEntry, Tree


class LoadError(Exception):
    pass


def load_csv(path: Path) -> Tree:
    df = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise LoadError(f"CSV missing required columns: {missing}")
    extras = [c for c in df.columns if c not in COLUMNS]
    if extras:
        print(f"[warn] extra columns (passed through unchanged): {extras}")

    dup_rows: dict[str, list[int]] = {}
    for i, n in enumerate(df["name"].tolist()):
        dup_rows.setdefault(n, []).append(i)
    collisions = {n: rs for n, rs in dup_rows.items() if len(rs) > 1}
    if collisions:
        lines = [f"  '{n}' at CSV rows {rs}" for n, rs in collisions.items()]
        raise LoadError("duplicate names in source CSV:\n" + "\n".join(lines))

    tree = Tree()
    for _, row in df.iterrows():
        name = row["name"]
        parent = row["parent"] or None
        payload = Payload(
            order=row["order"],
            label=row["label"],
            tags=row["tags"],
            codesystem=row["codesystem"],
            code=row["code"],
            ontology_uri=row["ontologyTermURI"],
            definition=row["definition"],
            children_col=row["children"],
            mg_draft=row["mg_draft"],
            definition_summary=row["definition_summary"],
        )
        tree.nodes[name] = Node(name=name, parent=parent, payload=payload)

    orphans: list[tuple[str, str]] = []
    for name, node in tree.nodes.items():
        if node.parent is None:
            tree.roots.append(name)
        elif node.parent not in tree.nodes:
            orphans.append((name, node.parent))
        else:
            tree.nodes[node.parent].children.append(name)
    if orphans:
        lines = [f"  '{n}' → '{p}'" for n, p in orphans]
        raise LoadError("parent references that don't exist as names:\n" + "\n".join(lines))

    visited: set[str] = set()

    def dfs(n: str) -> None:
        if n in visited:
            return
        visited.add(n)
        for c in tree.nodes[n].children:
            dfs(c)

    for r in tree.roots:
        dfs(r)
    if len(visited) != len(tree.nodes):
        unreachable = sorted(set(tree.nodes) - visited)
        raise LoadError(
            f"cycle or disconnected node(s) detected; unreachable from roots: {unreachable[:10]}"
        )

    for node in tree.nodes.values():
        node.children.sort()
    tree.roots.sort()

    for _, row in df.iterrows():
        tree.provenance.append(
            ProvenanceEntry(
                source_order=row["order"],
                source_name=row["name"],
                source_parent=row["parent"],
                current_name=row["name"],
                current_parent=row["parent"],
                op="load",
                op_id="load",
                notes="",
            )
        )

    return tree
