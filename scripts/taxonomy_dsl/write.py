from __future__ import annotations

from pathlib import Path

import pandas as pd

from .model import COLUMNS, Tree


def write_tree_csv(tree: Tree, path: Path) -> None:
    rows = []
    for name, node in tree.nodes.items():
        p = node.payload
        rows.append(
            {
                "order": p.order,
                "name": name,
                "label": p.label,
                "tags": p.tags,
                "parent": node.parent or "",
                "codesystem": p.codesystem,
                "code": p.code,
                "ontologyTermURI": p.ontology_uri,
                "definition": p.definition,
                "children": p.children_col,
                "mg_draft": p.mg_draft,
                "definition_summary": p.definition_summary,
            }
        )
    rows.sort(key=lambda r: (r["parent"], r["name"]))
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(path, index=False)


def write_provenance_csv(tree: Tree, path: Path) -> None:
    rows = []
    for e in tree.provenance:
        rows.append(
            {
                "source_order": e.source_order,
                "source_name": e.source_name,
                "source_parent": e.source_parent,
                "current_name": e.current_name,
                "current_parent": e.current_parent,
                "op": e.op,
                "op_id": e.op_id,
                "notes": e.notes,
            }
        )
    rows.sort(key=lambda r: (r["source_name"],))
    df = pd.DataFrame(
        rows,
        columns=[
            "source_order",
            "source_name",
            "source_parent",
            "current_name",
            "current_parent",
            "op",
            "op_id",
            "notes",
        ],
    )
    df.to_csv(path, index=False)
