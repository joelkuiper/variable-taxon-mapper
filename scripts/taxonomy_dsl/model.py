from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

COLUMNS = [
    "order",
    "name",
    "label",
    "tags",
    "parent",
    "codesystem",
    "code",
    "ontologyTermURI",
    "definition",
    "children",
    "mg_draft",
    "definition_summary",
]


@dataclass
class Payload:
    order: str = ""
    label: str = ""
    tags: str = ""
    codesystem: str = ""
    code: str = ""
    ontology_uri: str = ""
    definition: str = ""
    children_col: str = ""
    mg_draft: str = ""
    definition_summary: str = ""


@dataclass
class Node:
    name: str
    parent: Optional[str]
    children: list[str] = field(default_factory=list)
    payload: Payload = field(default_factory=Payload)


@dataclass
class ProvenanceEntry:
    source_order: str
    source_name: str
    source_parent: str
    current_name: str
    current_parent: str
    op: str
    op_id: str
    notes: str = ""


@dataclass
class Tree:
    nodes: dict[str, Node] = field(default_factory=dict)
    roots: list[str] = field(default_factory=list)
    provenance: list[ProvenanceEntry] = field(default_factory=list)
