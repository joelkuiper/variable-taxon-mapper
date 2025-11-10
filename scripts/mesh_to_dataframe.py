#!/usr/bin/env python3
"""Convert the MeSH XML descriptor file into a flat CSV taxonomy export.

The output is intentionally compact so it can be consumed by the keyword
importer that expects ``identifier``, ``parent``, ``label`` and definition
columns.  The script keeps the parsing logic small and dependency free which
makes it easier to run in constrained environments.

Usage::

    uv run python scripts/mesh_to_dataframe.py desc2024.xml mesh.csv

The generated CSV uses UTF-8 and writes a header row.  Multiple parents are
represented using ``|`` separators so downstream tooling can pick the first
match when a strict tree structure is required.  Each descriptor is guaranteed
to have a ``label`` by falling back to the preferred term so that root-level
nodes never appear with bare identifier codes in downstream trees.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
import xml.etree.ElementTree as ET


@dataclass(slots=True)
class MeshEntry:
    identifier: str
    label: str
    definition: str
    definition_summary: str
    parent_ids: tuple[str, ...]
    tree_numbers: tuple[str, ...]


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def _preferred_concept(record: ET.Element) -> ET.Element | None:
    concepts = record.findall("ConceptList/Concept")
    for concept in concepts:
        if concept.get("PreferredConceptYN", "N") == "Y":
            return concept
    return concepts[0] if concepts else None


def _extract_label(
    record: ET.Element, preferred: ET.Element | None, fallback: str
) -> str:
    label = _clean(record.findtext("DescriptorName/String"))
    if label:
        return label
    if preferred is not None:
        for term in preferred.findall("TermList/Term/String"):
            candidate = _clean(term.text)
            if candidate:
                return candidate
    return fallback


def _collect_tree_number_index(root: ET.Element) -> dict[str, str]:
    index: dict[str, str] = {}
    for record in root.findall("DescriptorRecord"):
        identifier = _clean(record.findtext("DescriptorUI"))
        if not identifier:
            continue
        for tn in record.findall("TreeNumberList/TreeNumber"):
            tree_number = _clean(tn.text)
            if tree_number:
                index[tree_number] = identifier
    return index


def _iter_entries(root: ET.Element) -> Iterator[MeshEntry]:
    tree_index = _collect_tree_number_index(root)

    for record in root.findall("DescriptorRecord"):
        identifier = _clean(record.findtext("DescriptorUI"))
        if not identifier:
            continue

        preferred = _preferred_concept(record)
        label = _extract_label(record, preferred, identifier)
        definition = ""
        if preferred is not None:
            definition = _clean(preferred.findtext("ScopeNote"))
            if not definition:
                definition = _clean(preferred.findtext("ConceptDescription/String"))
        if not definition:
            definition = _clean(record.findtext("Annotation"))

        tree_numbers: list[str] = []
        parent_ids: set[str] = set()
        for tn_element in record.findall("TreeNumberList/TreeNumber"):
            tree_number = _clean(tn_element.text)
            if not tree_number:
                continue
            tree_numbers.append(tree_number)
            if "." not in tree_number:
                continue
            parent_tree = tree_number.rsplit(".", 1)[0]
            parent_id = tree_index.get(parent_tree)
            if parent_id and parent_id != identifier:
                parent_ids.add(parent_id)

        if len(definition) > 320:
            cutoff = max(1, 320 - 1)
            definition_summary = f"{definition[:cutoff].rstrip()}…"
        else:
            definition_summary = definition

        yield MeshEntry(
            identifier=identifier,
            label=label,
            definition=definition,
            definition_summary=definition_summary,
            parent_ids=tuple(sorted(parent_ids)),
            tree_numbers=tuple(tree_numbers),
        )


def _write_csv(entries: Iterable[MeshEntry], output_path: Path) -> None:
    fieldnames = [
        "identifier",
        "label",
        "definition",
        "definition_summary",
        "parent",
        "parents",
        "tree_numbers",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            parents_joined = "|".join(entry.parent_ids)
            tree_numbers_joined = "|".join(entry.tree_numbers)
            parent = entry.parent_ids[0] if entry.parent_ids else ""
            writer.writerow(
                {
                    "identifier": entry.identifier,
                    "label": entry.label,
                    "definition": entry.definition,
                    "definition_summary": entry.definition_summary,
                    "parent": parent,
                    "parents": parents_joined,
                    "tree_numbers": tree_numbers_joined,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the MeSH descriptor XML into CSV"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the MeSH descriptor XML file (e.g. desc2024.xml)",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Destination CSV path",
    )
    args = parser.parse_args()

    root = ET.parse(args.input).getroot()
    entries = list(_iter_entries(root))
    entries.sort(key=lambda entry: entry.identifier)
    _write_csv(entries, args.output)


if __name__ == "__main__":  # pragma: no cover - convenience script
    main()
