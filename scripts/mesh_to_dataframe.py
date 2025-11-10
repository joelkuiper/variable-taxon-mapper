#!/usr/bin/env python3
"""
Convert the MeSH XML descriptor file into a flat CSV taxonomy export.

Features:
- pulls identifier, label, definition, definition_summary
- resolves ALL parents from tree numbers
- can optionally *explode* multiple parents into multiple rows
  so that each row has at most one parent (good for strict taxonomies)

Usage:
    python mesh_to_csv.py desc2024.xml mesh.csv
    python mesh_to_csv.py --explode-parents desc2024.xml mesh.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence
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
    # 1) normal descriptor name
    label = _clean(record.findtext("DescriptorName/String"))
    if label:
        return label
    # 2) fall back to a term on the preferred concept
    if preferred is not None:
        for term in preferred.findall("TermList/Term/String"):
            candidate = _clean(term.text)
            if candidate:
                return candidate
    # 3) absolute fallback to identifier
    return fallback


def _collect_tree_number_index(root: ET.Element) -> dict[str, str]:
    """
    Map tree number -> descriptor identifier.

    We only ever walk *one level up* (strip the last segment), so it's okay
    if MeSH reuses tree number prefixes; we'll still end up with a sensible parent.
    """
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

        # definition: ScopeNote → ConceptDescription → Annotation
        definition = ""
        if preferred is not None:
            definition = _clean(preferred.findtext("ScopeNote"))
            if not definition:
                definition = _clean(preferred.findtext("ConceptDescription/String"))
        if not definition:
            definition = _clean(record.findtext("Annotation"))

        # make a short version
        if len(definition) > 320:
            definition_summary = f"{definition[:319].rstrip()}…"
        else:
            definition_summary = definition

        tree_numbers: list[str] = []
        parent_ids: set[str] = set()

        for tn_element in record.findall("TreeNumberList/TreeNumber"):
            tree_number = _clean(tn_element.text)
            if not tree_number:
                continue
            tree_numbers.append(tree_number)

            # derive parent tree
            if "." in tree_number:
                parent_tree = tree_number.rsplit(".", 1)[0]
                parent_id = tree_index.get(parent_tree)
                if parent_id and parent_id != identifier:
                    parent_ids.add(parent_id)

        yield MeshEntry(
            identifier=identifier,
            label=label,
            definition=definition,
            definition_summary=definition_summary,
            parent_ids=tuple(sorted(parent_ids)),
            tree_numbers=tuple(tree_numbers),
        )


def _entry_depth(entry: MeshEntry) -> int:
    """
    Approximate depth by the shallowest tree number.
    This helps us write parents before children, even if identifiers don't line up.
    """
    if not entry.tree_numbers:
        return 0
    return min(tn.count(".") for tn in entry.tree_numbers)


def _write_csv_flat(entries: Sequence[MeshEntry], output_path: Path) -> None:
    """
    Original behavior: one row per descriptor, multiple parents packed.
    """
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


def _write_csv_exploded(entries: Sequence[MeshEntry], output_path: Path) -> None:
    """
    Exploded behavior: one row per (descriptor, parent).
    Root nodes (no parents) still get one row.
    """
    fieldnames = [
        "identifier",
        "label",
        "definition",
        "definition_summary",
        "parent",
        "parents",       # keep the full set if you still want it
        "tree_numbers",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            parents_joined = "|".join(entry.parent_ids)
            tree_numbers_joined = "|".join(entry.tree_numbers)

            # if no parents → still write one row
            if not entry.parent_ids:
                writer.writerow(
                    {
                        "identifier": entry.identifier,
                        "label": entry.label,
                        "definition": entry.definition,
                        "definition_summary": entry.definition_summary,
                        "parent": "",
                        "parents": parents_joined,
                        "tree_numbers": tree_numbers_joined,
                    }
                )
                continue

            # else: one row per parent
            for parent in entry.parent_ids:
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
    parser.add_argument("input", type=Path, help="Path to the MeSH descriptor XML (e.g. desc2024.xml)")
    parser.add_argument("output", type=Path, help="Destination CSV path")
    parser.add_argument(
        "--explode-parents",
        action="store_true",
        help="Write one row per parent instead of packing them into '|'",
    )
    args = parser.parse_args()

    root = ET.parse(args.input).getroot()
    entries = list(_iter_entries(root))

    # sort so parents come before children
    entries.sort(key=lambda e: (_entry_depth(e), e.identifier))

    if args.explode_parents:
        _write_csv_exploded(entries, args.output)
    else:
        _write_csv_flat(entries, args.output)


if __name__ == "__main__":
    main()
