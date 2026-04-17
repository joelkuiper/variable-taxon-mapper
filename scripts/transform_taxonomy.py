#!/usr/bin/env python3
"""Transform a keywords taxonomy CSV via a DSL script.

Usage:
    python scripts/transform_taxonomy.py \\
        --in data/Keywords_summarized_new.csv \\
        --dsl transforms/noop.yaml \\
        --out data/Keywords_cleaned.csv \\
        --provenance data/provenance.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from taxonomy_dsl.load import load_csv
from taxonomy_dsl.runner import load_script, run_script
from taxonomy_dsl.write import write_provenance_csv, write_tree_csv


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Transform a taxonomy CSV via a DSL script."
    )
    ap.add_argument("--in", dest="input", type=Path, required=True)
    ap.add_argument("--dsl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--provenance", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    print(f"Loading {args.input} ...")
    tree = load_csv(args.input)
    source_names = {p.source_name for p in tree.provenance}
    print(
        f"  {len(tree.nodes)} nodes, {len(tree.roots)} roots, "
        f"{len(source_names)} source rows"
    )

    print(f"Loading DSL {args.dsl} ...")
    script = load_script(args.dsl)
    n_rules = len(script.get("rules", []))
    print(f"  {n_rules} rules")

    print("Running transforms ...")
    run_script(tree, script, source_names, verbose=args.verbose)
    print(f"  after: {len(tree.nodes)} nodes, {len(tree.roots)} roots")

    if args.dry_run:
        print("[dry-run] skipping writes")
        return 0

    print(f"Writing {args.out} ...")
    write_tree_csv(tree, args.out)
    print(f"Writing {args.provenance} ...")
    write_provenance_csv(tree, args.provenance)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
