from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import typer

from vtm.taxonomy import build_taxonomy_graph, ensure_traversal_cache, path_to_root
from vtm.utils import ensure_file_exists, load_table

from .app import app, logger


SYNTHETIC_ROOT_ID = "__taxonomy_root__"
SYNTHETIC_ROOT_NAME = "Taxonomy"


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def _clean_prediction_label(value: Any) -> list[str]:
    text = _clean_text(value)
    if not text:
        return []
    parts = text.split("|") if "|" in text else [text]
    labels: list[str] = []
    for part in parts:
        cleaned = part.strip().strip('"').strip("'")
        if cleaned:
            labels.append(cleaned)
    return labels


def _palette_for_roots(roots: List[str]) -> Dict[str, str]:
    colors = [
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc948",
        "#b07aa1",
        "#ff9da7",
        "#9c755f",
        "#bab0ab",
        "#86bcb6",
        "#f1ce63",
        "#8cd17d",
        "#d4a6c8",
        "#79706e",
        "#5f9ed1",
        "#c85252",
        "#7bba8e",
        "#cfcfcf",
    ]
    return {name: colors[idx % len(colors)] for idx, name in enumerate(roots)}


def _format_path(parts: List[str]) -> str:
    return " / ".join(parts)


def _coerce_order(value: Any) -> float:
    if value is None or pd.isna(value):
        return math.inf
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.inf


def _node_identifier(name: str) -> str:
    return f"node:{name}"


def _build_payload(
    taxonomy_df: pd.DataFrame,
    results_df: pd.DataFrame,
    *,
    prediction_column: str,
    title: str,
) -> dict[str, Any]:
    required_taxonomy_cols = {"name", "parent"}
    missing_taxonomy_cols = sorted(required_taxonomy_cols - set(taxonomy_df.columns))
    if missing_taxonomy_cols:
        raise typer.BadParameter(
            f"Taxonomy table is missing required columns: {', '.join(missing_taxonomy_cols)}"
        )
    if prediction_column not in results_df.columns:
        raise typer.BadParameter(
            f"Results table does not contain prediction column '{prediction_column}'"
        )

    canonical = taxonomy_df.copy()
    canonical["name"] = canonical["name"].map(_clean_text)
    canonical["parent"] = canonical["parent"].map(_clean_text)
    canonical = canonical.dropna(subset=["name"]).copy()

    if canonical["name"].duplicated().any():
        duplicates = canonical.loc[canonical["name"].duplicated(), "name"].tolist()
        sample = ", ".join(sorted(dict.fromkeys(duplicates))[:10])
        raise typer.BadParameter(
            f"Taxonomy contains duplicate names; expected unique node labels. Sample: {sample}"
        )

    if "order" not in canonical.columns:
        canonical["order"] = pd.NA

    graph = build_taxonomy_graph(
        canonical,
        name_col="name",
        parent_col="parent",
        order_col="order",
    )
    cache = ensure_traversal_cache(graph)

    info_by_name: dict[str, dict[str, Any]] = {}
    for row in canonical.to_dict(orient="records"):
        name = row["name"]
        if not isinstance(name, str):
            continue
        info_by_name[name] = row

    order_map = {
        str(row["name"]): _coerce_order(row.get("order"))
        for row in canonical.to_dict(orient="records")
        if isinstance(row.get("name"), str)
    }

    def _child_sort_key(node_name: str) -> tuple[float, str]:
        return (order_map.get(node_name, math.inf), node_name.lower())

    roots = [name for name in graph.nodes if graph.in_degree(name) == 0]
    roots.sort(key=_child_sort_key)
    root_palette = _palette_for_roots(roots)

    exact_counts: dict[str, int] = {name: 0 for name in graph.nodes}
    unmatched_counts: dict[str, int] = {}
    total_prediction_rows = 0

    for value in results_df[prediction_column].tolist():
        labels = _clean_prediction_label(value)
        if not labels:
            continue
        total_prediction_rows += 1
        for label in labels:
            if label in exact_counts:
                exact_counts[label] += 1
            else:
                unmatched_counts[label] = unmatched_counts.get(label, 0) + 1

    total_prediction_assignments = sum(exact_counts.values()) + sum(unmatched_counts.values())

    def _top_root_for(node_name: str) -> str:
        parts = path_to_root(graph, node_name)
        return parts[0] if parts else node_name

    flat_nodes: list[dict[str, Any]] = []
    flat_links: list[dict[str, Any]] = []

    def _build_node(name: str, parent_name: Optional[str], depth: int) -> dict[str, Any]:
        row = info_by_name.get(name, {})
        child_names = sorted(graph.successors(name), key=_child_sort_key)
        children = [_build_node(child, name, depth + 1) for child in child_names]
        self_count = int(exact_counts.get(name, 0))
        subtree_count = self_count + sum(child["subtree_count"] for child in children)
        top_root = _top_root_for(name)
        path_parts = path_to_root(graph, name)
        path_text = _format_path(path_parts)
        definition_summary = _clean_text(row.get("definition_summary"))
        definition = _clean_text(row.get("definition"))

        node = {
            "id": _node_identifier(name),
            "name": name,
            "label": _clean_text(row.get("label")) or name,
            "parent_id": _node_identifier(parent_name) if parent_name else SYNTHETIC_ROOT_ID,
            "depth": depth,
            "order": order_map.get(name, math.inf),
            "path": path_parts,
            "path_text": path_text,
            "top_root": top_root,
            "color": root_palette[top_root],
            "self_count": self_count,
            "subtree_count": subtree_count,
            "share_global": (
                subtree_count / total_prediction_assignments if total_prediction_assignments else 0.0
            ),
            "definition": definition or "",
            "definition_summary": definition_summary or "",
            "children": children,
        }

        flat_nodes.append(
            {
                "id": node["id"],
                "name": node["name"],
                "label": node["label"],
                "parent_id": node["parent_id"],
                "depth": node["depth"],
                "path_text": node["path_text"],
                "top_root": node["top_root"],
                "self_count": node["self_count"],
                "subtree_count": node["subtree_count"],
                "share_global": node["share_global"],
                "color": node["color"],
            }
        )

        for child in children:
            flat_links.append(
                {
                    "source": node["id"],
                    "target": child["id"],
                    "count": child["subtree_count"],
                    "share_global": (
                        child["subtree_count"] / total_prediction_assignments
                        if total_prediction_assignments
                        else 0.0
                    ),
                    "share_parent": (
                        child["subtree_count"] / subtree_count if subtree_count else 0.0
                    ),
                    "color": child["color"],
                }
            )

        return node

    tree_children = [_build_node(root_name, None, 1) for root_name in roots]
    total_subtree_count = sum(child["subtree_count"] for child in tree_children)

    synthetic_root = {
        "id": SYNTHETIC_ROOT_ID,
        "name": SYNTHETIC_ROOT_NAME,
        "label": title,
        "parent_id": None,
        "depth": 0,
        "order": -1,
        "path": [SYNTHETIC_ROOT_NAME],
        "path_text": SYNTHETIC_ROOT_NAME,
        "top_root": SYNTHETIC_ROOT_NAME,
        "color": "#31424f",
        "self_count": 0,
        "subtree_count": total_subtree_count,
        "share_global": (
            total_subtree_count / total_prediction_assignments
            if total_prediction_assignments
            else 0.0
        ),
        "definition": "",
        "definition_summary": "",
        "children": tree_children,
    }

    flat_nodes.insert(
        0,
        {
            "id": synthetic_root["id"],
            "name": synthetic_root["name"],
            "label": synthetic_root["label"],
            "parent_id": None,
            "depth": 0,
            "path_text": synthetic_root["path_text"],
            "top_root": synthetic_root["top_root"],
            "self_count": synthetic_root["self_count"],
            "subtree_count": synthetic_root["subtree_count"],
            "share_global": synthetic_root["share_global"],
            "color": synthetic_root["color"],
        },
    )
    for child in tree_children:
        flat_links.insert(
            0,
            {
                "source": synthetic_root["id"],
                "target": child["id"],
                "count": child["subtree_count"],
                "share_global": (
                    child["subtree_count"] / total_prediction_assignments
                    if total_prediction_assignments
                    else 0.0
                ),
                "share_parent": (
                    child["subtree_count"] / total_subtree_count if total_subtree_count else 0.0
                ),
                "color": child["color"],
            },
        )

    depth_map = cache.get("depth_map", {})
    max_depth = max((int(depth) for depth in depth_map.values() if depth is not None), default=0)
    leaf_count = sum(1 for name in graph.nodes if graph.out_degree(name) == 0)
    matched_nodes = sum(1 for count in exact_counts.values() if count > 0)

    return {
        "meta": {
            "title": title,
            "taxonomy_node_count": len(graph.nodes),
            "taxonomy_edge_count": len(graph.edges),
            "taxonomy_leaf_count": leaf_count,
            "taxonomy_root_count": len(roots),
            "max_depth": max_depth,
            "prediction_row_count": int(len(results_df)),
            "prediction_rows_with_labels": int(total_prediction_rows),
            "prediction_assignments": int(total_prediction_assignments),
            "matched_node_count": matched_nodes,
            "unmatched_label_count": len(unmatched_counts),
            "unmatched_labels": unmatched_counts,
            "prediction_column": prediction_column,
            "description": (
                "Rectangular phylogeny-style tree where branch thickness encodes how many "
                "VTM predictions fall within each branch and its descendants."
            ),
        },
        "tree": synthetic_root,
        "nodes": flat_nodes,
        "links": flat_links,
        "root_palette": root_palette,
    }


def _build_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, indent=2).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{payload["meta"]["title"]}</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --panel: #ffffff;
      --ink: #15222b;
      --muted: #5d6c78;
      --accent: #275d8c;
      --grid: rgba(39, 93, 140, 0.08);
      --line: rgba(21, 34, 43, 0.12);
      --shadow: 0 18px 40px rgba(21, 34, 43, 0.08);
      --font-ui: "Avenir Next", "Segoe UI", sans-serif;
      --font-display: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      color: var(--ink);
      font-family: var(--font-ui);
      background: var(--bg);
    }}

    .shell {{
      display: grid;
      grid-template-columns: 360px 1fr;
      min-height: 100vh;
      gap: 18px;
      padding: 18px;
    }}

    .panel {{
      background: var(--panel);
      border: 1px solid rgba(29, 42, 49, 0.08);
      border-radius: 20px;
      box-shadow: var(--shadow);
    }}

    .sidebar {{
      padding: 20px 18px;
      position: sticky;
      top: 18px;
      height: calc(100vh - 36px);
      overflow: auto;
    }}

    .title {{
      margin: 0 0 6px;
      font: 700 1.75rem/1.1 var(--font-display);
      letter-spacing: 0.01em;
    }}

    .subtitle {{
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.45;
      font-size: 0.96rem;
    }}

    .controls {{
      display: grid;
      gap: 14px;
      margin-bottom: 18px;
    }}

    .control-group {{
      padding: 14px;
      border-radius: 16px;
      background: #f8fbff;
      border: 1px solid rgba(29, 42, 49, 0.06);
    }}

    .control-group label,
    .metric-label {{
      display: block;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    .toggle-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}

    button,
    select,
    input[type="search"],
    input[type="number"],
    input[type="range"] {{
      font: inherit;
    }}

    .toggle {{
      appearance: none;
      border: 1px solid rgba(29, 42, 49, 0.10);
      background: rgba(255, 255, 255, 0.76);
      color: var(--ink);
      padding: 9px 12px;
      border-radius: 999px;
      cursor: pointer;
      transition: background 140ms ease, border-color 140ms ease, transform 140ms ease;
    }}

    .toggle:hover {{
      transform: translateY(-1px);
    }}

    .toggle.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #f8f7f3;
    }}

    .action-button {{
      width: 100%;
      appearance: none;
      border: 1px solid rgba(29, 42, 49, 0.10);
      background: #ffffff;
      color: var(--ink);
      padding: 10px 12px;
      border-radius: 12px;
      cursor: pointer;
      transition: background 140ms ease, border-color 140ms ease, transform 140ms ease;
    }}

    .action-button:hover {{
      transform: translateY(-1px);
      background: #f8fbff;
    }}

    .search {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(29, 42, 49, 0.12);
      background: rgba(255, 255, 255, 0.84);
      color: var(--ink);
    }}

    select {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(29, 42, 49, 0.12);
      background: #ffffff;
      color: var(--ink);
    }}

    .range-row {{
      display: grid;
      gap: 10px;
    }}

    .count-row {{
      display: grid;
      grid-template-columns: 1fr 120px;
      gap: 10px;
      align-items: center;
    }}

    .count-row input[type="number"] {{
      width: 100%;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid rgba(29, 42, 49, 0.12);
      background: rgba(255, 255, 255, 0.84);
    }}

    .metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 18px;
    }}

    .metric {{
      padding: 12px;
      border-radius: 14px;
      background: #f8fbff;
      border: 1px solid rgba(29, 42, 49, 0.06);
    }}

    .metric-value {{
      font: 700 1.2rem/1.1 var(--font-display);
    }}

    .metric-small {{
      font-size: 0.82rem;
      color: var(--muted);
      margin-top: 4px;
    }}

    .details {{
      padding: 14px;
      border-radius: 16px;
      background: #f8fbff;
      border: 1px solid rgba(29, 42, 49, 0.06);
      min-height: 220px;
    }}

    .details h2 {{
      margin: 0 0 6px;
      font: 700 1.2rem/1.2 var(--font-display);
    }}

    .path {{
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.45;
      margin-bottom: 14px;
    }}

    .definition {{
      line-height: 1.55;
      color: var(--ink);
      font-size: 0.95rem;
    }}

    .stats {{
      display: grid;
      gap: 8px;
      margin-top: 16px;
    }}

    .stat {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding-bottom: 8px;
      border-bottom: 1px dashed rgba(29, 42, 49, 0.10);
      font-size: 0.92rem;
    }}

    .viz-panel {{
      padding: 16px;
      overflow: auto;
      min-height: 100vh;
    }}

    .viz-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 12px;
      align-items: baseline;
      flex-wrap: wrap;
    }}

    .viz-caption {{
      color: var(--muted);
      max-width: 820px;
      line-height: 1.45;
      font-size: 0.95rem;
    }}

    .status-line {{
      margin-bottom: 12px;
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.45;
    }}

    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      margin-bottom: 12px;
    }}

    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 0.84rem;
    }}

    .legend-swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      border: 1px solid rgba(29, 42, 49, 0.10);
      flex: 0 0 auto;
    }}

    #chart-wrap {{
      border-radius: 18px;
      background:
        linear-gradient(90deg, var(--grid) 1px, transparent 1px),
        linear-gradient(180deg, var(--grid) 1px, transparent 1px),
        #ffffff;
      background-size: 120px 120px, 120px 120px, auto;
      border: 1px solid rgba(29, 42, 49, 0.08);
      overflow: auto;
      min-height: 720px;
    }}

    svg {{
      display: block;
    }}

    .link {{
      fill: none;
      stroke-linecap: round;
      stroke-linejoin: round;
      transition: opacity 120ms ease;
    }}

    .link.hidden,
    .node.hidden,
    .label.hidden {{
      opacity: 0.08;
    }}

    .node-halo {{
      fill: rgba(255, 255, 255, 0.96);
      stroke: rgba(21, 34, 43, 0.18);
      stroke-width: 1.5px;
    }}

    .node-dot {{
      stroke: rgba(21, 34, 43, 0.30);
      stroke-width: 0.9px;
    }}

    .label {{
      fill: var(--ink);
      font-size: 13px;
      letter-spacing: 0.01em;
      user-select: none;
    }}

    .label-box {{
      fill: rgba(255, 255, 255, 0.92);
      stroke: rgba(21, 34, 43, 0.10);
      stroke-width: 1px;
    }}

    .label-box.selected {{
      fill: rgba(255, 255, 255, 0.98);
      stroke: var(--accent);
      stroke-width: 1.6px;
    }}

    .label.internal {{
      fill: rgba(29, 42, 49, 0.70);
      font-size: 12px;
    }}

    .label.dim {{
      fill: rgba(29, 42, 49, 0.34);
    }}

    .scale-bar {{
      font-size: 12px;
      fill: var(--muted);
    }}

    .tooltip {{
      position: fixed;
      pointer-events: none;
      z-index: 10;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(22, 31, 36, 0.94);
      color: #f6f1e6;
      box-shadow: 0 16px 34px rgba(22, 31, 36, 0.28);
      border: 1px solid rgba(255, 255, 255, 0.10);
      min-width: 220px;
      max-width: 320px;
      line-height: 1.45;
      font-size: 0.86rem;
      opacity: 0;
      transform: translateY(4px);
      transition: opacity 100ms ease, transform 100ms ease;
    }}

    .tooltip.visible {{
      opacity: 1;
      transform: translateY(0);
    }}

    .hint {{
      margin-top: 14px;
      color: var(--muted);
      font-size: 0.82rem;
      line-height: 1.45;
    }}

    @media (max-width: 1100px) {{
      .shell {{
        grid-template-columns: 1fr;
      }}

      .sidebar {{
        position: static;
        height: auto;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar panel">
      <h1 class="title">{payload["meta"]["title"]}</h1>
      <p class="subtitle">{payload["meta"]["description"]}</p>

      <div class="metrics" id="summary-metrics"></div>

      <div class="controls">
        <section class="control-group">
          <label for="sort-select">Sort Branches</label>
          <select id="sort-select">
            <option value="predictions">Most VTM predictions</option>
            <option value="taxonomy">Taxonomy order</option>
            <option value="alpha">A-Z</option>
          </select>
        </section>

        <section class="control-group">
          <label for="search-input">Search Taxa</label>
          <input class="search" id="search-input" type="search" placeholder="Search node names or paths">
        </section>

        <section class="control-group">
          <label for="min-count-input">Min VTM Predictions</label>
          <div class="range-row">
            <input id="count-slider" type="range" min="0" max="0" value="0">
            <div class="count-row">
              <div id="count-caption" class="metric-small">Showing all branches</div>
              <input id="min-count-input" type="number" min="0" value="0">
            </div>
          </div>
        </section>

        <section class="control-group">
          <button class="action-button" id="reset-view" type="button">Reset View</button>
        </section>
      </div>

      <section class="details" id="details-panel"></section>
      <p class="hint">Click a node to inspect it. Double-click an internal node to collapse or expand its branch. Branch width tracks VTM predictions for that branch plus descendants.</p>
    </aside>

    <main class="viz-panel panel">
      <div class="viz-head">
        <div>
          <div class="metric-label">Viewer</div>
          <div class="viz-caption">Rectangular phylogeny-style view only. Branch order can be switched between taxonomy order, alphabetical order, and prediction volume to keep the labels readable.</div>
        </div>
      </div>
      <div class="status-line" id="status-line"></div>
      <div class="legend" id="legend"></div>
      <div id="chart-wrap">
        <svg id="chart" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Taxonomy visualization"></svg>
      </div>
    </main>
  </div>

  <div class="tooltip" id="tooltip"></div>

  <script>
    const DATA = {payload_json};
  </script>
  <script>
    const state = {{
      sortMode: "predictions",
      minCount: 0,
      collapsed: new Set(),
      selectedId: null,
      query: "",
    }};

    const chartWrap = document.getElementById("chart-wrap");
    const tooltip = document.getElementById("tooltip");
    const detailsPanel = document.getElementById("details-panel");
    const metricsPanel = document.getElementById("summary-metrics");
    const legendPanel = document.getElementById("legend");
    const statusLine = document.getElementById("status-line");
    const sortSelect = document.getElementById("sort-select");
    const searchInput = document.getElementById("search-input");
    const countSlider = document.getElementById("count-slider");
    const minCountInput = document.getElementById("min-count-input");
    const countCaption = document.getElementById("count-caption");
    const resetViewButton = document.getElementById("reset-view");

    const nodeIndex = new Map();
    function indexTree(node) {{
      nodeIndex.set(node.id, node);
      for (const child of node.children || []) {{
        indexTree(child);
      }}
    }}
    indexTree(DATA.tree);
    const allNodes = Array.from(nodeIndex.values());

    const totalAssignments = DATA.meta.prediction_assignments || 0;
    const maxDepth = Math.max(...allNodes.map((node) => node.depth));
    const maxCount = Math.max(...allNodes.map((node) => node.subtree_count));
    const sliderStep = Math.max(1, Math.floor(maxCount / 400));
    countSlider.max = String(maxCount);
    countSlider.step = String(sliderStep);
    minCountInput.max = String(maxCount);

    function formatInteger(value) {{
      return new Intl.NumberFormat("en-US").format(value);
    }}

    function formatPercent(value) {{
      return `${{(value * 100).toFixed(value >= 0.1 ? 1 : 2)}}%`;
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    function sortModeLabel(mode) {{
      if (mode === "taxonomy") {{
        return "Taxonomy order";
      }}
      if (mode === "alpha") {{
        return "A-Z";
      }}
      return "Most VTM predictions";
    }}

    function buildLegend() {{
      const topRoots = [...new Set(allNodes
        .filter((node) => node.depth === 1)
        .map((node) => node.name))]
        .slice(0, 12);
      legendPanel.innerHTML = topRoots.map((name) => {{
        const color = DATA.root_palette[name];
        return `<span class="legend-item"><span class="legend-swatch" style="background:${{color}}"></span>${{escapeHtml(name)}}</span>`;
      }}).join("");
    }}

    function renderSummaryMetrics() {{
      const items = [
        ["Nodes", formatInteger(DATA.meta.taxonomy_node_count), `${{formatInteger(DATA.meta.taxonomy_leaf_count)}} leaves`],
        ["Predictions", formatInteger(DATA.meta.prediction_assignments), `${{formatInteger(DATA.meta.prediction_rows_with_labels)}} labeled rows`],
        ["Roots", formatInteger(DATA.meta.taxonomy_root_count), `depth ${{formatInteger(DATA.meta.max_depth)}}`],
        ["Matched taxa", formatInteger(DATA.meta.matched_node_count), `${{formatInteger(DATA.meta.unmatched_label_count)}} unmatched labels`],
      ];
      metricsPanel.innerHTML = items.map(([label, value, detail]) => `
        <div class="metric">
          <div class="metric-label">${{label}}</div>
          <div class="metric-value">${{value}}</div>
          <div class="metric-small">${{detail}}</div>
        </div>
      `).join("");
    }}

    function findMatches(query) {{
      if (!query) {{
        return new Set();
      }}
      const lowered = query.toLowerCase();
      const matches = new Set();
      for (const node of allNodes) {{
        const haystacks = [node.name, node.label, node.path_text];
        if (haystacks.some((value) => (value || "").toLowerCase().includes(lowered))) {{
          matches.add(node.id);
        }}
      }}
      return matches;
    }}

    function computeHighlights(matchIds) {{
      const highlighted = new Set(matchIds);
      for (const id of matchIds) {{
        let current = nodeIndex.get(id);
        while (current && current.parent_id) {{
          highlighted.add(current.id);
          current = nodeIndex.get(current.parent_id);
        }}
      }}
      if (state.selectedId) {{
        let current = nodeIndex.get(state.selectedId);
        while (current) {{
          highlighted.add(current.id);
          current = current.parent_id ? nodeIndex.get(current.parent_id) : null;
        }}
      }}
      return highlighted;
    }}

    function filterTree(node, minCount) {{
      const children = [];
      for (const child of node.children || []) {{
        const kept = filterTree(child, minCount);
        if (kept) {{
          children.push(kept);
        }}
      }}
      const keepSelf = node.id === DATA.tree.id || node.subtree_count >= minCount;
      if (!keepSelf && children.length === 0) {{
        return null;
      }}
      return {{
        ...node,
        children,
      }};
    }}

    function sortTree(node) {{
      const sortedChildren = (node.children || []).map(sortTree);
      sortedChildren.sort((left, right) => {{
        if (state.sortMode === "alpha") {{
          return left.name.localeCompare(right.name, undefined, {{ sensitivity: "base" }});
        }}
        if (state.sortMode === "taxonomy") {{
          const orderDiff = (left.order ?? Number.POSITIVE_INFINITY) - (right.order ?? Number.POSITIVE_INFINITY);
          if (orderDiff !== 0) {{
            return orderDiff;
          }}
          return left.name.localeCompare(right.name, undefined, {{ sensitivity: "base" }});
        }}
        if (right.subtree_count !== left.subtree_count) {{
          return right.subtree_count - left.subtree_count;
        }}
        if (right.self_count !== left.self_count) {{
          return right.self_count - left.self_count;
        }}
        return left.name.localeCompare(right.name, undefined, {{ sensitivity: "base" }});
      }});
      return {{
        ...node,
        children: sortedChildren,
      }};
    }}

    function applyCollapsedState(node) {{
      const next = {{
        ...node,
        children: (node.children || []).map(applyCollapsedState),
      }};
      if (state.collapsed.has(node.id)) {{
        next._collapsedChildren = next.children;
        next.children = [];
      }}
      return next;
    }}

    function assignRectangularLayout(root) {{
      const margin = {{ top: 32, right: 430, bottom: 64, left: 96 }};
      const leafSpacing = 26;
      const depthStep = 235;
      let leafIndex = 0;
      const nodes = [];
      const links = [];

      function walk(node, parent = null) {{
        node._visibleChildren = node.children || [];
        if (node._visibleChildren.length === 0) {{
          node._leafIndex = leafIndex++;
          node._y = margin.top + node._leafIndex * leafSpacing;
        }} else {{
          for (const child of node._visibleChildren) {{
            walk(child, node);
          }}
          const ys = node._visibleChildren.map((child) => child._y);
          node._y = ys.reduce((sum, value) => sum + value, 0) / ys.length;
        }}
        node._x = margin.left + node.depth * depthStep;
        nodes.push(node);
        if (parent) {{
          links.push({{ source: parent, target: node }});
        }}
      }}

      walk(root, null);
      const width = margin.left + (maxDepth + 1) * depthStep + margin.right;
      const height = Math.max(780, margin.top + Math.max(1, leafIndex - 1) * leafSpacing + margin.bottom);
      return {{ nodes, links, width, height }};
    }}

    function thicknessFor(count) {{
      if (!totalAssignments) {{
        return 1.25;
      }}
      const share = count / totalAssignments;
      return 1.25 + Math.sqrt(share) * 48;
    }}

    function circleRadius(node) {{
      if (node.depth === 0) {{
        return 0;
      }}
      return 3.4 + Math.sqrt(Math.max(0, node.self_count)) * 0.22;
    }}

    function linkOpacity(link, highlightIds) {{
      if (!state.query && !state.selectedId) {{
        return 0.82;
      }}
      const active = highlightIds.has(link.target.id) || highlightIds.has(link.source.id);
      return active ? 0.96 : 0.10;
    }}

    function nodeOpacity(node, highlightIds) {{
      if (node.depth === 0) {{
        return 0;
      }}
      if (!state.query && !state.selectedId) {{
        return 0.95;
      }}
      return highlightIds.has(node.id) ? 0.98 : 0.16;
    }}

    function showTooltip(event, node) {{
      const share = totalAssignments ? node.subtree_count / totalAssignments : 0;
      tooltip.innerHTML = `
        <strong>${{escapeHtml(node.name)}}</strong><br>
        <span>${{escapeHtml(node.path_text)}}</span><br>
        VTM predictions: ${{formatInteger(node.subtree_count)}} (${{formatPercent(share)}})
      `;
      tooltip.classList.add("visible");
      tooltip.style.left = `${{event.clientX + 14}}px`;
      tooltip.style.top = `${{event.clientY + 14}}px`;
    }}

    function hideTooltip() {{
      tooltip.classList.remove("visible");
    }}

    function updateDetails(node) {{
      const active = node || DATA.tree;
      const share = totalAssignments ? active.subtree_count / totalAssignments : 0;
      const shareParent = active.parent_id
        ? (() => {{
            const parent = nodeIndex.get(active.parent_id);
            if (!parent || !parent.subtree_count) {{
              return 0;
            }}
            return active.subtree_count / parent.subtree_count;
          }})()
        : 1;
      const description = active.definition_summary || active.definition || "No definition available for this node.";
      detailsPanel.innerHTML = `
        <h2>${{escapeHtml(active.label || active.name)}}</h2>
        <div class="path">${{escapeHtml(active.path_text)}}</div>
        <div class="definition">${{escapeHtml(description)}}</div>
        <div class="stats">
          <div class="stat"><span>VTM predictions</span><strong>${{formatInteger(active.subtree_count)}}</strong></div>
          <div class="stat"><span>Share of all VTM predictions</span><strong>${{formatPercent(share)}}</strong></div>
          <div class="stat"><span>Share of parent branch</span><strong>${{formatPercent(shareParent)}}</strong></div>
          <div class="stat"><span>Depth</span><strong>${{formatInteger(active.depth)}}</strong></div>
        </div>
      `;
    }}

    function rectangularPath(source, target) {{
      return `M${{source._x.toFixed(2)}},${{source._y.toFixed(2)}} H${{target._x.toFixed(2)}} V${{target._y.toFixed(2)}}`;
    }}

    function fitLabelBoxes(svg) {{
      svg.querySelectorAll("[data-label-id]").forEach((group) => {{
        const text = group.querySelector("text");
        const rect = group.querySelector("rect");
        if (!text || !rect) {{
          return;
        }}
        const bbox = text.getBBox();
        rect.setAttribute("x", (bbox.x - 4).toFixed(2));
        rect.setAttribute("y", (bbox.y - 2).toFixed(2));
        rect.setAttribute("width", Math.max(24, bbox.width + 8).toFixed(2));
        rect.setAttribute("height", Math.max(16, bbox.height + 4).toFixed(2));
      }});
    }}

    function render() {{
      const query = state.query.trim();
      const matchIds = findMatches(query);
      const highlightIds = computeHighlights(matchIds);
      const filtered = filterTree(DATA.tree, state.minCount);
      const sortedTree = sortTree(filtered);
      const working = applyCollapsedState(sortedTree);
      const layout = assignRectangularLayout(working);

      const parts = [];
      parts.push(`<svg id="chart" width="${{layout.width}}" height="${{layout.height}}" viewBox="0 0 ${{layout.width}} ${{layout.height}}" xmlns="http://www.w3.org/2000/svg">`);

      for (const depth of Array.from({{ length: maxDepth }}, (_, idx) => idx + 1)) {{
        const x = 96 + depth * 235;
        parts.push(`<line x1="${{x}}" y1="12" x2="${{x}}" y2="${{layout.height - 24}}" stroke="rgba(39,93,140,0.08)" stroke-dasharray="4 8" />`);
        parts.push(`<text x="${{x + 8}}" y="24" class="scale-bar">depth ${{depth}}</text>`);
      }}

      for (const link of layout.links) {{
        if (link.target.depth === 0) {{
          continue;
        }}
        const opacity = linkOpacity(link, highlightIds);
        const strokeWidth = thicknessFor(link.target.subtree_count);
        parts.push(
          `<path class="link" d="${{rectangularPath(link.source, link.target)}}" stroke="${{link.target.color}}" stroke-width="${{strokeWidth.toFixed(2)}}" opacity="${{opacity.toFixed(3)}}" data-target="${{link.target.id}}" data-source="${{link.source.id}}" />`
        );
      }}

      for (const node of layout.nodes) {{
        if (node.depth === 0) {{
          continue;
        }}
        const selected = state.selectedId === node.id;
        const opacity = nodeOpacity(node, highlightIds);
        const innerRadius = selected ? Math.max(5.6, circleRadius(node) + 1.8) : circleRadius(node);
        const haloRadius = innerRadius + 2.6;
        parts.push(
          `<g class="node" data-node-id="${{node.id}}" opacity="${{opacity.toFixed(3)}}" transform="translate(${{node._x.toFixed(2)}},${{node._y.toFixed(2)}})">
            <circle class="node-halo" r="${{haloRadius.toFixed(2)}}" />
            <circle class="node-dot" r="${{innerRadius.toFixed(2)}}" fill="${{node.color}}" />
          </g>`
        );

        const isLeaf = (node.children || []).length === 0;
        const isMatch = highlightIds.has(node.id);
        if (isLeaf || node.depth <= 2 || isMatch || state.selectedId === node.id) {{
          const cls = [
            "label",
            isLeaf ? "" : "internal",
            isMatch || !query || node.depth <= 2 || state.selectedId === node.id ? "" : "dim",
          ].filter(Boolean).join(" ");
          const rectClass = selected ? "label-box selected" : "label-box";
          const dx = isLeaf ? 8 : 10;
          parts.push(
            `<g data-label-id="${{node.id}}">
              <rect class="${{rectClass}}" x="${{(node._x + dx - 4).toFixed(2)}}" y="${{(node._y - 10).toFixed(2)}}" width="24" height="16" rx="4" ry="4" />
              <text class="${{cls}}" x="${{(node._x + dx).toFixed(2)}}" y="${{(node._y + 4).toFixed(2)}}">${{escapeHtml(node.name)}}</text>
            </g>`
          );
        }}
      }}

      parts.push("</svg>");
      chartWrap.innerHTML = parts.join("");
      const nextChart = chartWrap.querySelector("svg");
      fitLabelBoxes(nextChart);
      bindInteractions(nextChart);
      updateDetails(state.selectedId ? nodeIndex.get(state.selectedId) : null);
      const visibleNodeCount = layout.nodes.filter((node) => node.depth > 0).length;
      const selectedNode = state.selectedId ? nodeIndex.get(state.selectedId) : null;
      const selectionText = selectedNode ? ` Selected: ${{selectedNode.name}}.` : "";
      statusLine.textContent = `Visible nodes: ${{formatInteger(visibleNodeCount)}}. Sort: ${{sortModeLabel(state.sortMode)}}. Min VTM predictions: ${{formatInteger(state.minCount)}}.${{selectionText}}`;
      countCaption.textContent = state.minCount > 0
        ? `Showing branches with at least ${{formatInteger(state.minCount)}} VTM predictions`
        : "Showing all branches";
    }}

    function bindInteractions(svg) {{
      svg.querySelectorAll("[data-node-id]").forEach((element) => {{
        const nodeId = element.getAttribute("data-node-id");
        const node = nodeIndex.get(nodeId);
        if (!node) {{
          return;
        }}
        element.addEventListener("mouseenter", (event) => showTooltip(event, node));
        element.addEventListener("mousemove", (event) => showTooltip(event, node));
        element.addEventListener("mouseleave", hideTooltip);
        element.addEventListener("click", () => {{
          state.selectedId = node.id;
          render();
        }});
        element.addEventListener("dblclick", () => {{
          if (!node.children || node.children.length === 0) {{
            return;
          }}
          if (state.collapsed.has(node.id)) {{
            state.collapsed.delete(node.id);
          }} else {{
            state.collapsed.add(node.id);
          }}
          state.selectedId = node.id;
          render();
        }});
      }});

      svg.querySelectorAll("[data-label-id]").forEach((element) => {{
        const nodeId = element.getAttribute("data-label-id");
        const node = nodeIndex.get(nodeId);
        if (!node) {{
          return;
        }}
        element.style.cursor = "pointer";
        element.addEventListener("mouseenter", (event) => showTooltip(event, node));
        element.addEventListener("mousemove", (event) => showTooltip(event, node));
        element.addEventListener("mouseleave", hideTooltip);
        element.addEventListener("click", () => {{
          state.selectedId = node.id;
          render();
        }});
        element.addEventListener("dblclick", () => {{
          if (!node.children || node.children.length === 0) {{
            return;
          }}
          if (state.collapsed.has(node.id)) {{
            state.collapsed.delete(node.id);
          }} else {{
            state.collapsed.add(node.id);
          }}
          state.selectedId = node.id;
          render();
        }});
      }});
    }}

    sortSelect.addEventListener("change", () => {{
      state.sortMode = sortSelect.value;
      render();
    }});

    searchInput.addEventListener("input", () => {{
      state.query = searchInput.value;
      render();
    }});

    countSlider.addEventListener("input", () => {{
      const value = Number.parseInt(countSlider.value, 10) || 0;
      state.minCount = value;
      minCountInput.value = String(value);
      render();
    }});

    minCountInput.addEventListener("input", () => {{
      let value = Number.parseInt(minCountInput.value, 10);
      if (!Number.isFinite(value) || value < 0) {{
        value = 0;
      }}
      value = Math.min(value, maxCount);
      state.minCount = value;
      countSlider.value = String(value);
      render();
    }});

    resetViewButton.addEventListener("click", () => {{
      state.sortMode = "predictions";
      state.minCount = 0;
      state.collapsed.clear();
      state.selectedId = null;
      state.query = "";
      sortSelect.value = "predictions";
      searchInput.value = "";
      countSlider.value = "0";
      minCountInput.value = "0";
      render();
    }});

    buildLegend();
    renderSummaryMetrics();
    updateDetails(null);
    render();
  </script>
</body>
</html>
"""


@app.command("taxonomy-viz")
def taxonomy_viz_command(
    taxonomy: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
        help="Path to the taxonomy CSV/Parquet/Feather containing name and parent columns.",
    ),
    results: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
        help="Path to the results CSV/Parquet/Feather containing predicted taxonomy labels.",
    ),
    output_dir: Path = typer.Option(
        Path("data/taxonomy_viz"),
        "--output-dir",
        "-o",
        help="Directory where the static visualization files will be written.",
        path_type=Path,
    ),
    prediction_column: str = typer.Option(
        "generated_keywords",
        "--prediction-column",
        help="Column in the results table containing the predicted taxonomy label.",
    ),
    title: str = typer.Option(
        "Variable Taxonomy Cladogram",
        "--title",
        help="Title shown in the generated visualization.",
    ),
) -> None:
    """Build a static, genetics-inspired taxonomy visualization from local files."""

    taxonomy_path = ensure_file_exists(taxonomy.resolve(), "taxonomy data file")
    results_path = ensure_file_exists(results.resolve(), "results data file")
    output_path = output_dir.resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading taxonomy from %s", taxonomy_path)
    taxonomy_df = load_table(taxonomy_path, low_memory=False)
    logger.info(
        "Loaded taxonomy table with %d rows and %d columns",
        len(taxonomy_df),
        len(taxonomy_df.columns),
    )

    logger.info("Loading results from %s", results_path)
    results_df = load_table(results_path, low_memory=False)
    logger.info(
        "Loaded results table with %d rows and %d columns",
        len(results_df),
        len(results_df.columns),
    )

    payload = _build_payload(
        taxonomy_df,
        results_df,
        prediction_column=prediction_column,
        title=title,
    )
    html = _build_html(payload)

    data_path = output_path / "taxonomy-data.json"
    html_path = output_path / "index.html"
    data_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")

    logger.info("Wrote visualization payload to %s", data_path)
    logger.info("Wrote visualization page to %s", html_path)
    logger.info(
        "Open %s directly in a browser, or serve the directory with: uv run python -m http.server --directory %s",
        html_path,
        output_path,
    )
