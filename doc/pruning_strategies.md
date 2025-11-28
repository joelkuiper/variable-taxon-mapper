# Pruning strategies

This document summarizes the pruning modes supported by the Variable Taxon Mapper and expresses each as short, scientific pseudocode. Each section briefly states the intent of the mode and then lists the algorithm sketch that the implementation follows. References point to canonical background material for the underlying graph concepts.

## Dominant forest

**Intent:** Grow a small covering forest around the anchors so that influential nearby nodes (by PageRank and graph distance) are included while respecting the node budget. The approach is inspired by dominating set heuristics on directed acyclic graphs ([Wikipedia: Dominating set](https://en.wikipedia.org/wiki/Dominating_set)).

**Pseudocode:**

```text
input: directed taxonomy G, anchor nodes A, node_budget B, depth limit D
candidates, pagerank_scores, distances = prepare_pagerank_scores(G, A, D)
ordered = anchors followed by other candidates sorted by (distance, -pagerank)
allowed = ∅
for node in ordered:
    if node ∉ G or (B>0 and |allowed| ≥ B): continue
    path = ancestors_to_root(G, node)
    add nodes from path to allowed until budget is hit
    if B>0 and |allowed| ≥ B: break
return allowed
```

## Anchor hull

**Intent:** Collect the minimal ancestor/descendant hull surrounding the anchors so the LLM sees their local neighborhood. This is similar to building a convex hull in graph space ([Wikipedia: Convex hull of a graph](https://en.wikipedia.org/wiki/Convex_hull#Graphs_and_networks)).

**Pseudocode:**

```text
input: G, anchors A, descendant depth D, community params, budget B
neighborhood = anchor_neighborhood(G, A, depth=D, communities)
for each anchor a in A: neighborhood ∪= ancestors_to_root(G, a)
if B>0: neighborhood = enforce_node_budget_with_ancestors(G, neighborhood, B)
return neighborhood
```

## Similarity threshold

**Intent:** Keep every taxonomy node whose embedding similarity to the variable exceeds a configured cutoff, plus necessary ancestors to maintain a tree. This is a score-based filter akin to top-p pruning in vector search.

**Pseudocode:**

```text
input: G, anchors A, similarity map S(name→score), threshold τ, budget B
allowed = {name ∈ G | S[name] ≥ τ} ∪ anchors
expanded = ⋃_{node ∈ allowed} ancestors_to_root(G, node)
if B>0: expanded = enforce_node_budget_with_ancestors(G, expanded, B)
return expanded
```

## Radius limited

**Intent:** Restrict the search to nodes within an undirected hop radius of any anchor, mirroring a fixed-radius ball around anchor seeds ([Wikipedia: Graph distance](https://en.wikipedia.org/wiki/Distance_(graph_theory))).

**Pseudocode:**

```text
input: G, anchors A, radius r, budget B
convert G to undirected graph U
neighbors = {v | distance_U(v, A) ≤ r} ∪ A
expanded = ⋃_{v ∈ neighbors} ancestors_to_root(G, v)
if B>0: expanded = enforce_node_budget_with_ancestors(G, expanded, B)
return expanded
```

## Community PageRank

**Intent:** Pick high-PageRank representatives per community around the anchors so dense regions are summarized without blowing the budget. Uses Louvain-like clique communities and PageRank centrality ([Wikipedia: PageRank](https://en.wikipedia.org/wiki/PageRank), [Wikipedia: Clique (graph theory)](https://en.wikipedia.org/wiki/Clique_(graph_theory))).

**Pseudocode:**

```text
input: G, anchors A, depth D, community params, budget B
candidates, pagerank_scores, distances = prepare_pagerank_scores(G, A, D, communities)
if no candidates: return ancestor closure of anchors (budgeted)
selected = anchors ∩ G
for each community touching an anchor:
    pick member with best (−pagerank, distance)
    selected ∪= {member}
for node in candidates sorted by (−pagerank, distance):
    if B>0 and |selected| ≥ B: break
    selected ∪= {node}
expanded = ⋃_{v ∈ selected} ancestors_to_root(G, v)
if B>0: expanded = enforce_node_budget_with_ancestors(G, expanded, B)
return expanded
```

## Steiner similarity

**Intent:** Connect anchors to the highest-similarity nodes via a Steiner tree approximation so the subgraph captures shared ancestors efficiently ([Wikipedia: Steiner tree problem](https://en.wikipedia.org/wiki/Steiner_tree_problem)).

**Pseudocode:**

```text
input: G, anchors A, similarity map S, budget B
terminals = anchors ∪ top_s similarity nodes (where s = B - |anchors|)
U = undirected version of G
steiner_nodes = steiner_tree(U, terminals).nodes ∪ terminals
expanded = ⋃_{v ∈ steiner_nodes} ancestors_to_root(G, v)
if B>0: expanded = enforce_node_budget_with_ancestors(G, expanded, B)
return expanded
```
