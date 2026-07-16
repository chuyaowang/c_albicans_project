"""Merge AIS fragments into whole cells from the GNN's predicted edges.

The inverse of `cell_merge_labels`: that module derives training labels from GT
whole-cell masks, this one reconstructs whole cells from what the model predicted.

Predicted-true edges are assembled into a graph, and each connected subnetwork is
one biological cell -- every fragment in a subnetwork is relabelled to that
network's integer id. Grouping is decided purely by the edge graph, never by pixel
adjacency: fragments of one hypha are frequently *not* physically touching, which
is the whole reason the model exists.

Because the subnetwork is a real graph, each cell's fragment **chain order** falls
out of it too. Hyphal chains are unbranched and acyclic, so a well-formed
prediction is a path; `topology` reports where that does not hold.
"""
import numpy as np
import networkx as nx

# Subnetwork shapes. 'path'/'singleton' are biologically well-formed; the other
# two mean the prediction violated the degree<=2 / acyclic constraints.
TOPOLOGY_SINGLETON = "singleton"
TOPOLOGY_PATH = "path"
TOPOLOGY_BRANCHED = "branched"
TOPOLOGY_CYCLIC = "cyclic"


def build_prediction_graph(fragment_labels, edge_index, pred_binary,
                           probs=None, true_labels=None, edge_classes=None,
                           attentions=None, centroids=None):
    """Assemble the predicted-true edges into a graph, one node per fragment.

    Nodes are fragment indices (regionprops order). Edges carry the predicted-true
    pairs only. Optional per-edge arrays are attached as edge attributes and the
    optional centroids as node attributes, so the saved graph is self-contained
    for downstream analysis.

    Args:
        fragment_labels: (N,) AIS label of each node, in node-index order.
        edge_index:      (2, E) directed candidate edges.
        pred_binary:     (E,) 0/1 predictions, already symmetrized.
        probs:           (E,) predicted probabilities, optional.
        true_labels:     (E,) ground-truth edge labels, optional.
        edge_classes:    (E,) 'TP'/'TN'/'FP'/'FN' strings, optional.
        attentions:      (a1, a2) tuple of (E,) arrays, optional.
        centroids:       (N, 2) float (row, col), optional.

    Returns:
        nx.Graph with every fragment as a node, even isolated ones.
    """
    fragment_labels = np.asarray(fragment_labels)
    edge_index = np.asarray(edge_index)
    pred_binary = np.asarray(pred_binary)

    graph = nx.Graph()
    for node in range(len(fragment_labels)):
        attrs = {"ais_label": int(fragment_labels[node])}
        if centroids is not None:
            attrs["y"] = float(centroids[node][0])
            attrs["x"] = float(centroids[node][1])
        graph.add_node(node, **attrs)

    # Predictions are symmetrized upstream, so (u,v) and (v,u) agree; adding both
    # to an undirected Graph collapses them to one edge.
    for e in np.nonzero(pred_binary >= 0.5)[0]:
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        if u == v:
            continue
        attrs = {}
        if probs is not None:
            attrs["prob"] = float(probs[e])
        if true_labels is not None:
            attrs["true_label"] = float(true_labels[e])
        if edge_classes is not None:
            attrs["edge_class"] = str(edge_classes[e])
        if attentions is not None:
            attrs["a1"] = float(attentions[0][e])
            attrs["a2"] = float(attentions[1][e])
        graph.add_edge(u, v, **attrs)
    return graph


def _chain_order(sub):
    """Read the fragment order along one subnetwork.

    Returns (ordered node list, topology string). A hyphal chain is unbranched and
    acyclic, so the well-formed cases are a single fragment or a path traversed
    endpoint to endpoint. A branched tree has no unique order, so the diameter
    path is returned as the best available chain and flagged. A subnetwork with a
    cycle has no endpoints to start from and yields no order.
    """
    n = sub.number_of_nodes()
    if n == 1:
        return list(sub.nodes), TOPOLOGY_SINGLETON

    degrees = dict(sub.degree())
    is_tree = sub.number_of_edges() == n - 1
    if not is_tree:
        return [], TOPOLOGY_CYCLIC

    if max(degrees.values()) <= 2:
        endpoints = [v for v, d in degrees.items() if d == 1]
        return nx.shortest_path(sub, endpoints[0], endpoints[1]), TOPOLOGY_PATH

    # Branched tree: fall back to the diameter path (two BFS sweeps from an
    # arbitrary node, then from the farthest node found).
    start = next(iter(sub.nodes))
    far_a = max(nx.single_source_shortest_path_length(sub, start).items(),
                key=lambda kv: kv[1])[0]
    far_b = max(nx.single_source_shortest_path_length(sub, far_a).items(),
                key=lambda kv: kv[1])[0]
    return nx.shortest_path(sub, far_a, far_b), TOPOLOGY_BRANCHED


def merge_fragments(ais_labels, fragment_labels, edge_index, pred_binary,
                    probs=None, true_labels=None, edge_classes=None,
                    attentions=None, centroids=None):
    """Relabel AIS fragments into whole cells using the predicted-true edges.

    Args:
        ais_labels:      (H, W) AIS instance label map.
        fragment_labels: (N,) AIS label of each node, in node-index order.
        edge_index:      (2, E) directed candidate edges.
        pred_binary:     (E,) 0/1 predictions, already symmetrized.
        probs, true_labels, edge_classes, attentions, centroids: optional
            per-edge / per-node arrays attached to the returned graph.

    Returns:
        merged_labels: (H, W) int32 map; fragments of one cell share an id,
                       background stays 0.
        cells:         list of dicts, one per merged cell, with keys
                       `label`, `fragments` (AIS labels in chain order),
                       `nodes` (node indices in chain order) and `topology`.
        graph:         the nx.Graph the merge was read from, with a `cell`
                       attribute added to every node.
    """
    graph = build_prediction_graph(
        fragment_labels, edge_index, pred_binary,
        probs=probs, true_labels=true_labels, edge_classes=edge_classes,
        attentions=attentions, centroids=centroids,
    )
    fragment_labels = np.asarray(fragment_labels)

    # Node index -> cell id, via each connected subnetwork. Components are sorted
    # by their smallest node so cell ids are stable across runs.
    components = sorted((sorted(c) for c in nx.connected_components(graph)),
                        key=lambda c: c[0])

    cells = []
    cell_of_node = {}
    for cell_id, component in enumerate(components, start=1):
        sub = graph.subgraph(component)
        nodes, topology = _chain_order(sub)
        if not nodes:                       # cyclic: no order, but still one cell
            nodes = list(component)
        for node in component:
            cell_of_node[node] = cell_id
            graph.nodes[node]["cell"] = cell_id
        cells.append({
            "label": cell_id,
            "nodes": [int(n) for n in nodes],
            "fragments": [int(fragment_labels[n]) for n in nodes],
            "topology": topology,
        })

    # Relabel through a lookup table rather than per-pixel work.
    lut = np.zeros(int(ais_labels.max()) + 1, dtype=np.int32)
    for node, cell_id in cell_of_node.items():
        lut[int(fragment_labels[node])] = cell_id
    merged_labels = lut[ais_labels]

    return merged_labels, cells, graph


def summarize_cells(cells):
    """One-line-per-topology tally, for logging alongside the merge figure."""
    counts = {}
    for cell in cells:
        counts[cell["topology"]] = counts.get(cell["topology"], 0) + 1
    parts = [f"{counts[t]} {t}" for t in
             (TOPOLOGY_SINGLETON, TOPOLOGY_PATH, TOPOLOGY_BRANCHED, TOPOLOGY_CYCLIC)
             if t in counts]
    return f"{len(cells)} cells: " + ", ".join(parts) if parts else "0 cells"
