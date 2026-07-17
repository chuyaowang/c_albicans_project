"""`node1_angle_diff` must describe the SOURCE of its directed edge, `node2` the target.

The nuclei pipeline's `create_pyg_data` runs `T.ToUndirected()`, which copies each edge_attr
row verbatim onto the reverse edge. `extract_graph` computes the row once for (i, j) with
i < j, so on the reverse edge the copied columns describe the target and the source
respectively -- backwards, and silently: every value is in range and nothing raises.

Mirrors tests/scene_graph_network/test_edge_attr_direction.py; the two pipelines carry the
same two columns and had the same bug. Named differently because the test dirs are implicit
namespace packages with no __init__.py, so pytest cannot collect two modules of one name.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from image_processing_tools.dapi_tracing.gnn_data import create_pyg_data

# The nuclei edge feature table, in extract_graph's column order.
COLS = ["average_intensity", "length", "node1_angle_diff", "node2_angle_diff",
        "min_diff_angle", "relative_angle"]
NODE_COLS = ["circularity", "eccentricity", "area", "average_intensity",
             "major_axis_length", "minor_axis_length"]
N1, N2 = COLS.index("node1_angle_diff"), COLS.index("node2_angle_diff")


def _graph(pairs, node_angles):
    """One graph whose node1/node2 columns are traceable: for pair (i, j) they hold the
    marker angle of i and of j, so we can assert which endpoint each column follows."""
    n = max(max(p) for p in pairs) + 1
    nodes = pd.DataFrame({"node_id": range(n), **{c: np.zeros(n) for c in NODE_COLS}})
    rows = []
    for i, j in pairs:
        r = {c: 0.0 for c in COLS}
        r["node1_angle_diff"] = node_angles[i]     # by construction: the LOWER index, i
        r["node2_angle_diff"] = node_angles[j]     # the HIGHER index, j
        r["source_node"], r["target_node"] = i, j
        rows.append(r)
    edges = pd.DataFrame(rows, columns=["source_node", "target_node"] + COLS)
    return create_pyg_data(
        edge_indices=[[[i for i, _ in pairs], [j for _, j in pairs]]],
        nuclei_features_list=[nodes], path_features_list=[edges], edge_labels_list=[[]],
    )[0]


def test_node1_follows_the_source_on_every_directed_edge():
    angles = {0: 0.11, 1: 0.22, 2: 0.33}
    d = _graph([(0, 1), (1, 2), (0, 2)], angles)

    assert d.edge_index.shape[1] == 6, "ToUndirected should have produced both directions"
    for k in range(d.edge_index.shape[1]):
        u, v = int(d.edge_index[0, k]), int(d.edge_index[1, k])
        assert d.edge_attr[k, N1].item() == pytest.approx(angles[u]), (
            f"edge {u}->{v}: node1_angle_diff should be the SOURCE's angle {angles[u]}, "
            f"got {d.edge_attr[k, N1].item()}"
        )
        assert d.edge_attr[k, N2].item() == pytest.approx(angles[v]), (
            f"edge {u}->{v}: node2_angle_diff should be the TARGET's angle {angles[v]}"
        )


def test_forward_and_reverse_have_the_two_columns_swapped():
    d = _graph([(0, 1)], {0: 0.11, 1: 0.22})
    lookup = {(int(u), int(v)): k for k, (u, v) in enumerate(d.edge_index.t().tolist())}
    f, r = lookup[(0, 1)], lookup[(1, 0)]
    assert d.edge_attr[f, N1] == d.edge_attr[r, N2]
    assert d.edge_attr[f, N2] == d.edge_attr[r, N1]
    assert d.edge_attr[f, N1] != d.edge_attr[f, N2], "markers must differ or this proves nothing"


def test_the_symmetric_columns_are_untouched():
    """average_intensity, length, min_diff_angle and relative_angle are order-independent by
    construction and must be copied unchanged."""
    n = 2
    nodes = pd.DataFrame({"node_id": range(n), **{c: np.zeros(n) for c in NODE_COLS}})
    r = {c: 0.5 + COLS.index(c) / 100.0 for c in COLS}     # a distinct marker per column
    r["source_node"], r["target_node"] = 0, 1
    edges = pd.DataFrame([r], columns=["source_node", "target_node"] + COLS)
    d = create_pyg_data(edge_indices=[[[0], [1]]], nuclei_features_list=[nodes],
                        path_features_list=[edges], edge_labels_list=[[]])[0]

    lookup = {(int(u), int(v)): k for k, (u, v) in enumerate(d.edge_index.t().tolist())}
    f, rv = lookup[(0, 1)], lookup[(1, 0)]
    for c in COLS:
        if c in ("node1_angle_diff", "node2_angle_diff"):
            continue
        i = COLS.index(c)
        assert d.edge_attr[f, i] == d.edge_attr[rv, i], f"{c} should be direction-invariant"
