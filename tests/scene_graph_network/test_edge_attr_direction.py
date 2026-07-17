"""`node1_angle_diff` must describe the SOURCE of its directed edge, `node2` the target.

`create_pyg_data` runs `T.ToUndirected()`, which copies each edge_attr row verbatim onto the
reverse edge. The producers compute the row once for (i, j) with i < j, so on the reverse
edge the copied columns describe the target and the source respectively -- backwards, and
silently: every value is in range and nothing raises.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from image_processing_tools.scene_graph_network.gnn_data import create_pyg_data

COLS = ["gap_intensity", "boundary_dist_norm", "node1_angle_diff", "node2_angle_diff",
        "min_diff_angle", "relative_angle", "contact_frac", "area_ratio",
        "axis_collinearity", "intensity_continuity"]
N1, N2 = COLS.index("node1_angle_diff"), COLS.index("node2_angle_diff")


def _graph(pairs, node_angles):
    """One graph whose node1/node2 columns are traceable: for pair (i, j) they are set to
    the marker angle of i and of j, so we can assert which endpoint each column follows."""
    n = max(max(p) for p in pairs) + 1
    nodes = pd.DataFrame({"node_id": range(n),
                          **{c: np.zeros(n) for c in
                             ["circularity", "eccentricity", "solidity", "area_norm",
                              "major_axis_norm", "minor_axis_norm",
                              "interior_intensity", "context_intensity"]}})
    rows = []
    for i, j in pairs:
        r = {c: 0.0 for c in COLS}
        r["node1_angle_diff"] = node_angles[i]     # by construction: the LOWER index, i
        r["node2_angle_diff"] = node_angles[j]     # the HIGHER index, j
        r["source_node"], r["target_node"] = i, j
        rows.append(r)
    edges = pd.DataFrame(rows, columns=["source_node", "target_node"] + COLS)
    src = [i for i, _ in pairs]
    tgt = [j for _, j in pairs]
    return create_pyg_data(edge_indices=[[src, tgt]], nuclei_features_list=[nodes],
                           path_features_list=[edges], edge_labels_list=[[]])[0]


def test_node1_follows_the_source_on_every_directed_edge():
    """The invariant. node_angles are distinct markers, so each column is traceable."""
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
    angles = {0: 0.11, 1: 0.22}
    d = _graph([(0, 1)], angles)

    lookup = {(int(u), int(v)): k for k, (u, v) in enumerate(d.edge_index.t().tolist())}
    f, r = lookup[(0, 1)], lookup[(1, 0)]
    assert d.edge_attr[f, N1] == d.edge_attr[r, N2]
    assert d.edge_attr[f, N2] == d.edge_attr[r, N1]
    assert d.edge_attr[f, N1] != d.edge_attr[f, N2], "markers must differ or this proves nothing"


def test_the_symmetric_columns_are_untouched():
    """Only the two angle columns are direction-dependent. The other eight are symmetric by
    construction (min/max, |cos|, correlations) and must be copied unchanged."""
    n = 2
    nodes = pd.DataFrame({"node_id": range(n),
                          **{c: np.zeros(n) for c in
                             ["circularity", "eccentricity", "solidity", "area_norm",
                              "major_axis_norm", "minor_axis_norm",
                              "interior_intensity", "context_intensity"]}})
    r = {c: 0.0 for c in COLS}
    for c in COLS:
        r[c] = 0.5 + COLS.index(c) / 100.0          # a distinct marker per column
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


def test_no_angle_columns_is_a_no_op():
    """create_pyg_data is generic; a feature table without the angle columns must pass
    through untouched rather than raise or mangle a column by position."""
    n = 2
    nodes = pd.DataFrame({"node_id": range(n), "a": [0.0, 0.0], "b": [0.0, 0.0]})
    edges = pd.DataFrame([{"source_node": 0, "target_node": 1, "x": 1.0, "y": 2.0}])
    d = create_pyg_data(edge_indices=[[[0], [1]]], nuclei_features_list=[nodes],
                        path_features_list=[edges], edge_labels_list=[[]])[0]
    assert d.edge_attr.shape == (2, 2)
    assert torch.equal(d.edge_attr[0], d.edge_attr[1])
