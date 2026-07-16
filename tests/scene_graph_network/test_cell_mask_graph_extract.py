import numpy as np
import pytest
from image_processing_tools.scene_graph_network.cell_mask_graph import (
    extract_cell_graph, NODE_FEATURE_COLUMNS, EDGE_FEATURE_COLUMNS,
)


def _labels_and_dic():
    ais = np.zeros((20, 60), dtype=np.int32)
    ais[8:12, 2:14] = 1
    ais[8:12, 16:28] = 2
    ais[8:12, 30:42] = 3
    rng = np.random.default_rng(0)
    dic = rng.random((20, 60)).astype(np.float32)
    return ais, dic


def test_contract_shapes_and_columns():
    ais, dic = _labels_and_dic()
    node_df, centroids, node_bboxes, edge_df, edge_index = extract_cell_graph(ais, dic, k=2)
    assert list(node_df.columns) == ["node_id"] + NODE_FEATURE_COLUMNS
    assert list(edge_df.columns) == ["source_node", "target_node"] + EDGE_FEATURE_COLUMNS
    assert len(node_df) == 3
    assert node_bboxes.shape == (3, 4)
    assert len(centroids) == 3
    assert len(edge_index) == 2 and len(edge_index[0]) == len(edge_df)


def test_edge_feature_column_order_and_ranges():
    ais, dic = _labels_and_dic()
    _, _, _, edge_df, _ = extract_cell_graph(ais, dic, k=2)
    # col 0 = raw gap_intensity; col 1 = normalized distance
    assert EDGE_FEATURE_COLUMNS[0] == "gap_intensity"
    assert EDGE_FEATURE_COLUMNS[1] == "boundary_dist_norm"
    for col in ["node1_angle_diff", "node2_angle_diff", "min_diff_angle",
                "relative_angle", "contact_frac", "area_ratio", "axis_collinearity"]:
        v = edge_df[col].to_numpy()
        assert np.all(v >= -1e-6) and np.all(v <= 1 + 1e-6)
    assert np.all(edge_df["intensity_continuity"].to_numpy() >= -1 - 1e-6)
    assert np.all(edge_df["boundary_dist_norm"].to_numpy() >= 0)


def test_channel_stack_is_rejected():
    """A stack must not be accepted: profile_line would return (L, C) and
    intensity_continuity's corrcoef would silently pin itself to +/-1 instead of
    comparing the two fragments' profiles. Channels are reduced upstream by
    ImageContainer([[*channel_paths]], config).merge()."""
    ais, dic = _labels_and_dic()
    stack = np.stack([dic, dic], axis=-1)
    with pytest.raises(ValueError, match="single 2D channel"):
        extract_cell_graph(ais, stack, k=2)


def test_empty_graph_single_fragment():
    ais = np.zeros((10, 10), dtype=np.int32)
    ais[3:7, 3:7] = 1
    dic = np.ones((10, 10), dtype=np.float32)
    node_df, centroids, node_bboxes, edge_df, edge_index = extract_cell_graph(ais, dic)
    assert len(node_df) == 1
    assert len(edge_df) == 0
    assert edge_index == [[], []]