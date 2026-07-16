import numpy as np
from image_processing_tools.scene_graph_network.cell_mask_graph import (
    _extract_fragments, _node_feature_row, _node_bbox_xyxy, NODE_FEATURE_COLUMNS,
)


def _one_fragment():
    ais = np.zeros((30, 30), dtype=np.int32)
    ais[10:20, 8:22] = 1                # a 10x14 rectangle
    dic = np.zeros((30, 30), dtype=np.float32)
    dic[10:20, 8:22] = 5.0              # bright interior
    dic[8:10, 8:22] = 2.0               # brighter-than-zero context ring above
    return ais, dic


def test_node_feature_row_keys_and_ranges():
    ais, dic = _one_fragment()
    frags, _, _, mean_area, mean_major = _extract_fragments(ais, dic)
    row = _node_feature_row(frags[0], dic, mean_area, mean_major, ring_width=3)
    assert set(row.keys()) == set(NODE_FEATURE_COLUMNS)
    assert 0.0 <= row["eccentricity"] <= 1.0
    assert 0.0 <= row["solidity"] <= 1.0
    assert row["interior_intensity"] == 5.0            # mean over interior pixels
    assert row["context_intensity"] > 0.0              # ring picked up nonzero context


def test_node_bbox_is_xyxy_and_clipped():
    ais, dic = _one_fragment()
    frags, _, _, _, _ = _extract_fragments(ais, dic)
    box = _node_bbox_xyxy(frags[0], pad_frac=0.0, shape=ais.shape)
    # bbox rows 10..20, cols 8..22 -> xyxy = [8, 10, 22, 20]
    assert box.tolist() == [8.0, 10.0, 22.0, 20.0]
    padded = _node_bbox_xyxy(frags[0], pad_frac=1.0, shape=ais.shape)
    assert padded[0] >= 0 and padded[1] >= 0                       # clipped to >=0
    assert padded[2] <= ais.shape[1] and padded[3] <= ais.shape[0]  # clipped to shape