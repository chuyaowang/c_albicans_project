import numpy as np
from image_processing_tools.scene_graph_network.cell_mask_graph import (
    _extract_fragments, _knn_edges,
)


def _labels():
    ais = np.zeros((20, 60), dtype=np.int32)
    ais[8:12, 2:14] = 1
    ais[8:12, 16:28] = 2
    ais[8:12, 30:42] = 3
    return ais


def test_extract_fragments_basic():
    ais = _labels()
    dic = np.ones_like(ais, dtype=np.float32)
    frags, trees, centroids, mean_area, mean_major = _extract_fragments(ais, dic)
    assert [f["label"] for f in frags] == [1, 2, 3]
    assert centroids.shape == (3, 2)
    assert set(trees.keys()) == {1, 2, 3}
    assert mean_area > 0 and mean_major > 0


def test_knn_edges_connect_adjacent_fragments():
    ais = _labels()
    dic = np.ones_like(ais, dtype=np.float32)
    frags, trees, centroids, _, _ = _extract_fragments(ais, dic)
    edges = _knn_edges(frags, trees, centroids, k=1)
    pairs = {(i, j) for (i, j, *_rest) in edges}
    # k=1 nearest-by-boundary: 1<->2 and 2<->3 (1 and 3 are not nearest neighbors)
    assert (0, 1) in pairs and (1, 2) in pairs
    assert (0, 2) not in pairs


def test_knn_distance_cap_drops_far_edges():
    ais = _labels()
    dic = np.ones_like(ais, dtype=np.float32)
    frags, trees, centroids, _, mean_major = _extract_fragments(ais, dic)
    # cap smaller than the ~2px inter-fragment gap → no edges survive
    edges = _knn_edges(frags, trees, centroids, k=2, dist_cap=0.5)
    assert edges == []