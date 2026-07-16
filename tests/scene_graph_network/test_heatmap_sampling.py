import numpy as np

from image_processing_tools.scene_graph_network.gnn_interpret import sample_heatmap_edges


def _labels(n_pos, n_neg):
    return np.concatenate([np.ones(n_pos), np.zeros(n_neg)])


def test_samples_n_per_class_from_each_label():
    idx = sample_heatmap_edges(_labels(200, 800), n_per_class=15, seed=0)
    labels = _labels(200, 800)
    assert len(idx) == 30
    assert (labels[idx] == 1).sum() == 15
    assert (labels[idx] == 0).sum() == 15


def test_short_pool_takes_every_edge_it_has():
    """Fewer than n_per_class in a class -> take all of them, not an error."""
    labels = _labels(3, 500)
    idx = sample_heatmap_edges(labels, n_per_class=15, seed=0)
    assert (labels[idx] == 1).sum() == 3      # all 3 positives
    assert (labels[idx] == 0).sum() == 15
    assert len(idx) == 18


def test_pool_smaller_than_requested_in_both_classes():
    labels = _labels(2, 4)
    idx = sample_heatmap_edges(labels, n_per_class=15, seed=0)
    assert sorted(idx) == list(range(6))      # everything


def test_same_seed_gives_the_same_sample():
    labels = _labels(200, 800)
    a = sample_heatmap_edges(labels, n_per_class=15, seed=7)
    b = sample_heatmap_edges(labels, n_per_class=15, seed=7)
    assert np.array_equal(a, b)


def test_different_seed_gives_a_different_sample():
    labels = _labels(200, 800)
    a = sample_heatmap_edges(labels, n_per_class=15, seed=0)
    b = sample_heatmap_edges(labels, n_per_class=15, seed=1)
    assert not np.array_equal(a, b)


def test_indices_are_sorted_and_unique():
    idx = sample_heatmap_edges(_labels(200, 800), n_per_class=15, seed=0)
    assert np.array_equal(idx, np.sort(idx))
    assert len(np.unique(idx)) == len(idx)


def test_empty_positive_class_is_handled():
    labels = _labels(0, 50)
    idx = sample_heatmap_edges(labels, n_per_class=15, seed=0)
    assert len(idx) == 15
    assert (labels[idx] == 0).all()
