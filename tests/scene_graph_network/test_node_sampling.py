import torch

from image_processing_tools.scene_graph_network.node_sampling import sample_balanced_nodes


def _types(n_bg, n_epi, n_hyph):
    return torch.tensor([0] * n_bg + [1] * n_epi + [2] * n_hyph, dtype=torch.long)


def test_equal_counts_per_class_at_ratio_one():
    t = _types(10, 40, 200)
    idx = sample_balanced_nodes(t, ratio=1.0)
    counts = torch.bincount(t[idx], minlength=3)
    assert counts.tolist() == [10, 10, 10]


def test_absent_class_is_skipped_not_sampled():
    """Images 0/1 have no epithelial nodes; the loss there is a 2-class problem."""
    t = _types(4, 0, 153)
    idx = sample_balanced_nodes(t, ratio=1.0)
    counts = torch.bincount(t[idx], minlength=3)
    assert counts.tolist() == [4, 0, 4]
    assert len(idx) == 8


def test_ratio_scales_the_commoner_classes():
    t = _types(10, 40, 200)
    idx = sample_balanced_nodes(t, ratio=2.0)
    counts = torch.bincount(t[idx], minlength=3)
    # The rarest class cannot exceed what it has.
    assert counts.tolist() == [10, 20, 20]


def test_never_samples_more_than_a_class_has():
    t = _types(10, 12, 200)
    idx = sample_balanced_nodes(t, ratio=5.0)
    counts = torch.bincount(t[idx], minlength=3)
    assert counts.tolist() == [10, 12, 50]


def test_single_class_returns_that_class():
    t = _types(0, 0, 20)
    idx = sample_balanced_nodes(t, ratio=1.0)
    assert torch.bincount(t[idx], minlength=3).tolist() == [0, 0, 20]


def test_indices_are_valid_and_unique():
    t = _types(10, 40, 200)
    idx = sample_balanced_nodes(t, ratio=1.0)
    assert idx.dtype == torch.long
    assert len(torch.unique(idx)) == len(idx)
    assert int(idx.max()) < len(t)


def test_generator_makes_it_reproducible():
    t = _types(10, 40, 200)
    a = sample_balanced_nodes(t, generator=torch.Generator().manual_seed(0))
    b = sample_balanced_nodes(t, generator=torch.Generator().manual_seed(0))
    assert torch.equal(a, b)


def test_resampling_rotates_the_majority_class():
    """Nothing is discarded permanently: a fresh draw sees different majority nodes."""
    t = _types(10, 40, 200)
    a = sample_balanced_nodes(t, generator=torch.Generator().manual_seed(0))
    b = sample_balanced_nodes(t, generator=torch.Generator().manual_seed(1))
    assert not torch.equal(torch.sort(a).values, torch.sort(b).values)


def test_empty_input_returns_empty():
    idx = sample_balanced_nodes(torch.zeros(0, dtype=torch.long))
    assert len(idx) == 0
