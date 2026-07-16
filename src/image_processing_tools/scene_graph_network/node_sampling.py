"""Balanced subsampling of nodes for the node-type loss.

Mirrors the edge loss, which takes every positive plus `neg_sample_ratio * n_pos` negatives
so the loss sees a fixed ratio regardless of graph size. Here the anchor is the rarest class
PRESENT in the batch, and every present class contributes `ratio * n_min` nodes.

The point is that the model never learns a prior on the class distribution. Class weighting
would do the opposite: the per-image class ratios swing hard (background 2.5%-35.6% on the
current dataset), so a weight fitted to the training folds actively mismatches the held-out
one -- the same failure that sank weighted BCE on the edge task.

Resampled every epoch, so no node is permanently discarded; the majority classes rotate.
"""
import torch


def sample_balanced_nodes(node_type, ratio=1.0, generator=None):
    """Indices of a class-balanced subset of nodes.

    Args:
        node_type: (N,) int tensor of class ids.
        ratio: how many nodes to take from each present class, as a multiple of the rarest
            present class's count. 1.0 gives exactly equal counts. Capped per class at what
            that class actually has.
        generator: optional torch.Generator for reproducible draws.

    Returns:
        (M,) int64 tensor of indices into `node_type`. Empty if `node_type` is empty.
    """
    if node_type.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=node_type.device)

    per_class = {}
    for c in torch.unique(node_type).tolist():
        per_class[c] = (node_type == c).nonzero(as_tuple=True)[0]

    n_min = min(len(v) for v in per_class.values())
    n_take = max(int(ratio * n_min), 1)

    picked = []
    for idx in per_class.values():
        take = min(n_take, len(idx))
        perm = torch.randperm(len(idx), generator=generator, device=idx.device)
        picked.append(idx[perm[:take]])
    return torch.cat(picked)
