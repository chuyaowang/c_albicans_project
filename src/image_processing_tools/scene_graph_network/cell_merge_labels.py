"""Generate GNN merge labels by matching AIS fragments to ground-truth whole cells.

Each AIS fragment is a node (regionprops order, labels ascending). A fragment is
assigned to the GT cell it overlaps most, unless that overlap is below
`min_overlap_frac` of the fragment's own area (→ background, label -1). Within each
GT cell, true edges are the edges of the minimum spanning tree over its fragments
(weighted by min boundary distance): a chain/tree, never a clique.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries


def _boundary_points(ais_labels):
    """Map each fragment label -> (M,2) array of its inner-boundary pixel coords."""
    bnd = find_boundaries(ais_labels, mode="inner")
    ys, xs = np.nonzero(bnd)
    blabels = ais_labels[ys, xs]
    out = {}
    for lab in np.unique(blabels):
        m = blabels == lab
        out[int(lab)] = np.column_stack([ys[m], xs[m]])
    return out


def assign_fragments_to_gt(ais_labels, gt_labels, min_overlap_frac=0.5):
    props = regionprops(ais_labels)
    gt_of_node = np.full(len(props), -1, dtype=np.int64)
    for i, p in enumerate(props):
        rows, cols = p.coords[:, 0], p.coords[:, 1]
        gvals = gt_labels[rows, cols]
        gvals = gvals[gvals != 0]
        if gvals.size == 0:
            continue
        vals, counts = np.unique(gvals, return_counts=True)
        best = int(np.argmax(counts))
        if counts[best] >= min_overlap_frac * p.area:
            gt_of_node[i] = int(vals[best])
    return gt_of_node


def cell_merge_labels(ais_labels, gt_labels, min_overlap_frac=0.5):
    props = regionprops(ais_labels)
    labels = [p.label for p in props]                      # node index -> ais label
    gt_of_node = assign_fragments_to_gt(ais_labels, gt_labels, min_overlap_frac)
    boundary = _boundary_points(ais_labels)

    edges = []
    for gt_id in np.unique(gt_of_node[gt_of_node != -1]):
        members = [i for i in range(len(labels)) if gt_of_node[i] == gt_id]
        if len(members) < 2:
            continue                                       # singleton cell -> no edges
        # dense pairwise min-boundary-distance matrix among this cell's fragments
        n = len(members)
        dist = np.zeros((n, n), dtype=np.float64)
        trees = [cKDTree(boundary[labels[m]]) for m in members]
        for a in range(n):
            for b in range(a + 1, n):
                d, _ = trees[b].query(boundary[labels[members[a]]], k=1)
                dist[a, b] = dist[b, a] = float(d.min())
        mst = minimum_spanning_tree(csr_matrix(dist)).tocoo()
        for a, b in zip(mst.row, mst.col):
            u, v = members[a], members[b]
            edges.append((min(u, v), max(u, v)))
    return sorted(set(edges))