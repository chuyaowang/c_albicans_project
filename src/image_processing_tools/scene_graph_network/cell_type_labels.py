"""Derive per-node type labels (background / epithelial / hyphal) from GT whole-cell masks.

A fragment's type comes from the GT cell it was ASSIGNED to, never from its own shape: a
fragment of a hypha is often round on its own, and only the whole GT cell carries the shape
that separates the classes. That is what makes this a learning problem rather than a
restatement of the existing node features.

The rule is per image because magnification differs across the dataset -- hyphae in the
long/thin images have a mean width of ~18 px against ~100+ px in the coculture images, so no
global cutoff transfers. Images with only one cell type must use `{"all": <type>}`: a
threshold computed inside them would slice their single population in half.
"""
import numpy as np
from skimage.measure import regionprops

from image_processing_tools.scene_graph_network.cell_merge_labels import assign_fragments_to_gt

# Background is 0 so it remains the "reject" class.
NODE_CLASSES = {"background": 0, "epithelial": 1, "hyphal": 2}
NODE_CLASS_NAMES = {v: k for k, v in NODE_CLASSES.items()}

# name -> (fn(regionprop) -> float, which side is hyphal)
# The direction lives with the metric so a rule only carries a number, and switching metric
# cannot silently invert the two classes.
CELL_TYPE_METRICS = {
    # Thinness. A hypha is thin everywhere; a spread-out epithelial cell is not, however
    # elongated or non-convex its outline. In pixels, so scale-dependent -- usable only
    # because the threshold is per-image.
    "mean_width":   (lambda p: p.area / max(p.major_axis_length, 1e-9), "low"),
    "minor_axis":   (lambda p: p.minor_axis_length, "low"),
    # Elongation / convexity. Dimensionless, but each confuses a spiky epithelial cell for a
    # filament to some degree.
    "extent":       (lambda p: p.extent, "low"),
    "axis_ratio":   (lambda p: p.major_axis_length / max(p.minor_axis_length, 1e-9), "high"),
    "solidity":     (lambda p: p.solidity, "low"),
    "circularity":  (lambda p: 4 * np.pi * p.area / max(p.perimeter ** 2, 1e-9), "low"),
    "eccentricity": (lambda p: p.eccentricity, "high"),
}


def gt_cell_types(gt_labels, rule):
    """Map each GT cell label to "epithelial" or "hyphal".

    Args:
        gt_labels: GT whole-cell instance map (H, W).
        rule: either {"all": "hyphal"} (every cell that type, metric ignored) or
            {"metric": <CELL_TYPE_METRICS key>, "threshold": <float>}.

    Returns:
        dict[int, str]: GT label -> type name.
    """
    props = regionprops(np.asarray(gt_labels, dtype=np.int32))
    if "all" in rule:
        return {int(p.label): rule["all"] for p in props}

    fn, direction = CELL_TYPE_METRICS[rule["metric"]]
    out = {}
    for p in props:
        v = fn(p)
        hyphal = v > rule["threshold"] if direction == "high" else v < rule["threshold"]
        out[int(p.label)] = "hyphal" if hyphal else "epithelial"
    return out


def node_type_labels(ais_labels, gt_labels, rule, min_overlap_frac=0.1):
    """Per-node type targets, in regionprops (label-ascending) order.

    Args:
        ais_labels: AIS instance label map (H, W); the node set.
        gt_labels: GT whole-cell instance map (H, W).
        rule: see `gt_cell_types`.
        min_overlap_frac: below this share of its own area overlapping its best GT cell, a
            fragment is background. Must match the value used for the edge labels -- one
            split feeds both.

    Returns:
        (N,) int64 array of NODE_CLASSES values.
    """
    assign = assign_fragments_to_gt(ais_labels, gt_labels, min_overlap_frac)
    types = gt_cell_types(gt_labels, rule)
    out = np.empty(len(assign), dtype=np.int64)
    for i, gt_id in enumerate(assign):
        out[i] = (NODE_CLASSES["background"] if gt_id == -1
                  else NODE_CLASSES[types[int(gt_id)]])
    return out
