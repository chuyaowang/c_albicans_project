"""Build a graph from micro-SAM AIS cell-fragment masks for merge prediction.

Nodes are AIS fragments (regionprops order). Candidate edges connect each fragment
to its k nearest neighbours by *minimum boundary-to-boundary distance*. Node and edge
features are merge-oriented (see the design spec). The public `extract_cell_graph`
emits the same (node_df, centroids, node_bboxes, edge_df, edge_index) contract the
existing `create_pyg_data` consumes.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

NODE_FEATURE_COLUMNS = [
    "circularity", "eccentricity", "solidity", "area_norm",
    "major_axis_norm", "minor_axis_norm", "interior_intensity", "context_intensity",
]
EDGE_FEATURE_COLUMNS = [
    "gap_intensity", "boundary_dist_norm", "node1_angle_diff", "node2_angle_diff",
    "min_diff_angle", "relative_angle", "contact_frac", "area_ratio",
    "axis_collinearity", "intensity_continuity",
]


def _extract_fragments(ais_labels, intensity_image):
    props = regionprops(ais_labels)
    bnd = find_boundaries(ais_labels, mode="inner")
    ys, xs = np.nonzero(bnd)
    blabels = ais_labels[ys, xs]

    frags = []
    trees = {}
    for p in props:
        lab = int(p.label)
        m = blabels == lab
        boundary = np.column_stack([ys[m], xs[m]])
        if boundary.shape[0] == 0:                 # degenerate: fall back to coords
            boundary = p.coords
        frags.append({
            "label": lab,
            "centroid": np.asarray(p.centroid, dtype=np.float64),
            "bbox": p.bbox,                        # (min_row, min_col, max_row, max_col)
            "area": float(p.area),
            "perimeter": float(p.perimeter),
            "major": float(p.major_axis_length),
            "minor": float(p.minor_axis_length),
            "orientation": float(p.orientation),
            "eccentricity": float(p.eccentricity),
            "solidity": float(p.solidity),
            "coords": p.coords,
            "boundary": boundary,
        })
        trees[lab] = cKDTree(boundary)

    centroids = np.array([f["centroid"] for f in frags], dtype=np.float64)
    mean_area = float(np.mean([f["area"] for f in frags])) if frags else 1.0
    majors = [f["major"] for f in frags if f["major"] > 0]
    mean_major = float(np.mean(majors)) if majors else 1.0
    return frags, trees, centroids, mean_area, mean_major


def _knn_edges(frags, trees, centroids, k=6, prefilter_mult=3, dist_cap=None):
    n = len(frags)
    if n < 2:
        return []
    ctree = cKDTree(centroids)
    m = min(n - 1, prefilter_mult * k)
    _, cand = ctree.query(centroids, k=min(n, m + 1))     # includes self
    cand = np.atleast_2d(cand)

    edges = {}
    for i in range(n):
        neigh = [int(c) for c in cand[i] if int(c) != i][:m]
        scored = []
        bi = frags[i]["boundary"]
        for j in neigh:
            d, idxj = trees[frags[j]["label"]].query(bi, k=1)
            a = int(np.argmin(d))
            dmin = float(d[a])
            pi = bi[a]
            pj = frags[j]["boundary"][int(idxj[a])]
            scored.append((dmin, j, pi, pj))
        scored.sort(key=lambda t: t[0])
        for dmin, j, pi, pj in scored[:k]:
            if dist_cap is not None and dmin > dist_cap:
                continue
            if i < j:
                key, ppi, ppj = (i, j), pi, pj
            else:
                key, ppi, ppj = (j, i), pj, pi
            if key not in edges:
                edges[key] = (dmin, ppi, ppj)
    return [(a, b, d, pi, pj) for (a, b), (d, pi, pj) in sorted(edges.items())]


from scipy import ndimage as _ndi


def _node_feature_row(frag, dic, mean_area, mean_major, ring_width):
    area, perim = frag["area"], frag["perimeter"]
    circularity = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0.0
    coords = frag["coords"]
    interior = float(dic[coords[:, 0], coords[:, 1]].mean())

    # Context ring: dilate the fragment mask on a local crop, subtract the mask.
    r0, c0, r1, c1 = frag["bbox"]
    pad = ring_width + 1
    R0, C0 = max(r0 - pad, 0), max(c0 - pad, 0)
    R1, C1 = min(r1 + pad, dic.shape[0]), min(c1 + pad, dic.shape[1])
    local_mask = np.zeros((R1 - R0, C1 - C0), dtype=bool)
    local_mask[coords[:, 0] - R0, coords[:, 1] - C0] = True
    dil = _ndi.binary_dilation(local_mask, iterations=ring_width)
    ring = dil & ~local_mask
    ring_vals = dic[R0:R1, C0:C1][ring]
    context = float(ring_vals.mean()) if ring_vals.size > 0 else interior

    return {
        "circularity": circularity,
        "eccentricity": frag["eccentricity"],
        "solidity": frag["solidity"],
        "area_norm": area / mean_area if mean_area > 0 else 0.0,
        "major_axis_norm": frag["major"] / mean_major if mean_major > 0 else 0.0,
        "minor_axis_norm": frag["minor"] / mean_major if mean_major > 0 else 0.0,
        "interior_intensity": interior,
        "context_intensity": context,
    }


def _node_bbox_xyxy(frag, pad_frac, shape):
    r0, c0, r1, c1 = frag["bbox"]
    h, w = r1 - r0, c1 - c0
    py, px = pad_frac * h, pad_frac * w
    x1 = max(c0 - px, 0.0)
    y1 = max(r0 - py, 0.0)
    x2 = min(c1 + px, float(shape[1]))
    y2 = min(r1 + py, float(shape[0]))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


from skimage.measure import profile_line


def _fold_angle(a):
    """Fold an angle difference into [0, pi/2] then scale to [0, 1]."""
    d = abs(a) % np.pi
    d = np.pi - d if d > np.pi / 2 else d
    return d / (np.pi / 2)


def _edge_feature_row(fi, fj, dmin, pi, pj, ti, tj, dic,
                      mean_major, contact_tau, gap_linewidth, continuity_L):
    # connecting direction (row/col) from i's point to j's point
    dy, dx = float(pj[0] - pi[0]), float(pj[1] - pi[1])
    seg_len = np.hypot(dy, dx)
    path_angle = np.arctan2(dx, dy)

    # gap_intensity (col 0): mean DIC along the p_i->p_j segment, endpoints dropped;
    # for touching fragments (tiny segment) fall back to a thin dilation-contact band.
    if seg_len >= 1.0:
        prof = profile_line(dic, pi, pj, linewidth=gap_linewidth, mode="constant", cval=0.0)
        inner = prof[1:-1] if prof.shape[0] > 2 else prof
        gap_intensity = float(inner.mean()) if inner.size > 0 else float(dic[tuple(pi)])
    else:
        gap_intensity = float((dic[tuple(pi)] + dic[tuple(pj)]) / 2.0)

    boundary_dist_norm = dmin / mean_major if mean_major > 0 else dmin
    node1 = _fold_angle(path_angle - fi["orientation"])
    node2 = _fold_angle(path_angle - fj["orientation"])
    relative = _fold_angle(fi["orientation"] - fj["orientation"])

    # contact_frac: fraction of the smaller mask's boundary within contact_tau of the other
    if fi["area"] <= fj["area"]:
        small, other_tree = fi, tj
    else:
        small, other_tree = fj, ti
    d_small, _ = other_tree.query(small["boundary"], k=1)
    contact_frac = float(np.mean(d_small <= contact_tau))

    area_ratio = min(fi["area"], fj["area"]) / max(fi["area"], fj["area"])
    axis_collinearity = abs(np.cos(fi["orientation"] - fj["orientation"]))

    # intensity_continuity: correlate DIC sampled inward from each junction point.
    if seg_len > 0:
        u = np.array([dy, dx]) / seg_len
        end_i = pi - u * continuity_L        # into fragment i (away from j)
        end_j = pj + u * continuity_L        # into fragment j (away from i)
        prof_i = profile_line(dic, pi, end_i, mode="constant", cval=0.0)
        prof_j = profile_line(dic, pj, end_j, mode="constant", cval=0.0)
        L = min(prof_i.shape[0], prof_j.shape[0])
        if L >= 3 and prof_i[:L].std() > 0 and prof_j[:L].std() > 0:
            intensity_continuity = float(np.corrcoef(prof_i[:L], prof_j[:L])[0, 1])
        else:
            intensity_continuity = 0.0
    else:
        intensity_continuity = 0.0

    return {
        "gap_intensity": gap_intensity,
        "boundary_dist_norm": boundary_dist_norm,
        "node1_angle_diff": node1,
        "node2_angle_diff": node2,
        "min_diff_angle": min(node1, node2),
        "relative_angle": relative,
        "contact_frac": contact_frac,
        "area_ratio": area_ratio,
        "axis_collinearity": axis_collinearity,
        "intensity_continuity": intensity_continuity,
    }


def extract_cell_graph(ais_labels, intensity_image, k=6, prefilter_mult=3,
                       dist_cap_factor=None, ring_width=5, contact_tau=2.0,
                       gap_linewidth=3, continuity_L=5, bbox_pad_frac=0.1):
    """Build fragment nodes and kNN candidate edges from an AIS label map.

    `intensity_image` must be a single 2D channel. Reduce a multi-channel image
    upstream with `ImageContainer([[*channel_paths]], config).merge()`, which
    sums the channels and stretches the result to full contrast.
    """
    dic = np.asarray(intensity_image, dtype=np.float32)
    if dic.ndim != 2:
        # A channel stack silently corrupts every intensity feature rather than
        # failing: profile_line returns (L, C), so intensity_continuity's
        # corrcoef stops comparing the two profiles and pins itself to +/-1.
        raise ValueError(
            f"intensity_image must be a single 2D channel, got shape {dic.shape}. "
            "Reduce channels upstream with "
            "ImageContainer([[*channel_paths]], config).merge()."
        )
    frags, trees, centroids, mean_area, mean_major = _extract_fragments(ais_labels, dic)

    node_rows = []
    node_bboxes = []
    centroid_list = []
    for idx, f in enumerate(frags):
        row = _node_feature_row(f, dic, mean_area, mean_major, ring_width)
        row["node_id"] = idx
        node_rows.append(row)
        node_bboxes.append(_node_bbox_xyxy(f, bbox_pad_frac, ais_labels.shape))
        centroid_list.append((float(f["centroid"][0]), float(f["centroid"][1])))
    node_df = pd.DataFrame(node_rows, columns=["node_id"] + NODE_FEATURE_COLUMNS)
    node_bboxes = (np.stack(node_bboxes) if node_bboxes
                   else np.zeros((0, 4), dtype=np.float32))

    dist_cap = None if dist_cap_factor is None else dist_cap_factor * mean_major
    candidates = _knn_edges(frags, trees, centroids, k=k,
                            prefilter_mult=prefilter_mult, dist_cap=dist_cap)

    edge_rows, src, tgt = [], [], []
    for i, j, dmin, pi, pj in candidates:
        feat = _edge_feature_row(
            frags[i], frags[j], dmin, pi, pj,
            trees[frags[i]["label"]], trees[frags[j]["label"]],
            dic, mean_major, contact_tau, gap_linewidth, continuity_L,
        )
        feat["source_node"], feat["target_node"] = i, j
        edge_rows.append(feat)
        src.append(i)
        tgt.append(j)
    edge_df = pd.DataFrame(edge_rows,
                           columns=["source_node", "target_node"] + EDGE_FEATURE_COLUMNS)
    return node_df, centroid_list, node_bboxes, edge_df, [src, tgt]