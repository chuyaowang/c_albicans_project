"""
Interpretation utilities for the GCN model.

Two analyses are provided, both computed at the best early-stopping epoch only:

  1. Pre-logit embedding visualization (PCA, PLS-DA)
     The D-dimensional vector produced by Classifier.mlp_body immediately before
     the final linear layer is extracted for every test edge. Scatter plots in
     PCA and PLS-DA space are color-coded by TP / TN / FP / FN.

  2. Per-edge gradient × input attribution heatmap
     For each edge prediction, the sensitivity of the logit to each input feature
     is measured as |∂pred_e/∂input × input|. Node features are attributed
     separately for the source and target node of the edge. Visual CNN outputs
     (if the visual branch is active) are kept as individual latent dimensions
     rather than aggregated, to enable direct correlation analysis between
     manual and microsam features (~100+ columns total).
     Columns are globally hierarchically clustered (Ward linkage) so that
     features with similar attribution patterns across edges appear adjacent.
     Group membership (Node src/tgt, Edge, Visual src/tgt/edge) is shown via
     colored x-tick labels and a legend patch.
     Values are log-transformed and row-normalized so each row (edge) sums to 1,
     enabling direct comparison of feature contributions within a single edge.

The primary output is plot_combined_figure(), which renders all three analyses
in a two-row layout: PCA and PLS-DA (square) on the top row, attribution
heatmap spanning the full width on the bottom row.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import torch

# Display palette and display order for the four prediction classes
_CLASS_COLORS = {
    'TP': '#2ca02c',
    'TN': '#1f77b4',
    'FP': '#d62728',
    'FN': '#ff7f0e',
}
_CLASS_ORDER = ['TP', 'TN', 'FP', 'FN']

# Short feature names matching the column order in data.x and data.edge_attr
_NODE_FEAT_NAMES = ['circ', 'ecc', 'area', 'int', 'maj', 'min']
_EDGE_FEAT_NAMES = ['e_int', 'e_len', 'ang1', 'ang2', 'min_ang', 'rel_ang']

# One distinct color per feature group, used for x-tick label annotation
_FEAT_GROUP_COLORS = {
    'Node (src)':    '#e41a1c',
    'Node (tgt)':    '#ff7f00',
    'Edge':          '#4daf4a',
    'Visual (src)':  '#984ea3',
    'Visual (tgt)':  '#377eb8',
    'Visual (edge)': '#a65628',
}


# ─────────────────────────────────────────────────────────────────────────────
# Core data collection
# ─────────────────────────────────────────────────────────────────────────────

def classify_edges(probs, true_labels, threshold):
    """Assign each edge a TP / TN / FP / FN string label.

    Args:
        probs:       (E,) float array of (symmetrized) predicted probabilities.
        true_labels: (E,) int/float array of ground-truth edge labels (0 or 1).
        threshold:   float, decision threshold.

    Returns:
        np.ndarray of shape (E,) with values in {'TP', 'TN', 'FP', 'FN'}.
    """
    preds = (probs >= threshold).astype(int)
    tgts  = true_labels.astype(int)
    result = np.empty(len(probs), dtype=object)
    result[(preds == 1) & (tgts == 1)] = 'TP'
    result[(preds == 0) & (tgts == 0)] = 'TN'
    result[(preds == 1) & (tgts == 0)] = 'FP'
    result[(preds == 0) & (tgts == 1)] = 'FN'
    return result.astype(str)


def collect_embeddings(model, data, device):
    """Extract pre-logit embeddings and raw predictions for every edge.

    The embedding is the output of Classifier.mlp_body (post-dropout, before
    the final Linear layer). Symmetry is NOT applied here; callers should apply
    enforce_symmetric_predictions to get the final classification probabilities.

    Args:
        model:  trained Model instance.
        data:   single unbatched PyG Data object.
        device: torch device.

    Returns:
        emb_np:       (E, D) float32 ndarray — pre-logit embeddings.
        raw_probs_np: (E,)   float32 ndarray — raw (non-symmetrized) sigmoid output.
    """
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        raw_pred, emb = model(data, return_embeddings=True)
    return emb.cpu().numpy().astype(np.float32), raw_pred.cpu().numpy().astype(np.float32)


def compute_per_edge_attributions(model, data, device):
    """Compute gradient × input attribution for every edge prediction.

    One backward pass is performed per edge. The attribution for edge e is
    |∂pred_e/∂input × input| collected for all tracked input tensors:
      - data.x          (tabular node features)
      - data.edge_attr  (tabular edge features)
      - node_visual / edge_visual (visual CNN outputs, if use_visual_features=True)

    Visual-branch attributions are kept as individual latent dimensions
    (not aggregated) to enable correlation analysis between manual and
    microsam features. Six groups are returned: Node (src), Node (tgt),
    Edge, Visual (src), Visual (tgt), Visual (edge).

    Model parameter gradients are suppressed during this computation to avoid
    unnecessary computation.

    Args:
        model:  trained Model instance.
        data:   single unbatched PyG Data object.
        device: torch device.

    Returns:
        attr_matrix:   (E, F) float32 ndarray of raw attribution values.
                       F = 18 tabular columns + 3*d_vis visual columns (if visual).
        feature_names: list of F column label strings.
        groups:        list of (label, start_col, end_col) tuples for annotation.
    """
    model.eval()
    data = data.to(device)

    # Suppress model-parameter gradients: we only need gradients on the input tensors.
    model.requires_grad_(False)
    try:
        pred, attr_tensors = model(data, attribution_mode=True)
    finally:
        model.requires_grad_(True)

    x  = attr_tensors['x']           # (N, 6)
    ea = attr_tensors['edge_attr']    # (E, 6)
    use_visual = 'node_visual' in attr_tensors
    if use_visual:
        nv    = attr_tensors['node_visual']  # (N, d_vis)
        ev    = attr_tensors['edge_visual']  # (E, d_vis)
        d_vis = nv.size(1)

    edge_index = data.edge_index
    num_edges  = edge_index.size(1)
    n_nf  = x.size(1)   # 6
    n_ef  = ea.size(1)  # 6
    n_tab = 2 * n_nf + n_ef
    n_cols = n_tab + (3 * d_vis if use_visual else 0)

    attr_matrix = np.zeros((num_edges, n_cols), dtype=np.float32)

    for e in range(num_edges):
        for t in attr_tensors.values():
            if t.grad is not None:
                t.grad.zero_()

        pred[e].backward(retain_graph=(e < num_edges - 1))

        src = edge_index[0, e].item()
        tgt = edge_index[1, e].item()

        col = 0
        attr_matrix[e, col:col + n_nf] = (x.grad[src] * x[src]).abs().detach().cpu().numpy()
        col += n_nf
        attr_matrix[e, col:col + n_nf] = (x.grad[tgt] * x[tgt]).abs().detach().cpu().numpy()
        col += n_nf
        attr_matrix[e, col:col + n_ef] = (ea.grad[e] * ea[e]).abs().detach().cpu().numpy()
        col += n_ef
        if use_visual:
            attr_matrix[e, col:col + d_vis]             = (nv.grad[src] * nv[src]).abs().detach().cpu().numpy()
            attr_matrix[e, col + d_vis:col + 2 * d_vis] = (nv.grad[tgt] * nv[tgt]).abs().detach().cpu().numpy()
            attr_matrix[e, col + 2 * d_vis:col + 3 * d_vis] = (ev.grad[e] * ev[e]).abs().detach().cpu().numpy()

    feature_names = (
        [f'src_{n}' for n in _NODE_FEAT_NAMES] +
        [f'tgt_{n}' for n in _NODE_FEAT_NAMES] +
        list(_EDGE_FEAT_NAMES)
    )
    groups = [
        ('Node (src)', 0,       n_nf),
        ('Node (tgt)', n_nf,    2 * n_nf),
        ('Edge',       2 * n_nf, n_tab),
    ]
    if use_visual:
        feature_names += (
            [f'vs_{k}' for k in range(d_vis)] +
            [f'vt_{k}' for k in range(d_vis)] +
            [f've_{k}' for k in range(d_vis)]
        )
        groups += [
            ('Visual (src)',  n_tab,              n_tab + d_vis),
            ('Visual (tgt)',  n_tab + d_vis,      n_tab + 2 * d_vis),
            ('Visual (edge)', n_tab + 2 * d_vis,  n_cols),
        ]

    return attr_matrix, feature_names, groups


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sort_indices(probs, edge_classes):
    """Sort edge indices by class (TP→TN→FP→FN) with within-class ordering.

    TP: ascending by prob  (borderline TPs, closest to threshold, appear first).
    TN: descending by prob (uncertain TNs, closest to threshold, appear first).
    FP: ascending by prob  (borderline FPs first).
    FN: ascending by prob  (borderline FNs first).
    """
    indices = []
    for cls in _CLASS_ORDER:
        mask = np.where(edge_classes == cls)[0]
        if len(mask) == 0:
            continue
        cls_probs = probs[mask]
        order = np.argsort(-cls_probs if cls == 'TN' else cls_probs)
        indices.extend(mask[order].tolist())
    return np.array(indices, dtype=int)


def _class_boundaries(sorted_classes):
    """Row indices where the class label changes (used to draw dividers)."""
    return [i for i in range(1, len(sorted_classes))
            if sorted_classes[i] != sorted_classes[i - 1]]


def _hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _cluster_columns(attr_norm, feature_names, groups):
    """Reorder columns by Ward hierarchical clustering on the attribution matrix.

    Features with similar attribution patterns across edges are placed adjacent.
    Group membership is encoded as a per-column color so it survives reordering.

    Args:
        attr_norm:     (E, F) row-normalised log-attribution matrix.
        feature_names: list of F column label strings (original order).
        groups:        list of (label, start, end) tuples (original order).

    Returns:
        attr_c:   (E, F) column-reordered matrix.
        names_c:  list of F reordered column label strings.
        colors_c: list of F matplotlib color strings (one per column, by group).
    """
    n_feat = attr_norm.shape[1]

    orig_colors = ['#888888'] * n_feat
    for label, start, end in groups:
        c = _FEAT_GROUP_COLORS.get(label, '#888888')
        for i in range(start, end):
            orig_colors[i] = c

    col_order = np.arange(n_feat)
    if n_feat >= 2 and attr_norm.shape[0] >= 2:
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            Z = linkage(attr_norm.T, method='ward', metric='euclidean')
            col_order = leaves_list(Z)
        except Exception:
            pass  # fall back to natural order if scipy unavailable

    attr_c   = attr_norm[:, col_order]
    names_c  = [feature_names[i] for i in col_order]
    colors_c = [orig_colors[i] for i in col_order]
    return attr_c, names_c, colors_c


def _prepare_heatmap_data(attr_matrix, probs, edge_classes, feature_names, groups):
    """Sort rows, apply log + row normalisation, then cluster columns.

    Args:
        attr_matrix:   (E, F) raw attribution matrix.
        probs:         (E,) predicted probabilities.
        edge_classes:  (E,) class label strings.
        feature_names: list of F column label strings.
        groups:        list of (label, start_col, end_col) tuples.

    Returns:
        attr_norm:   (E, F) normalised, column-clustered array.
        probs_s:     (E,) sorted probabilities.
        classes_s:   (E,) sorted class labels.
        vmax:        float, global max of attr_norm.
        feat_names:  list of F reordered column label strings.
        feat_colors: list of F matplotlib color strings for tick annotation.
    """
    sorted_idx = _sort_indices(probs, edge_classes)
    attr_s    = attr_matrix[sorted_idx]
    probs_s   = probs[sorted_idx]
    classes_s = edge_classes[sorted_idx]

    # Log-transform then row-normalize
    attr_log = np.log1p(attr_s)
    row_sum  = attr_log.sum(axis=1, keepdims=True)
    row_sum  = np.where(row_sum < 1e-8, 1.0, row_sum)
    attr_norm = attr_log / row_sum

    attr_norm, feat_names, feat_colors = _cluster_columns(attr_norm, feature_names, groups)
    return attr_norm, probs_s, classes_s, float(attr_norm.max()) or 1.0, feat_names, feat_colors


# ─────────────────────────────────────────────────────────────────────────────
# Axis-level fill functions (shared by standalone and combined figures)
# ─────────────────────────────────────────────────────────────────────────────

def _fill_pca_ax(ax, embeddings, edge_classes,
                 train_embeddings=None, train_true_labels=None):
    from matplotlib.lines import Line2D
    has_train = train_embeddings is not None and train_true_labels is not None

    # Fit on combined data so test and train share the same coordinate system
    all_emb = np.vstack([embeddings, train_embeddings]) if has_train else embeddings
    pca = PCA(n_components=2)
    pca.fit(all_emb)
    var          = pca.explained_variance_ratio_
    coords       = pca.transform(embeddings)
    train_coords = pca.transform(train_embeddings) if has_train else None

    for cls in _CLASS_ORDER:
        mask = edge_classes == cls
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=_CLASS_COLORS[cls],
                       alpha=0.85, s=45, edgecolors='k', linewidths=0.4,
                       zorder=3)

    # Training edges colored by true label (pos=1 → TP-green, neg=0 → TN-blue),
    # plotted behind test points as open dashed circles.
    if has_train:
        train_lbl = train_true_labels.astype(int)
        for lv, color, name in [(1, _CLASS_COLORS['TP'], 'pos (tr)'),
                                 (0, _CLASS_COLORS['TN'], 'neg (tr)')]:
            tmask = train_lbl == lv
            if tmask.any():
                sc = ax.scatter(train_coords[tmask, 0], train_coords[tmask, 1],
                                facecolors='none', edgecolors=color,
                                alpha=0.7, s=40, linewidths=1.2, zorder=2)
                sc.set_linestyle('--')

    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)', fontsize=8)
    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)', fontsize=8)
    ax.set_title('PCA — pre-logit embeddings', fontsize=9)
    ax.set_box_aspect(1)

    handles = []
    for cls in _CLASS_ORDER:
        if np.any(edge_classes == cls):
            handles.append(Line2D([0], [0], linestyle='none', marker='o',
                                  markerfacecolor=_CLASS_COLORS[cls],
                                  markeredgecolor='k', markeredgewidth=0.4,
                                  markersize=6, label=cls))
    if has_train:
        train_lbl = train_true_labels.astype(int)
        for lv, color, name in [(1, _CLASS_COLORS['TP'], 'pos (tr)'),
                                 (0, _CLASS_COLORS['TN'], 'neg (tr)')]:
            if np.any(train_lbl == lv):
                handles.append(Line2D([0], [0], linestyle='none', marker='o',
                                      markerfacecolor='none',
                                      markeredgecolor=color,
                                      markeredgewidth=1.2, markersize=6,
                                      label=name))
    ax.legend(handles=handles, fontsize=6,
              ncol=2 if has_train else 1,
              handletextpad=0.3, columnspacing=0.8)


def _fill_plsda_ax(ax, embeddings, true_labels, edge_classes,
                   train_embeddings=None, train_true_labels=None):
    from matplotlib.lines import Line2D
    has_train = train_embeddings is not None and train_true_labels is not None
    try:
        # Fit on combined data so both sets share the same latent space
        if has_train:
            all_emb    = np.vstack([embeddings, train_embeddings])
            all_labels = np.concatenate([true_labels.astype(float),
                                         train_true_labels.astype(float)])
        else:
            all_emb, all_labels = embeddings, true_labels.astype(float)

        pls = PLSRegression(n_components=2)
        pls.fit(all_emb, all_labels)

        # R²X(k) = SS(t_k p_k') / SS(X_centered), re-running NIPALS deflation.
        # x_weights_ holds the unit-norm NIPALS w vectors; x_scores_ = X @ W* which
        # uses a rotated basis and can exceed SS_tot.
        X_c    = all_emb - all_emb.mean(axis=0)
        SS_tot = np.sum(X_c ** 2) + 1e-8
        var_lv = np.zeros(2)
        X_res  = X_c.copy()
        for k in range(2):
            w = pls.x_weights_[:, k]
            t = X_res @ w
            p = X_res.T @ t / (np.dot(t, t) + 1e-8)
            var_lv[k] = np.dot(t, t) * np.dot(p, p) / SS_tot
            X_res -= np.outer(t, p)

        result = pls.transform(embeddings)
        coords = result[0] if isinstance(result, tuple) else result
        if has_train:
            result_tr    = pls.transform(train_embeddings)
            train_coords = result_tr[0] if isinstance(result_tr, tuple) else result_tr

        for cls in _CLASS_ORDER:
            mask = edge_classes == cls
            if mask.any():
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=_CLASS_COLORS[cls],
                           alpha=0.85, s=45, edgecolors='k', linewidths=0.4,
                           zorder=3)

        # Training edges colored by true label (pos=1 → TP-green, neg=0 → TN-blue)
        if has_train:
            train_lbl = train_true_labels.astype(int)
            for lv, color in [(1, _CLASS_COLORS['TP']), (0, _CLASS_COLORS['TN'])]:
                tmask = train_lbl == lv
                if tmask.any():
                    sc = ax.scatter(train_coords[tmask, 0], train_coords[tmask, 1],
                                    facecolors='none', edgecolors=color,
                                    alpha=0.7, s=40, linewidths=1.2, zorder=2)
                    sc.set_linestyle('--')

        ax.set_xlabel(f'LV1 ({var_lv[0]*100:.1f}%)', fontsize=8)
        ax.set_ylabel(f'LV2 ({var_lv[1]*100:.1f}%)', fontsize=8)
        ax.set_title('PLS-DA — pre-logit embeddings', fontsize=9)
        ax.set_box_aspect(1)

        handles = []
        for cls in _CLASS_ORDER:
            if np.any(edge_classes == cls):
                handles.append(Line2D([0], [0], linestyle='none', marker='o',
                                      markerfacecolor=_CLASS_COLORS[cls],
                                      markeredgecolor='k', markeredgewidth=0.4,
                                      markersize=6, label=cls))
        if has_train:
            train_lbl = train_true_labels.astype(int)
            for lv, color, name in [(1, _CLASS_COLORS['TP'], 'pos (tr)'),
                                     (0, _CLASS_COLORS['TN'], 'neg (tr)')]:
                if np.any(train_lbl == lv):
                    handles.append(Line2D([0], [0], linestyle='none', marker='o',
                                          markerfacecolor='none',
                                          markeredgecolor=color,
                                          markeredgewidth=1.2, markersize=6,
                                          label=name))
        ax.legend(handles=handles, fontsize=6,
                  ncol=2 if has_train else 1,
                  handletextpad=0.3, columnspacing=0.8)

    except Exception as exc:
        ax.text(0.5, 0.5, f'PLS-DA failed:\n{exc}',
                transform=ax.transAxes, ha='center', va='center', fontsize=8)
        ax.set_title('PLS-DA — failed', fontsize=9)
        ax.set_box_aspect(1)


def _fill_heatmap_axes(ax_cls, ax_prob, ax_attr,
                       attr_norm, probs_s, classes_s,
                       feat_names, feat_colors, vmax):
    """Render the three-panel heatmap section.

    Group membership is conveyed via colored x-tick labels and a legend;
    columns have been globally clustered so groups are no longer contiguous.

    Args:
        feat_names:  list of F column label strings (clustering-reordered).
        feat_colors: list of F matplotlib color strings for x-tick label coloring.
    """
    n_edges, n_feat = attr_norm.shape

    # ── Class color strip ──────────────────────────────────────────────────
    cls_rgb = np.array(
        [_hex_to_rgb(_CLASS_COLORS[c]) for c in classes_s],
        dtype=np.float32,
    ).reshape(-1, 1, 3)
    ax_cls.imshow(cls_rgb, aspect='auto', interpolation='nearest')
    ax_cls.set_xticks([])
    ax_cls.set_yticks(range(n_edges))
    ax_cls.set_yticklabels(classes_s, fontsize=6)
    ax_cls.set_title('cls', fontsize=7, pad=3)

    # ── Predicted probability column ───────────────────────────────────────
    ax_prob.imshow(probs_s.reshape(-1, 1), aspect='auto', cmap='Greens',
                   vmin=0.0, vmax=1.0, interpolation='nearest')
    for i, p in enumerate(probs_s):
        text_color = 'white' if p > 0.65 else 'black'
        ax_prob.text(0, i, f'{p:.2f}', ha='center', va='center',
                     fontsize=5.5, color=text_color)
    ax_prob.set_xticks([])
    ax_prob.set_yticks([])
    ax_prob.set_title('prob', fontsize=7, pad=3)

    # ── Attribution heatmap ────────────────────────────────────────────────
    im = ax_attr.imshow(attr_norm, aspect='auto', cmap='hot',
                        vmin=0.0, vmax=vmax, interpolation='nearest')
    ax_attr.set_xticks(range(n_feat))
    ax_attr.set_xticklabels(feat_names, rotation=90, fontsize=5)
    ax_attr.set_yticks([])
    ax_attr.set_title(
        'Attribution  (log|∇×input|, row-normalised, cols Ward-clustered)',
        fontsize=7, pad=3,
    )

    # Color each x-tick label by its group
    for tick, color in zip(ax_attr.get_xticklabels(), feat_colors):
        tick.set_color(color)

    # Group legend: show only the groups that are actually present
    present_colors = set(feat_colors)
    legend_elements = [
        Patch(facecolor=c, label=lbl)
        for lbl, c in _FEAT_GROUP_COLORS.items()
        if c in present_colors
    ]
    ax_attr.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=5.5, ncol=2,
        framealpha=0.75,
        handlelength=1.0, handleheight=0.8,
        borderpad=0.4, labelspacing=0.2,
    )

    # Horizontal class-boundary dividers across all three panels
    for bnd in _class_boundaries(classes_s):
        for ax in (ax_cls, ax_prob, ax_attr):
            ax.axhline(y=bnd - 0.5, color='white', linewidth=1.0)

    cb = ax_attr.get_figure().colorbar(im, ax=ax_attr, fraction=0.015, pad=0.01)
    cb.ax.tick_params(labelsize=6)
    cb.set_label('row share', fontsize=6)


# ─────────────────────────────────────────────────────────────────────────────
# Public visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_combined_figure(embeddings, true_labels, edge_classes,
                         attr_matrix, probs, feature_names, groups,
                         train_embeddings=None, train_true_labels=None):
    """Combined interpretation figure for one test fold.

    Layout (two rows):
      Row 0 (fixed, square):  PCA scatter  |  PLS-DA scatter.
      Row 1 (edge-scaled):    Class strip  |  prob column  |  attribution heatmap.

    Both scatter plots use set_box_aspect(1) to enforce a square axis box.
    Attribution columns are globally Ward-clustered so correlated features
    (manual and visual) appear adjacent.

    If train_embeddings / train_true_labels are provided, training edges are
    overlaid on the scatter plots as open dashed circles colored by true label
    (positive=1 → TP-green, negative=0 → TN-blue) so the test embeddings can
    be compared against the training distribution without applying any threshold
    to the training data.  PCA and PLS-DA are fit on the combined data so both
    sets share the same coordinate system.

    Args:
        embeddings:        (E, D) ndarray — test pre-logit embeddings.
        true_labels:       (E,)   int/float array of ground-truth (0 or 1).
        edge_classes:      (E,)   string array from classify_edges (test).
        attr_matrix:       (E, F) float32 ndarray from compute_per_edge_attributions.
        probs:             (E,)   float array — symmetrized predicted probabilities.
        feature_names:     list of F column label strings.
        groups:            list of (label, start_col, end_col).
        train_embeddings:  (E_tr, D) ndarray — training pre-logit embeddings, optional.
        train_true_labels: (E_tr,)  int/float ground-truth for training edges, optional.

    Returns:
        matplotlib Figure.
    """
    attr_norm, probs_s, classes_s, vmax, feat_names, feat_colors = \
        _prepare_heatmap_data(attr_matrix, probs, edge_classes, feature_names, groups)
    n_edges, n_feat = attr_norm.shape

    # ── Sizing ──────────────────────────────────────────────────────────────
    scatter_sz    = 4.5
    scatter_row_h = scatter_sz + 1.2      # extra for axis labels / title
    row_h         = max(0.20, min(0.38, 8.0 / max(n_edges, 1)))
    heatmap_row_h = max(5.0, n_edges * row_h + 2.5)
    fig_h         = scatter_row_h + heatmap_row_h + 0.8

    col_w           = max(0.10, min(0.18, 18.0 / max(n_feat, 1)))
    heatmap_data_w  = max(8.0, n_feat * col_w)
    heatmap_total_w = heatmap_data_w + 1.2   # cls strip + prob col
    scatter_total_w = 2 * scatter_sz + 2.5   # two squares + gap + margins
    fig_w = max(scatter_total_w, heatmap_total_w) + 0.5

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs_outer = fig.add_gridspec(
        2, 1,
        height_ratios=[scatter_row_h, heatmap_row_h],
        hspace=0.45,
        left=0.04, right=0.97, top=0.95, bottom=0.10,
    )

    # Top row: PCA (left) and PLS-DA (right), both square
    gs_scatter = gs_outer[0].subgridspec(1, 2, wspace=0.35)
    ax_pca = fig.add_subplot(gs_scatter[0])
    ax_pls = fig.add_subplot(gs_scatter[1])

    # Bottom row: class strip | prob column | attribution heatmap
    gs_heat = gs_outer[1].subgridspec(
        1, 3,
        width_ratios=[0.45, 0.65, heatmap_data_w],
        wspace=0.03,
    )
    ax_cls  = fig.add_subplot(gs_heat[0])
    ax_prob = fig.add_subplot(gs_heat[1])
    ax_attr = fig.add_subplot(gs_heat[2])

    _fill_pca_ax(ax_pca, embeddings, edge_classes,
                 train_embeddings, train_true_labels)
    _fill_plsda_ax(ax_pls, embeddings, true_labels, edge_classes,
                   train_embeddings, train_true_labels)
    _fill_heatmap_axes(ax_cls, ax_prob, ax_attr,
                       attr_norm, probs_s, classes_s,
                       feat_names, feat_colors, vmax)
    return fig


def plot_pca_figure(embeddings, edge_classes,
                    train_embeddings=None, train_true_labels=None):
    """Standalone 2-D PCA scatter. See plot_combined_figure for the primary output."""
    fig, ax = plt.subplots(figsize=(5, 5))
    _fill_pca_ax(ax, embeddings, edge_classes, train_embeddings, train_true_labels)
    fig.tight_layout()
    return fig


def plot_plsda_figure(embeddings, true_labels, edge_classes,
                      train_embeddings=None, train_true_labels=None):
    """Standalone PLS-DA scatter. See plot_combined_figure for the primary output."""
    fig, ax = plt.subplots(figsize=(5, 5))
    _fill_plsda_ax(ax, embeddings, true_labels, edge_classes,
                   train_embeddings, train_true_labels)
    fig.tight_layout()
    return fig


def plot_attribution_heatmap(attr_matrix, probs, edge_classes, feature_names, groups):
    """Standalone attribution heatmap. See plot_combined_figure for the primary output."""
    attr_norm, probs_s, classes_s, vmax, feat_names, feat_colors = \
        _prepare_heatmap_data(attr_matrix, probs, edge_classes, feature_names, groups)
    n_edges, n_feat = attr_norm.shape

    row_h = max(0.20, min(0.38, 8.0 / max(n_edges, 1)))
    col_w = max(0.10, min(0.18, 18.0 / max(n_feat, 1)))
    fig_h = max(4.0, n_edges * row_h + 2.5)
    heatmap_data_w = max(8.0, n_feat * col_w)
    fig_w = heatmap_data_w + 1.2 + 0.5

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[0.45, 0.65, heatmap_data_w],
        wspace=0.03,
        left=0.02, right=0.96, top=0.86, bottom=0.25,
    )
    ax_cls  = fig.add_subplot(gs[0])
    ax_prob = fig.add_subplot(gs[1])
    ax_attr = fig.add_subplot(gs[2])
    _fill_heatmap_axes(ax_cls, ax_prob, ax_attr,
                       attr_norm, probs_s, classes_s,
                       feat_names, feat_colors, vmax)
    return fig