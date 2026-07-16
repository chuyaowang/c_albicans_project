import copy
from datetime import datetime

import numpy as np
import torch
from torch.optim import Muon, AdamW
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_recall_curve,
)

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches

from image_processing_tools.dapi_tracing.simple_gnn import Model, _node_boxes, _edge_boxes
from image_processing_tools.dapi_tracing.gnn_data import create_data_loader
from image_processing_tools.dapi_tracing.gnn_interpret import (
    collect_embeddings, classify_edges, compute_per_edge_attributions,
    plot_combined_figure, plot_pca_figure,
)


def enforce_symmetric_predictions(pred, edge_index, num_nodes):
    """
    Averages the predicted probabilities of forward and reverse edges
    to ensure symmetric predictions for undirected graphs.
    """
    mask = edge_index[0] != edge_index[1]
    if not mask.any():
        return pred

    masked_edge_index = edge_index[:, mask]
    min_node = torch.min(masked_edge_index[0], masked_edge_index[1])
    max_node = torch.max(masked_edge_index[0], masked_edge_index[1])
    edge_hash = min_node * num_nodes + max_node

    _, sorted_idx = torch.sort(edge_hash)
    orig_indices = mask.nonzero(as_tuple=True)[0][sorted_idx]

    idx_forward = orig_indices[0::2]
    idx_reverse = orig_indices[1::2]

    avg_pred = (pred[idx_forward] + pred[idx_reverse]) / 2.0

    sym_pred = pred.clone()
    sym_pred[idx_forward] = avg_pred
    sym_pred[idx_reverse] = avg_pred

    return sym_pred


def enforce_symmetric_max(pred, edge_index, num_nodes):
    """
    Takes the maximum of the predicted probabilities of forward and reverse edges.
    Forces the topological mask to confidently pick a direction to minimize BCE loss.

    Args:
        pred (torch.Tensor): The directed predicted probabilities for each edge.
        edge_index (torch.Tensor): Graph connectivity matrix of shape [2, num_edges].
        num_nodes (int): Total number of nodes across the batched graphs.

    Returns:
        torch.Tensor: Symmetrized predicted probabilities securely clamped between 0.0 and 1.0.
    """
    mask = edge_index[0] != edge_index[1]
    if not mask.any():
        return pred

    masked_edge_index = edge_index[:, mask]
    min_node = torch.min(masked_edge_index[0], masked_edge_index[1])
    max_node = torch.max(masked_edge_index[0], masked_edge_index[1])
    edge_hash = min_node * num_nodes + max_node

    _, sorted_idx = torch.sort(edge_hash)
    orig_indices = mask.nonzero(as_tuple=True)[0][sorted_idx]

    idx_forward = orig_indices[0::2]
    idx_reverse = orig_indices[1::2]

    # Take the MAX of the directed components to force confident directional masking
    max_pred = torch.max(pred[idx_forward], pred[idx_reverse])

    # Use scatter to avoid inplace autograd errors during backprop
    sym_pred = pred.clone()
    sym_pred = sym_pred.scatter(0, idx_forward, max_pred)
    sym_pred = sym_pred.scatter(0, idx_reverse, max_pred)

    return torch.clamp(sym_pred, 0.0, 1.0)


def get_muon_optimizers(model, learning_rate, adam_lr_factor=1.0, muon_weight_decay=0.1, adam_weight_decay=0.01):
    """
    Separates model parameters into groups for the Muon and AdamW optimizers
    as recommended by the Muon documentation.

    - Muon optimizes 2D weight matrices of hidden layers.
    - AdamW optimizes everything else (biases, batchnorm params, classifier head).
    """
    muon_params = []
    adam_params = []

    for name, param in model.named_parameters():
        if 'classifier' in name:
            adam_params.append(param)
        elif 'weight' in name and param.ndim == 2:
            muon_params.append(param)
        else:
            adam_params.append(param)

    # Muon has different hyperparameter defaults and sensitivities
    # Using defaults from the official implementation where appropriate.
    opt_muon = Muon(muon_params, lr=learning_rate, weight_decay=muon_weight_decay, momentum=0.95, nesterov=True)
    opt_adam = AdamW(adam_params, lr=learning_rate * adam_lr_factor, weight_decay=adam_weight_decay)

    return [opt_muon, opt_adam]


def train_model(model, loader, optimizers, criterion, degree_penalty_weight=0.0, neg_sample_ratio=1.0, label_smoothing=0.0):
    """
    Performs one epoch of training for an edge classification task.

    Args:
        model (torch.nn.Module): The GNN model to train.
        loader (DataLoader): The DataLoader containing the training graphs.
        optimizers (list): A list of optimizers for updating model weights.
        criterion (callable): The loss function.
        degree_penalty_weight (float): The weight for the degree constraint penalty.
        neg_sample_ratio (float): Ratio of negative to positive edges to sample per batch.

    Returns:
        tuple: The average loss and accuracy over the training dataset.
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_bce_loss = 0
    total_penalty_loss = 0
    total_unsampled_bce = 0
    total_pred_mean = 0
    total_pred_std = 0

    device = next(model.parameters()).device

    for data in loader:
        data = data.to(device)

        # Zero gradients for all optimizers
        for opt in optimizers:
            opt.zero_grad()

        # 1. Get predictions for all edges in the batch. The model's forward pass handles this.
        pred = model(data)

        # Enforce perfectly symmetric predictions
        pred = enforce_symmetric_predictions(pred, data.edge_index, data.num_nodes)

        # 2. Ground truth labels are stored in data.edge_label
        ground_truth = data.edge_label

        # --- Negative Edge Subsampling for BCE Loss ---
        # We dynamically sample negatives to maintain a stable class ratio,
        # preventing the majority class from washing out the gradients.
        pos_mask = ground_truth == 1
        neg_mask = ground_truth == 0

        pos_indices = pos_mask.nonzero(as_tuple=True)[0]
        neg_indices = neg_mask.nonzero(as_tuple=True)[0]

        num_pos = len(pos_indices)
        num_neg_to_sample = int(num_pos * neg_sample_ratio)

        if num_neg_to_sample > 0 and len(neg_indices) > 0:
            # Ensure we don't sample more negatives than exist
            num_neg_to_sample = min(num_neg_to_sample, len(neg_indices))

            # Randomly select a subset of negative indices for this specific epoch
            perm = torch.randperm(len(neg_indices), device=device)
            sampled_neg_indices = neg_indices[perm[:num_neg_to_sample]]

            loss_indices = torch.cat([pos_indices, sampled_neg_indices])
            sampled_targets = ground_truth[loss_indices]
        else:
            # Fallback if graph is weirdly empty of one class
            loss_indices = torch.arange(len(ground_truth), device=device)
            sampled_targets = ground_truth

        if label_smoothing > 0:
            sampled_targets = sampled_targets * (1 - label_smoothing) + (1 - sampled_targets) * label_smoothing
        bce_loss = criterion(pred[loss_indices], sampled_targets)

        # --- Degree Constraint Penalty ---
        penalty_loss = 0.0
        if degree_penalty_weight > 0:
            # This is a more intelligent degree loss that forces sparsity.
            # Instead of summing all incident probabilities, we sum only the top-k,
            # where k is the true degree of the node. This prevents the model from
            # satisfying the loss with many low-probability edges.
            node_violations = []
            for node_idx in range(data.num_nodes):
                true_deg = data.true_degree[node_idx].item()

                # Find all edges where this node is the source
                incident_edge_mask = (data.edge_index[0] == node_idx)

                if not torch.any(incident_edge_mask):
                    predicted_deg = 0.0
                    violation = torch.tensor((predicted_deg - true_deg)**2, dtype=torch.float, device=device)
                else:
                    incident_probs = pred[incident_edge_mask]

                    if true_deg == 0:
                        # Skip degree 0 nodes. The BCE loss already suppresses pure negatives.
                        # Squaring a sum here causes a quadratic explosion that crushes all probabilities globally.
                        continue
                    else:
                        k = int(min(true_deg, len(incident_probs)))
                        if k > 0:
                            top_k_probs = torch.topk(incident_probs, k).values
                            # The primary violation: the top k edges should sum to k
                            predicted_deg = torch.sum(top_k_probs)

                            # The secondary violation: all other edges should sum to 0
                            if len(incident_probs) > k:
                                # Use the mean to prevent explosion from node density
                                rest_mean = (torch.sum(incident_probs) - predicted_deg) / (len(incident_probs) - k)
                                predicted_deg = predicted_deg - rest_mean

                            violation = (predicted_deg - true_deg)**2
                        else:
                            predicted_deg = 0.0
                            violation = torch.tensor((predicted_deg - true_deg)**2, dtype=torch.float, device=device)

                node_violations.append(violation)

            if node_violations:
                penalty_loss = torch.mean(torch.stack(node_violations))

        # Total loss is the sum of classification loss and the degree penalty
        loss = bce_loss + degree_penalty_weight * penalty_loss
        loss.backward()
        for opt in optimizers:
            opt.step()

        total_loss += loss.item() * data.num_graphs
        pred_labels = (pred > 0.5).float()
        total_correct += (pred_labels == ground_truth).sum().item()
        total_samples += ground_truth.size(0)

        total_bce_loss += bce_loss.item() * data.num_graphs
        penalty_val = penalty_loss.item() if isinstance(penalty_loss, torch.Tensor) else penalty_loss
        total_penalty_loss += penalty_val * data.num_graphs

        with torch.no_grad():
            unsampled_bce = criterion(pred, ground_truth).item()
            total_unsampled_bce += unsampled_bce * data.num_graphs
            total_pred_mean += pred.mean().item() * data.num_graphs
            total_pred_std += pred.std(unbiased=False).item() * data.num_graphs

    n = len(loader.dataset)
    return (
        total_loss / n, total_correct / total_samples,
        total_bce_loss / n, total_penalty_loss / n,
        total_unsampled_bce / n, total_pred_mean / n, total_pred_std / n,
    )


def test_model(model, loader, criterion):
    """
    Evaluates the model on a test set for an edge classification task.

    Args:
        model (torch.nn.Module): The trained GNN model to evaluate.
        loader (DataLoader): The DataLoader containing the test/validation graphs.
        criterion (callable): The loss function.

    Returns:
        tuple: A tuple containing average loss, accuracy, and ROC AUC score (float, float, float).
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_ground_truths = []

    with torch.no_grad():
        for data in loader:
            data = data.to(next(model.parameters()).device)
            pred = model(data)
            pred = enforce_symmetric_predictions(pred, data.edge_index, data.num_nodes)

            ground_truth = data.edge_label

            loss = criterion(pred, ground_truth)
            total_loss += loss.item() * data.num_graphs

            all_preds.append(pred)
            all_ground_truths.append(ground_truth)

    avg_loss = total_loss / len(loader.dataset)

    all_preds_cat = torch.cat(all_preds)
    pred_mean = all_preds_cat.mean().item()
    pred_std = all_preds_cat.std(unbiased=False).item()

    final_preds_probs = all_preds_cat.cpu().numpy()
    final_truths = torch.cat(all_ground_truths).cpu().numpy()

    # Dynamically find the optimal threshold to maximize F1-score
    try:
        precisions, recalls, thresholds = precision_recall_curve(final_truths, final_preds_probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[min(best_idx, len(thresholds) - 1)]
    except ValueError:
        best_threshold = 0.5

    pred_binary = (final_preds_probs >= best_threshold).astype(float)
    accuracy = np.mean(pred_binary == final_truths)

    f1 = f1_score(final_truths, pred_binary, zero_division=0)

    try:
        auc_score = roc_auc_score(final_truths, final_preds_probs)
        pr_auc = average_precision_score(final_truths, final_preds_probs)
    except ValueError:
        auc_score = float('nan')
        pr_auc = float('nan')

    return avg_loss, accuracy, auc_score, pr_auc, f1, best_threshold, pred_mean, pred_std


def plot_edge_predictions(image, centroids, edge_index, predictions, ground_truths=None, offset_amount=1.5, pred_probs=None, attentions=None, node_potentials=None, node_boxes=None, edge_boxes=None):
    """
    Overlays GNN edge predictions on the original microscopy image.
    Plots forward and reverse edges side-by-side with arrows.

    Optional `node_boxes` (N, 4) and `edge_boxes` (E, 4) in (x1, y1, x2, y2)
    are overlaid as thin rectangles so the RoIAlign windows used by the
    visual branch are visible alongside the predictions.
    """
    height, width = image.shape[:2]
    # Figure inches are fixed (width/100, height/100); render dpi is dropped to
    # 50 to shrink the logged PNG ~4x. Background image is also downsampled 2x
    # and stretched back to the original coordinate frame via imshow extent,
    # so centroid/box coordinates pass through unchanged.
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=50)
    ax = fig.add_axes([0, 0, 1, 1])

    img_arr = np.asarray(image)
    img_arr = img_arr[::2, ::2]
    extent = [0, width, height, 0]

    # Display: 2D → grayscale; (H,W,1) → grayscale; (H,W,2) → composite with
    # DAPI (ch0) in blue and DIC (ch1) in grayscale so both are visible.
    # (H,W,3) is shown as-is.
    if img_arr.ndim == 2 or (img_arr.ndim == 3 and img_arr.shape[2] == 1):
        ax.imshow(np.squeeze(img_arr), cmap='gray', extent=extent)
    elif img_arr.ndim == 3 and img_arr.shape[2] == 2:
        def _norm(c):
            c = c.astype(np.float32)
            lo, hi = np.percentile(c, 1), np.percentile(c, 99)
            return np.clip((c - lo) / max(hi - lo, 1e-6), 0, 1)
        dapi = _norm(img_arr[..., 0])
        dic = _norm(img_arr[..., 1])
        rgb = np.stack([dic, dic, np.clip(dic + dapi, 0, 1)], axis=-1)
        ax.imshow(rgb, cmap='gray', extent=extent)
    else:
        ax.imshow(img_arr, extent=extent)

    # Track which undirected edges have been labeled to avoid duplicate text
    labeled_edges = set()

    # Draw edges
    num_edges = edge_index.shape[1]
    for i in range(num_edges):
        u, v = edge_index[0, i], edge_index[1, i]
        c1 = centroids[u]
        c2 = centroids[v]

        # Coordinates are (row, col) which maps to (y, x) for plotting
        y1, x1 = c1[0], c1[1]
        y2, x2 = c2[0], c2[1]

        # Calculate perpendicular offset to draw bi-directional edges side-by-side
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        if length > 0:
            nx = -dy / length
            ny = dx / length

            ox1 = x1 + nx * offset_amount
            oy1 = y1 + ny * offset_amount
            ox2 = x2 + nx * offset_amount
            oy2 = y2 + ny * offset_amount
        else:
            ox1, oy1, ox2, oy2 = x1, y1, x2, y2

        pred = predictions[i]

        if ground_truths is not None:
            gt = ground_truths[i]
            if pred == 1 and gt == 1:
                color, alpha, ls, lw = 'lime', 0.9, '-', 2.5
            elif pred == 1 and gt == 0:
                color, alpha, ls, lw = 'red', 0.9, '-', 2.5
            elif pred == 0 and gt == 1:
                color, alpha, ls, lw = 'orange', 0.9, '--', 2
            else:
                color, alpha, ls, lw = 'white', 0.3, ':', 1
        else:
            if pred == 1:
                color, alpha, ls, lw = 'lime', 0.9, '-', 2.5
            else:
                color, alpha, ls, lw = 'red', 0.3, ':', 1

        arrow = patches.FancyArrowPatch(
            (ox1, oy1), (ox2, oy2),
            connectionstyle="arc3,rad=0",
            arrowstyle="->,head_length=8,head_width=5",
            color=color,
            alpha=alpha,
            linewidth=lw,
            linestyle=ls,
            zorder=1
        )
        ax.add_patch(arrow)

        if pred_probs is not None:
            edge_pair = (min(u, v), max(u, v))
            if edge_pair not in labeled_edges:
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2

                label_text = f"P: {pred_probs[i]:.2f}"
                if attentions is not None:
                    a1_probs, a2_probs = attentions
                    label_text += f"\nA1: {a1_probs[i]:.2f} | A2: {a2_probs[i]:.2f}"

                ax.text(mx, my, label_text, color='white', fontsize=7,
                        ha='center', va='center', zorder=3,
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=0.3))
                labeled_edges.add(edge_pair)

    # Overlay RoIAlign windows so it's visible which image regions the visual branch sees.
    # Edge boxes are drawn once per undirected pair (same dedup trick as the probability labels).
    if edge_boxes is not None:
        seen_edges = set()
        for i in range(num_edges):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            pair = (min(u, v), max(u, v))
            if pair in seen_edges:
                continue
            seen_edges.add(pair)
            x1, y1, x2, y2 = edge_boxes[i]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.2, edgecolor='yellow', facecolor='none',
                linestyle=':', alpha=0.5, zorder=1,
            )
            ax.add_patch(rect)

    if node_boxes is not None:
        for i in range(len(node_boxes)):
            x1, y1, x2, y2 = node_boxes[i]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.2, edgecolor='cyan', facecolor='none',
                linestyle='--', alpha=0.5, zorder=1,
            )
            ax.add_patch(rect)

    # Plot node centroids
    if len(centroids) > 0:
        ys, xs = zip(*centroids)
        ax.plot(xs, ys, 'o', color='cyan', markersize=4, alpha=0.6, markeredgecolor='black', zorder=2)

        if node_potentials is not None:
            for idx, (y, x) in enumerate(centroids):
                ax.text(x, y, f"{node_potentials[idx]:.2f}", color='yellow', fontsize=6,
                        ha='center', va='bottom', zorder=4,
                        bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=0.1))

    if ground_truths is not None:
        legend_elements = [
            Line2D([0], [0], color='lime', lw=2.5, label='True Positive'),
            Line2D([0], [0], color='red', lw=2.5, label='False Positive'),
            Line2D([0], [0], color='orange', lw=2, linestyle='--', label='False Negative'),
            Line2D([0], [0], color='blue', lw=1, linestyle=':', alpha=0.5, label='True Negative')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    ax.axis('off')
    return fig




def _collect_train_embeddings(model, train_dataset, device):
    """Collect and concatenate pre-logit embeddings and true labels for all training graphs.

    Returns:
        train_emb:    (E_tr, D) float32 ndarray of concatenated embeddings.
        train_labels: (E_tr,)   float32 ndarray of concatenated true labels.
    """
    emb_list, label_list = [], []
    for data in train_dataset:
        emb, _ = collect_embeddings(model, data, device)
        emb_list.append(emb)
        label_list.append(data.edge_label.cpu().numpy().astype(np.float32))
    return np.vstack(emb_list), np.concatenate(label_list)


def _log_figures(model, test_dataset, test_idx, writer, final_test_thresh, device,
                 train_dataset=None):
    """Render prediction overlays and interpretation figures for each test graph.

    A single forward pass per graph is shared by both outputs so that the
    edge classifications in the prediction overlay and interpretation plots
    are guaranteed to be identical.

    If train_dataset is provided, training embeddings are overlaid on the PCA
    and PLS-DA scatter plots as open dashed circles colored by true label
    (positive → TP-green, negative → TN-blue).
    """
    if not test_dataset:
        return

    has_pred_figs = (
        hasattr(test_dataset[0], 'image') and hasattr(test_dataset[0], 'centroids')
    )

    model.eval()

    # Collect training embeddings once, reused for every test graph in this fold
    train_emb, train_labels = None, None
    if train_dataset:
        try:
            train_emb, train_labels = _collect_train_embeddings(model, train_dataset, device)
        except Exception as exc:
            print(f"[warn] Could not collect training embeddings: {exc}")

    for i, orig_idx in enumerate(test_idx):
        data = test_dataset[i]

        # ── Single forward pass (shared by prediction overlay and interpretation) ──
        with torch.no_grad():
            raw_pred, emb, layer_attentions = model(
                data, return_embeddings=True, return_attention=True
            )
            alpha1 = enforce_symmetric_predictions(
                layer_attentions[0].squeeze(-1), data.edge_index, data.num_nodes
            )
            alpha2 = enforce_symmetric_predictions(
                layer_attentions[1].squeeze(-1), data.edge_index, data.num_nodes
            )
            sym_pred    = enforce_symmetric_predictions(raw_pred, data.edge_index, data.num_nodes)
            sym_probs   = sym_pred.cpu().numpy()
            pred_labels = (sym_pred >= final_test_thresh).float().cpu().numpy()
            attn_np     = (alpha1.cpu().numpy(), alpha2.cpu().numpy())
            emb_np      = emb.cpu().numpy().astype(np.float32)

        true_labels  = data.edge_label.cpu().numpy()
        edge_classes = classify_edges(sym_probs, true_labels, final_test_thresh)

        # ── Prediction overlay ─────────────────────────────────────────────
        if has_pred_figs:
            img       = data.image
            centroids = data.centroids
            centroids_for_boxes = (
                centroids if torch.is_tensor(centroids)
                else torch.as_tensor(centroids, dtype=torch.float32)
            )
            if torch.is_tensor(centroids):
                centroids = centroids.cpu().numpy()

            node_boxes_np = None
            edge_boxes_np = None
            if getattr(model, 'use_visual_features', False):
                node_boxes_t = _node_boxes(centroids_for_boxes, model.node_box_size)
                edge_boxes_t = _edge_boxes(
                    centroids_for_boxes, data.edge_index.cpu(),
                    model.edge_box_margin_frac, model.edge_box_margin_floor,
                )
                node_boxes_np = node_boxes_t.cpu().numpy()
                edge_boxes_np = edge_boxes_t.cpu().numpy()

            fig = plot_edge_predictions(
                img, centroids, data.edge_index.cpu().numpy(), pred_labels, true_labels,
                pred_probs=sym_probs, attentions=attn_np,
                node_boxes=node_boxes_np, edge_boxes=edge_boxes_np,
            )
            writer.add_figure(f'Predictions/Graph_{orig_idx}', fig, 0)
            plt.close(fig)

        # ── Interpretation figures ─────────────────────────────────────────
        try:
            attr_matrix, feature_names, groups = compute_per_edge_attributions(
                model, data, device
            )
            fig = plot_combined_figure(
                emb_np, true_labels, edge_classes,
                attr_matrix, sym_probs, feature_names, groups,
                train_embeddings=train_emb,
                train_true_labels=train_labels,
            )
            writer.add_figure(f'Interpretation/Graph_{orig_idx}', fig, 0)
            plt.close(fig)
        except Exception as exc:
            import traceback
            print(f"[warn] Interpretation failed for graph {orig_idx}: {exc}")
            traceback.print_exc()
            try:
                fig_fb = plot_pca_figure(
                    emb_np, edge_classes,
                    train_embeddings=train_emb, train_true_labels=train_labels,
                )
                writer.add_figure(f'Interpretation/Graph_{orig_idx}', fig_fb, 0)
                plt.close(fig_fb)
            except Exception:
                pass


def _apply_feature_zscore(train_dataset, test_dataset=None):
    """Z-score node features + edge intensity (col 0) using train-fold stats only."""
    if not train_dataset:
        return

    all_x = torch.cat([data.x for data in train_dataset], dim=0)
    all_edge_attr = torch.cat([data.edge_attr for data in train_dataset], dim=0)
    x_mean, x_std = all_x.mean(dim=0), all_x.std(dim=0)
    edge_mean, edge_std = all_edge_attr.mean(dim=0), all_edge_attr.std(dim=0)

    datasets = [train_dataset] + ([test_dataset] if test_dataset is not None else [])
    for ds in datasets:
        for data in ds:
            data.x = (data.x - x_mean) / (x_std + 1e-7)
            # Only normalize edge intensity (col 0). Length (col 1) is biologically normalized,
            # angle columns are pre-scaled to [0, 1] — Z-scoring would destroy those.
            data.edge_attr[:, 0] = (data.edge_attr[:, 0] - edge_mean[0]) / (edge_std[0] + 1e-7)


def n_fold_validation(dataset, num_folds, max_epochs, batch_size, learning_rate, model_params, experiment=None, patience=10, degree_penalty_weight=0.0, neg_sample_ratio=1.0, min_epoch=0, label_smoothing=0.0, log_train_embeddings=True):
    """
    Performs N-fold cross-validation and tracks results, preventing data leakage
    by dynamically applying Z-score normalization per fold.
    """
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    results = []

    print(f"Starting {num_folds}-fold cross-validation...")

    if experiment is None:
        experiment = datetime.now().strftime('%Y%m%d_%H%M%S')

    root_experiment = experiment.split('_')[0]
    log_dir = f'output/cv_experiment/{root_experiment}/{experiment}'

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"\n----- Fold {fold+1}/{num_folds} -----")

        train_dataset = [copy.deepcopy(dataset[i]) for i in train_idx]
        test_dataset = [copy.deepcopy(dataset[i]) for i in test_idx]

        _apply_feature_zscore(train_dataset, test_dataset)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_dataset = [d.to(device) for d in train_dataset]
        test_dataset = [d.to(device) for d in test_dataset]

        train_loader = create_data_loader(train_dataset, batch_size, shuffle=True)
        test_loader = create_data_loader(test_dataset, batch_size, shuffle=False)

        model = Model(**model_params).to(device)

        # Dummy forward to initialize Lazy modules on the correct device
        dummy_data = next(iter(train_loader)).to(device)
        with torch.no_grad():
            model(dummy_data)

        optimizers = get_muon_optimizers(model, learning_rate)
        criterion = torch.nn.BCELoss()

        writer = SummaryWriter(log_dir=f'{log_dir}/fold_{fold+1}')

        best_val_auc = -1.0
        best_epoch = 0
        epochs_no_improve = 0
        best_model_state = None

        epoch_pbar = tqdm(range(1, max_epochs + 1), desc=f"Training Fold {fold+1}/{num_folds}")
        for epoch in epoch_pbar:
            train_loss, train_acc, train_bce, train_penalty, train_bce_unsampled, train_pred_mean, train_pred_std = train_model(model, train_loader, optimizers, criterion, degree_penalty_weight, neg_sample_ratio, label_smoothing)
            test_loss, test_acc, test_auc, test_pr_auc, test_f1, _, test_pred_mean, test_pred_std = test_model(model, test_loader, criterion)

            if epoch >= min_epoch:
                if test_auc > best_val_auc:
                    best_val_auc = test_auc
                    best_epoch = epoch
                    epochs_no_improve = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    epochs_no_improve += 1

            epoch_pbar.set_postfix({
                'Loss (Train)': f"{train_loss:.4f}", 'Acc (Train)': f"{train_acc:.4f}",
                'AUC': f"{test_auc:.4f}", 'PR_AUC': f"{test_pr_auc:.4f}", 'F1': f"{test_f1:.4f}"
            })

            writer.add_scalar('Loss/Train_Total', train_loss, epoch)
            writer.add_scalar('Loss/Train_BCE', train_bce, epoch)
            writer.add_scalar('Loss/Train_DegreePenalty', train_penalty, epoch)
            writer.add_scalar('Loss/Train_BCE_Unsampled', train_bce_unsampled, epoch)
            writer.add_scalar('Diag/Pred_Mean', train_pred_mean, epoch)
            writer.add_scalar('Diag/Pred_Std', train_pred_std, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Loss/Test', test_loss, epoch)
            writer.add_scalar('Accuracy/Test', test_acc, epoch)
            writer.add_scalar('AUC/Test', test_auc, epoch)
            writer.add_scalar('PR_AUC/Test', test_pr_auc, epoch)
            writer.add_scalar('F1/Test', test_f1, epoch)
            writer.add_scalar('Diag/Pred_Mean_Test', test_pred_mean, epoch)
            writer.add_scalar('Diag/Pred_Std_Test', test_pred_std, epoch)

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        final_test_loss, final_test_acc, final_test_auc, final_pr_auc, final_f1, final_test_thresh, _, _ = test_model(model, test_loader, criterion)
        print(f"Fold {fold+1} Final -> Loss: {final_test_loss:.4f}, AUC: {final_test_auc:.4f}, PR_AUC: {final_pr_auc:.4f}, F1: {final_f1:.4f}, Thresh: {final_test_thresh:.4f}")

        writer.add_scalar('EarlyStopping/Best_Epoch', best_epoch, 0)
        writer.add_scalar('EarlyStopping/Best_AUC', best_val_auc, 0)

        summary_text = f"Train {train_idx.tolist()} Test {test_idx.tolist()} Best Epoch {best_epoch} Loss {final_test_loss:.4f} AUC {final_test_auc:.4f} PR_AUC {final_pr_auc:.4f} F1 {final_f1:.4f} Thresh {final_test_thresh:.4f}"
        writer.add_text('Fold Summary', summary_text, 0)

        _log_figures(
            model, test_dataset, test_idx, writer, final_test_thresh, device,
            train_dataset=train_dataset if log_train_embeddings else None,
        )

        results.append({'test_loss': final_test_loss, 'test_acc': final_test_acc, 'test_auc': final_test_auc, 'best_threshold': final_test_thresh})
        writer.close()

    avg_loss = np.mean([r['test_loss'] for r in results])
    avg_acc = np.mean([r['test_acc'] for r in results])
    avg_auc = np.nanmean([r['test_auc'] for r in results])
    avg_thresh = np.nanmean([r['best_threshold'] for r in results])
    print("\n----- Cross-Validation Summary -----")
    print(f"Average Test Loss: {avg_loss:.4f}")
    print(f"Average Test Acc: {avg_acc:.4f}")
    print(f"Average Test AUC: {avg_auc:.4f}")
    print(f"Average Best Threshold: {avg_thresh:.4f}")

    return results


def train_overfit_test(dataset, max_epochs, batch_size, learning_rate, model_params, experiment=None, patience=10, degree_penalty_weight=0.0, neg_sample_ratio=1.0, min_epoch=0, label_smoothing=0.0):
    """
    Trains the model on the entire dataset to test if it has the capacity to overfit.
    It evaluates the model on the exact same dataset it trains on.
    """
    print("Starting overfit test on the entire dataset...")

    if experiment is None:
        experiment = datetime.now().strftime('%Y%m%d_%H%M%S')

    root_experiment = experiment.split('_')[0]
    log_dir = f'output/overfit_experiment/{root_experiment}/{experiment}'

    train_dataset = [copy.deepcopy(data) for data in dataset]
    _apply_feature_zscore(train_dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = [d.to(device) for d in train_dataset]

    train_loader = create_data_loader(train_dataset, batch_size, shuffle=True)
    eval_loader = create_data_loader(train_dataset, batch_size, shuffle=False)

    model = Model(**model_params).to(device)

    dummy_data = next(iter(train_loader)).to(device)
    with torch.no_grad():
        model(dummy_data)

    optimizers = get_muon_optimizers(model, learning_rate)
    criterion = torch.nn.BCELoss()

    writer = SummaryWriter(log_dir=log_dir)

    best_val_auc = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state = None

    epoch_pbar = tqdm(range(1, max_epochs + 1), desc="Training Overfit Model")
    for epoch in epoch_pbar:
        train_loss, train_acc, train_bce, train_penalty, train_bce_unsampled, train_pred_mean, train_pred_std = train_model(model, train_loader, optimizers, criterion, degree_penalty_weight, neg_sample_ratio, label_smoothing)
        test_loss, test_acc, test_auc, test_pr_auc, test_f1, _, test_pred_mean, test_pred_std = test_model(model, eval_loader, criterion)

        if epoch >= min_epoch:
            if test_auc > best_val_auc:
                best_val_auc = test_auc
                best_epoch = epoch
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1

        epoch_pbar.set_postfix({
            'Loss (Train)': f"{train_loss:.4f}", 'Acc (Train)': f"{train_acc:.4f}",
            'AUC': f"{test_auc:.4f}", 'PR_AUC': f"{test_pr_auc:.4f}", 'F1': f"{test_f1:.4f}"
        })

        writer.add_scalar('Loss/Train_Total', train_loss, epoch)
        writer.add_scalar('Loss/Train_BCE', train_bce, epoch)
        writer.add_scalar('Loss/Train_DegreePenalty', train_penalty, epoch)
        writer.add_scalar('Loss/Train_BCE_Unsampled', train_bce_unsampled, epoch)
        writer.add_scalar('Diag/Pred_Mean', train_pred_mean, epoch)
        writer.add_scalar('Diag/Pred_Std', train_pred_std, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Loss/Eval', test_loss, epoch)
        writer.add_scalar('Accuracy/Eval', test_acc, epoch)
        writer.add_scalar('AUC/Eval', test_auc, epoch)
        writer.add_scalar('PR_AUC/Eval', test_pr_auc, epoch)
        writer.add_scalar('F1/Eval', test_f1, epoch)
        writer.add_scalar('Diag/Pred_Mean_Eval', test_pred_mean, epoch)
        writer.add_scalar('Diag/Pred_Std_Eval', test_pred_std, epoch)

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_test_loss, final_test_acc, final_test_auc, final_pr_auc, final_f1, final_test_thresh, _, _ = test_model(model, eval_loader, criterion)
    print(f"Final Overfit Results -> Loss: {final_test_loss:.4f}, AUC: {final_test_auc:.4f}, PR_AUC: {final_pr_auc:.4f}, F1: {final_f1:.4f}, Thresh: {final_test_thresh:.4f}")

    writer.add_scalar('EarlyStopping/Best_Epoch', best_epoch, 0)
    writer.add_scalar('EarlyStopping/Best_AUC', best_val_auc, 0)

    summary_text = f"Best Epoch {best_epoch} Final Loss {final_test_loss:.4f} Final AUC {final_test_auc:.4f}, PR_AUC {final_pr_auc:.4f}, F1 {final_f1:.4f} Thresh: {final_test_thresh:.4f}"
    writer.add_text('Overfit Test Summary', summary_text, 0)

    writer.close()

    return {'test_loss': final_test_loss, 'test_acc': final_test_acc, 'test_auc': final_test_auc, 'best_threshold': final_test_thresh}