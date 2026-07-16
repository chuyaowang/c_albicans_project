import torch
import numpy as np
from tqdm.auto import tqdm
import copy
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, average_precision_score
from torch.utils.tensorboard import SummaryWriter

# Import baseline data and utility functions to reuse
from image_processing_tools.dapi_tracing.gnn_data import (
    create_data_loader, get_muon_optimizers, plot_edge_predictions,
    enforce_symmetric_predictions
)
from image_processing_tools.dapi_tracing.sinkhorn_gnn import AcyclicModel

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


def train_model_acyclic(model, loader, optimizers, criterion, degree_penalty_weight=0.0, neg_sample_ratio=1.0):
    """
    Performs one epoch of training for the Acyclic GNN model.
    
    Args:
        model (torch.nn.Module): The Acyclic model to train.
        loader (DataLoader): PyTorch Geometric DataLoader containing the training graphs.
        optimizers (list): List containing optimizers (e.g., Muon, AdamW).
        criterion (callable): Loss function (e.g., BCELoss).
        degree_penalty_weight (float): Weight applied to the degree constraint penalty.
        neg_sample_ratio (float): Ratio of negative edges to sample relative to positive edges.
        
    Returns:
        tuple: Contains (average_total_loss, accuracy, average_bce_loss, average_penalty_loss).
    """
    model.train()
    total_loss, total_correct, total_samples, total_bce_loss, total_penalty_loss = 0, 0, 0, 0, 0
    device = next(model.parameters()).device

    for data in loader:
        data = data.to(device)
        for opt in optimizers:
            opt.zero_grad()

        # 1. Forward Pass (Returns MASKED directed predictions)
        raw_directed_pred = model(data.x, data.edge_index, data.edge_attr, getattr(data, 'batch', None))
        
        # 2. Symmetrize via MAX for BCE and Degree Penalty Evaluation
        sym_pred = enforce_symmetric_max(raw_directed_pred, data.edge_index, data.num_nodes)
        ground_truth = data.edge_label

        # --- Negative Edge Subsampling for BCE Loss ---
        pos_mask = ground_truth == 1
        neg_mask = ground_truth == 0
        
        pos_indices = pos_mask.nonzero(as_tuple=True)[0]
        neg_indices = neg_mask.nonzero(as_tuple=True)[0]
        
        num_neg_to_sample = int(len(pos_indices) * neg_sample_ratio)
        
        if num_neg_to_sample > 0 and len(neg_indices) > 0:
            num_neg_to_sample = min(num_neg_to_sample, len(neg_indices))
            perm = torch.randperm(len(neg_indices), device=device)
            sampled_neg_indices = neg_indices[perm[:num_neg_to_sample]]
            loss_indices = torch.cat([pos_indices, sampled_neg_indices])
            
            # Evaluate loss on the SYMMETRIC max
            bce_loss = criterion(sym_pred[loss_indices], ground_truth[loss_indices])
        else:
            bce_loss = criterion(sym_pred, ground_truth)

        # --- Degree Constraint Penalty (Evaluated on Symmetric Prediction) ---
        penalty_loss = 0.0
        if degree_penalty_weight > 0:
            node_violations = []
            for node_idx in range(data.num_nodes):
                true_deg = data.true_degree[node_idx].item()
                incident_edge_mask = (data.edge_index[0] == node_idx)
                
                if not torch.any(incident_edge_mask):
                    predicted_deg = 0.0
                    violation = torch.tensor((predicted_deg - true_deg)**2, dtype=torch.float, device=device)
                else:
                    incident_probs = sym_pred[incident_edge_mask]
                    
                    if true_deg == 0:
                        continue
                        # predicted_mean = torch.mean(incident_probs)
                        # violation = predicted_mean**2
                    else:
                        k = int(min(true_deg, len(incident_probs)))
                        if k > 0:
                            top_k_probs = torch.topk(incident_probs, k).values
                            predicted_deg = torch.sum(top_k_probs)
                            
                            if len(incident_probs) > k:
                                rest_mean = (torch.sum(incident_probs) - predicted_deg) / (len(incident_probs) - k)
                                predicted_deg = predicted_deg - rest_mean
                            
                            violation = (predicted_deg - true_deg)**2
                        else:
                            violation = torch.tensor(true_deg**2, dtype=torch.float, device=device)
                            
                node_violations.append(violation)
                
            if node_violations:
                penalty_loss = torch.mean(torch.stack(node_violations))

        loss = bce_loss + degree_penalty_weight * penalty_loss
        loss.backward()        
        for opt in optimizers:
            opt.step()

        total_loss += loss.item() * data.num_graphs
        pred_labels = (sym_pred > 0.5).float()
        total_correct += (pred_labels == ground_truth).sum().item()
        total_samples += ground_truth.size(0)
        
        total_bce_loss += bce_loss.item() * data.num_graphs
        penalty_val = penalty_loss.item() if isinstance(penalty_loss, torch.Tensor) else penalty_loss
        total_penalty_loss += penalty_val * data.num_graphs

    return total_loss / len(loader.dataset), total_correct / total_samples, total_bce_loss / len(loader.dataset), total_penalty_loss / len(loader.dataset)


def test_model_acyclic(model, loader, criterion):
    """
    Evaluates the Acyclic model on a test or validation set.
    
    Args:
        model (torch.nn.Module): The trained Acyclic model to evaluate.
        loader (DataLoader): PyTorch Geometric DataLoader containing the test graphs.
        criterion (callable): Loss function for evaluating BCE Loss.
        
    Returns:
        tuple: Contains (average_loss, accuracy, auc_score, pr_auc_score, f1_score, best_threshold).
    """
    model.eval()
    total_loss = 0
    all_preds, all_ground_truths = [], []
    device = next(model.parameters()).device

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            raw_directed_pred = model(data.x, data.edge_index, data.edge_attr, getattr(data, 'batch', None))
            sym_pred = enforce_symmetric_max(raw_directed_pred, data.edge_index, data.num_nodes)
            ground_truth = data.edge_label

            loss = criterion(sym_pred, ground_truth)
            total_loss += loss.item() * data.num_graphs

            all_preds.append(sym_pred)
            all_ground_truths.append(ground_truth)

    avg_loss = total_loss / len(loader.dataset)
    final_preds_probs = torch.cat(all_preds).cpu().numpy()
    final_truths = torch.cat(all_ground_truths).cpu().numpy()
    
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
        auc_score, pr_auc = float('nan'), float('nan')

    return avg_loss, accuracy, auc_score, pr_auc, f1, best_threshold


def n_fold_validation_acyclic(dataset, num_folds, max_epochs, batch_size, learning_rate, model_params, experiment=None, patience=10, degree_penalty_weight=0.0, neg_sample_ratio=1.0):
    """
    Performs N-fold cross-validation specifically adapted for the acyclic ranking architecture.
    
    Args:
        dataset (list): The complete list of PyG Data objects.
        num_folds (int): The number of cross-validation folds.
        max_epochs (int): Maximum number of training epochs per fold.
        batch_size (int): Number of graphs to load per batch.
        learning_rate (float): Base learning rate for the optimizers.
        model_params (dict): Dictionary of hyperparameters to initialize the AcyclicModel.
        experiment (str, optional): Custom name for the experiment for TensorBoard logging.
        patience (int): Number of epochs with no AUC improvement before early stopping.
        degree_penalty_weight (float): Weight for the degree constraint penalty.
        neg_sample_ratio (float): Negative edge sampling ratio for the BCE Loss.
        
    Returns:
        list: A list of dictionaries containing final metrics (loss, auc, threshold) for each fold.
    """
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    results = []
    log_dir = f"output/acyclic_cv/{experiment or datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"\n----- Acyclic Fold {fold+1}/{num_folds} -----")

        train_dataset = [copy.deepcopy(dataset[i]) for i in train_idx]
        test_dataset = [copy.deepcopy(dataset[i]) for i in test_idx]

        if len(train_dataset) > 0:
            all_x = torch.cat([data.x for data in train_dataset], dim=0)
            all_edge_attr = torch.cat([data.edge_attr for data in train_dataset], dim=0)
            x_mean, x_std = all_x.mean(dim=0), all_x.std(dim=0)
            edge_mean, edge_std = all_edge_attr.mean(dim=0), all_edge_attr.std(dim=0)
            
            for d_set in [train_dataset, test_dataset]:
                for data in d_set:
                    data.x = (data.x - x_mean) / (x_std + 1e-7)
                    data.edge_attr[:, :2] = (data.edge_attr[:, :2] - edge_mean[:2]) / (edge_std[:2] + 1e-7)

        train_loader = create_data_loader(train_dataset, batch_size, shuffle=True)
        test_loader = create_data_loader(test_dataset, batch_size, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AcyclicModel(**model_params).to(device)
        optimizers = get_muon_optimizers(model, learning_rate)
        criterion = torch.nn.BCELoss()
        writer = SummaryWriter(log_dir=f'{log_dir}/fold_{fold+1}')

        best_val_auc = -1.0
        epochs_no_improve = 0
        best_model_state = None

        epoch_pbar = tqdm(range(1, max_epochs + 1))
        for epoch in epoch_pbar:
            tr_loss, tr_acc, tr_bce, tr_pen = train_model_acyclic(model, train_loader, optimizers, criterion, degree_penalty_weight, neg_sample_ratio)
            te_loss, te_acc, te_auc, te_pr, te_f1, te_thresh = test_model_acyclic(model, test_loader, criterion)
            
            if te_auc > best_val_auc:
                best_val_auc = te_auc
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1

            epoch_pbar.set_postfix({'Loss': f"{tr_loss:.4f}", 'AUC': f"{te_auc:.4f}", 'F1': f"{te_f1:.4f}"})
            
            # Log metrics to TensorBoard
            writer.add_scalar('Loss/Train_Total', tr_loss, epoch)
            writer.add_scalar('Loss/Train_BCE', tr_bce, epoch)
            writer.add_scalar('Loss/Train_DegreePenalty', tr_pen, epoch)
            writer.add_scalar('Accuracy/Train', tr_acc, epoch)
            writer.add_scalar('Loss/Test', te_loss, epoch)
            writer.add_scalar('Accuracy/Test', te_acc, epoch)
            writer.add_scalar('AUC/Test', te_auc, epoch)
            writer.add_scalar('PR_AUC/Test', te_pr, epoch)
            writer.add_scalar('F1/Test', te_f1, epoch)
            
            if epochs_no_improve >= patience: break

        if best_model_state: model.load_state_dict(best_model_state)
        final_loss, final_acc, final_auc, final_pr, final_f1, final_thresh = test_model_acyclic(model, test_loader, criterion)
        
        print(f"Fold {fold+1} Final -> Loss: {final_loss:.4f}, AUC: {final_auc:.4f}, F1: {final_f1:.4f}, Final Threshold: {final_thresh:.4f}")
        
        summary_text = f"Train {train_idx.tolist()} Test {test_idx.tolist()} Loss {final_loss:.4f} AUC {final_auc:.4f} PR_AUC {final_pr:.4f} F1 {final_f1:.4f} Thresh {final_thresh:.4f}"
        writer.add_text('Fold Summary', summary_text, 0)
        
        # Visualize predictions for the test set and log to TensorBoard
        if len(test_dataset) > 0 and hasattr(test_dataset[0], 'image') and hasattr(test_dataset[0], 'centroids'):
            model.eval()
            with torch.no_grad():
                for i, orig_idx in enumerate(test_idx):
                    data = test_dataset[i].to(device)
                    pred, layer_attentions, node_potentials = model(data.x, data.edge_index, data.edge_attr, getattr(data, 'batch', None), return_attention=True)
                    
                    alpha1, alpha2 = layer_attentions
                    alpha1 = alpha1.squeeze(-1)
                    alpha2 = alpha2.squeeze(-1)
                    
                    sym_pred = enforce_symmetric_max(pred, data.edge_index, data.num_nodes)
                    alpha1 = enforce_symmetric_predictions(alpha1, data.edge_index, data.num_nodes)
                    alpha2 = enforce_symmetric_predictions(alpha2, data.edge_index, data.num_nodes)
                    
                    pred_labels = (sym_pred >= final_thresh).float().cpu().numpy()
                    pred_probs = sym_pred.cpu().numpy()
                    attn_probs = (alpha1.cpu().numpy(), alpha2.cpu().numpy())
                    gt_labels = data.edge_label.cpu().numpy()
                    node_potentials_np = node_potentials.cpu().numpy()
                    e_idx = data.edge_index.cpu().numpy()
                    
                    fig = plot_edge_predictions(data.image, data.centroids, e_idx, pred_labels, gt_labels, pred_probs=pred_probs, attentions=attn_probs, node_potentials=node_potentials_np)
                    writer.add_figure(f'Predictions/Graph_{orig_idx}', fig, 0)
                    
        results.append({'test_loss': final_loss, 'test_auc': final_auc, 'best_threshold': final_thresh})
        writer.close()
        
    # Calculate and print average results
    avg_loss = np.mean([r['test_loss'] for r in results])
    avg_auc = np.nanmean([r['test_auc'] for r in results])
    avg_thresh = np.nanmean([r['best_threshold'] for r in results])
    print("\n----- Cross-Validation Summary -----")
    print(f"Average Test Loss: {avg_loss:.4f}")
    print(f"Average Test AUC: {avg_auc:.4f}")
    print(f"Average Best Threshold: {avg_thresh:.4f}")
    
    return results
