from torch_geometric.nn import MessagePassing
from torch_geometric.nn.norm import GraphNorm
from torch.nn import (
    LazyLinear, Linear, Sequential, ReLU, LayerNorm, init, Dropout,
    Conv2d, AdaptiveAvgPool2d,
)
import torch
from torchvision.ops import roi_align
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import softmax

class CustomLazyLinear(LazyLinear):
    def reset_parameters(self):
        # Overrides default initialization of lazy linear layer to use Kaiming (He) Normal
        # upon inferring the shape during the first forward pass.
        if not self.has_uninitialized_params() and self.in_features != 0:
            init.kaiming_normal_(self.weight, nonlinearity='relu') # kaiming works well with relu
            if self.bias is not None:
                init.zeros_(self.bias)

class GCNConv(MessagePassing):
    def __init__(self, out_channels, dropout_p=0.5):
        super().__init__(aggr='sum')

        self.mlp = Sequential(
            # Uses lazy initialization to infer the input feature dimension automatically from the node feature matrix
            CustomLazyLinear(out_channels, bias=False),
            LayerNorm(out_channels),
            ReLU(),
            Dropout(p=dropout_p),
            Linear(out_channels, out_channels, bias=True)
        )

        self.update_mlp = Sequential(
            CustomLazyLinear(out_channels, bias=False),
            LayerNorm(out_channels),
            ReLU(),
            Dropout(p=dropout_p),
            Linear(out_channels, out_channels, bias=True)
        )

        # A small linear layer to learn the importance (attention weight) of each edge
        self.attn_mlp = Sequential(
            CustomLazyLinear(1, bias=True)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for mlp in [self.mlp, self.update_mlp, self.attn_mlp]:
            for layer in mlp:
                if isinstance(layer, CustomLazyLinear):
                    # Handled automatically during the first forward pass
                    pass
                elif isinstance(layer, Linear):
                    # Use Xavier (Glorot) initialization for the final linear layer
                    init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        init.zeros_(layer.bias)

    def forward(self, x, edge_index, edge_attr):
        # x: the node feature matrix, has shape [n_nodes, num_node_features]
        # edge_index: the graph connectivity matrix, has shape [2, n_edges]
        # edge_attr: the edge feature matrix, has shape [n_edges, num_edge_features]

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Update the node embeddings
        update_input = torch.cat([x, out], dim=-1)
        node_embeddings = self.update_mlp(update_input)
        return node_embeddings, self._alpha

    def message(self, x_i, x_j, edge_attr, index):
        # x_i: features for the target node
        # x_j: features for the source node
        # edge_attr: edge features
        # index: the indices of the target nodes (automatically passed by PyG)

        features_cat = torch.cat([x_j - x_i, edge_attr], dim=1)
        msg = self.mlp(features_cat)

        # 1. Calculate a raw attention score for each edge
        alpha = self.attn_mlp(features_cat)

        # 2. Normalize the scores across the neighborhood of each target node so they sum to 1.0
        alpha = softmax(alpha, index)

        # Store the normalized attention weights so they can be retrieved after propagation
        self._alpha = alpha

        # 3. Scale the message features by the learned attention percentage
        return msg * alpha

class Classifier(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_p=0.5):
        super().__init__()

        # Split into body (pre-logit embedding) and head (final projection) so callers
        # can optionally extract the intermediate representation before the linear output.
        self.mlp_body = Sequential(
            CustomLazyLinear(hidden_channels, bias=False),
            LayerNorm(hidden_channels),
            ReLU(),
            Dropout(p=dropout_p),
        )
        self.mlp_head = Linear(hidden_channels, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in list(self.mlp_body) + [self.mlp_head]:
            if isinstance(layer, CustomLazyLinear):
                pass  # Handled automatically during the first forward pass
            elif isinstance(layer, Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, x, edge_attr, edge_index, return_embeddings=False):
        # x: feature tensor of the nodes, shape [n_nodes, num_node_features]
        # edge_index: the graph connectivity matrix

        # Get the embeddings for the source and target nodes of each edge
        source_feature = x[edge_index[0]]
        target_feature = x[edge_index[1]]

        # Combine source and target features, predict a single number, and apply sigmoid
        edge_features = torch.cat([source_feature, target_feature, edge_attr], dim=-1)
        emb = self.mlp_body(edge_features)
        out = torch.sigmoid(self.mlp_head(emb).squeeze(-1))
        if return_embeddings:
            return out, emb
        return out

class EdgeUpdater(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_p=0.5):
        super().__init__()

        self.mlp = Sequential(
            CustomLazyLinear(hidden_channels, bias=False),
            LayerNorm(hidden_channels),
            ReLU(),
            Dropout(p=dropout_p),
            # Outputs a new edge embedding of size 'hidden_channels'
            Linear(hidden_channels, hidden_channels, bias=True)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, CustomLazyLinear):
                pass  # Handled automatically during the first forward pass
            elif isinstance(layer, Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, x, edge_index, edge_attr):
        # Get the embeddings for the source and target nodes of each edge
        source_feature = x[edge_index[0]]
        target_feature = x[edge_index[1]]

        # Combine source features, target features, and the previous edge attributes
        edge_inputs = torch.cat([source_feature, target_feature, edge_attr], dim=-1)
        return self.mlp(edge_inputs)


class VisualCNN(torch.nn.Module):
    """Maps a RoI-aligned SAM patch (256, roi, roi) to a D-dim vector."""
    def __init__(self, d_visual, dropout_p=0.0):
        super().__init__()
        self.conv = Sequential(
            Conv2d(256, 64, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(64, 32, kernel_size=3, padding=1),
            ReLU(),
            AdaptiveAvgPool2d(1),
        )
        self.fc = Linear(32, d_visual)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.conv:
            if isinstance(m, Conv2d):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)

    def forward(self, roi_features):
        # roi_features: (K, 256, roi, roi)
        x = self.conv(roi_features)
        x = x.flatten(1)
        return self.fc(x)


class FusionMLP(torch.nn.Module):
    """Fuses tabular features with visual features and projects to hidden_channels."""
    def __init__(self, hidden_channels, dropout_p=0.0):
        super().__init__()
        self.mlp = Sequential(
            CustomLazyLinear(hidden_channels, bias=False),
            LayerNorm(hidden_channels),
            ReLU(),
            Dropout(p=dropout_p),
            Linear(hidden_channels, hidden_channels, bias=True),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, Linear) and not isinstance(layer, CustomLazyLinear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, tabular, visual):
        fused = torch.cat([tabular, visual], dim=-1)
        return self.mlp(fused)


def _node_boxes(centroids, box_size, node_bboxes=None, pad_frac=0.0):
    """Node RoIAlign boxes (x1,y1,x2,y2). Uses padded mask bboxes when given."""
    if node_bboxes is not None:
        if pad_frac > 0:
            w = node_bboxes[:, 2] - node_bboxes[:, 0]
            h = node_bboxes[:, 3] - node_bboxes[:, 1]
            pad = torch.stack([w, h, w, h], dim=1) * pad_frac
            shift = torch.stack([-pad[:, 0], -pad[:, 1], pad[:, 2], pad[:, 3]], dim=1)
            return node_bboxes + shift
        return node_bboxes
    half = box_size / 2.0
    y = centroids[:, 0]
    x = centroids[:, 1]
    return torch.stack([x - half, y - half, x + half, y + half], dim=1)


def _edge_boxes(centroids, edge_index, margin_frac, margin_floor, node_bboxes=None):
    """Edge RoIAlign boxes. Union of endpoint mask bboxes when node_bboxes given,
    else the bbox of the two endpoint centroids padded by a fractional margin + floor."""
    if node_bboxes is not None:
        bi = node_bboxes[edge_index[0]]
        bj = node_bboxes[edge_index[1]]
        x_min = torch.minimum(bi[:, 0], bj[:, 0])
        y_min = torch.minimum(bi[:, 1], bj[:, 1])
        x_max = torch.maximum(bi[:, 2], bj[:, 2])
        y_max = torch.maximum(bi[:, 3], bj[:, 3])
        return torch.stack([x_min, y_min, x_max, y_max], dim=1)
    src = centroids[edge_index[0]]
    tgt = centroids[edge_index[1]]
    ys = torch.stack([src[:, 0], tgt[:, 0]], dim=1)
    xs = torch.stack([src[:, 1], tgt[:, 1]], dim=1)
    y_min, y_max = ys.min(dim=1).values, ys.max(dim=1).values
    x_min, x_max = xs.min(dim=1).values, xs.max(dim=1).values
    width = x_max - x_min
    height = y_max - y_min
    margin = torch.clamp(torch.maximum(width, height) * margin_frac, min=float(margin_floor))
    return torch.stack([x_min - margin, y_min - margin, x_max + margin, y_max + margin], dim=1)


class Model(torch.nn.Module):
    """Full GNN model. Two GCN layers, edge updaters, and a classifier. Optionally
    enriches raw node and edge features with a visual branch that extracts
    MicroSAM embeddings via RoIAlign + a small CNN, then fuses them with the
    tabular features through a per-stream fusion MLP before the first GCN layer.

    Skip connections use the PRE-fusion raw features (`data.x`, `data.edge_attr`)
    to match the behavior of the baseline model. See GCN Design Choices for the
    rationale and alternative skip-source strategies that are still on the table.
    """

    def __init__(
        self,
        hidden_channels,
        dropout_p=0.2,
        use_visual_features=False,
        d_visual=16,
        node_box_size=150,
        edge_box_margin_frac=0.15,
        edge_box_margin_floor=20,
        roi_output_size=7,
        node_bbox_pad_frac=0.1,
        node_feature_dim=8,
        edge_feature_dim=10,
    ):
        super().__init__()
        self.use_visual_features = use_visual_features
        self.d_visual = d_visual
        self.node_box_size = node_box_size
        self.edge_box_margin_frac = edge_box_margin_frac
        self.edge_box_margin_floor = edge_box_margin_floor
        self.roi_output_size = roi_output_size
        self.node_bbox_pad_frac = node_bbox_pad_frac

        if use_visual_features:
            self.node_visual_cnn = VisualCNN(d_visual)
            self.edge_visual_cnn = VisualCNN(d_visual)
            self.node_fusion = FusionMLP(hidden_channels, dropout_p)
            self.edge_fusion = FusionMLP(hidden_channels, dropout_p)

        self.conv1 = GCNConv(hidden_channels, dropout_p=0)  # always input all original features
        self.edge_updater = EdgeUpdater(hidden_channels, dropout_p)
        self.conv2 = GCNConv(hidden_channels, dropout_p)
        self.edge_updater_1 = EdgeUpdater(hidden_channels, dropout_p)
        self.classifier = Classifier(hidden_channels, dropout_p)

        # GraphNorm for residual skip connections: normalizes per graph, no running stats.
        self.norm_x1 = GraphNorm(hidden_channels + node_feature_dim)
        self.norm_e1 = GraphNorm(hidden_channels + edge_feature_dim)
        self.norm_x2 = GraphNorm(hidden_channels + node_feature_dim)
        self.norm_e2 = GraphNorm(hidden_channels + edge_feature_dim)

    def _extract_visual(self, data):
        """Return (node_visual, edge_visual) from data.microsam_embedding via RoIAlign."""
        centroids = data.centroids  # (total_nodes, 2) in (y, x), pixel coords
        edge_index = data.edge_index

        # Per-node graph index. In unbatched mode data.batch is absent.
        batch_vec = data.batch if getattr(data, 'batch', None) is not None \
            else torch.zeros(centroids.size(0), dtype=torch.long, device=centroids.device)
        edge_batch = batch_vec[edge_index[0]]  # same graph as target since edges don't cross graphs

        # Feature map: (B, 256, H, W) when batched, (256, H, W) when unbatched.
        feat = data.microsam_embedding
        if feat.dim() == 3:
            feat = feat.unsqueeze(0)

        # Uniform spatial scale across the batch (assert this holds).
        ppf = data.pixels_per_feature.view(-1)
        if not torch.all(ppf == ppf[0]):
            raise ValueError("pixels_per_feature must be uniform across a batch.")
        spatial_scale = 1.0 / float(ppf[0].item())

        # Build boxes in pixel coordinates; prepend per-box batch index for torchvision RoIAlign.
        node_bboxes = getattr(data, "node_bboxes", None)
        node_xyxy = _node_boxes(centroids, self.node_box_size,
                                node_bboxes=node_bboxes, pad_frac=self.node_bbox_pad_frac)
        edge_xyxy = _edge_boxes(centroids, edge_index, self.edge_box_margin_frac,
                                self.edge_box_margin_floor, node_bboxes=node_bboxes)
        node_rois = torch.cat([batch_vec.float().unsqueeze(1), node_xyxy], dim=1)
        edge_rois = torch.cat([edge_batch.float().unsqueeze(1), edge_xyxy], dim=1)

        node_patches = roi_align(
            feat, node_rois,
            output_size=self.roi_output_size,
            spatial_scale=spatial_scale,
            aligned=True,
        )
        edge_patches = roi_align(
            feat, edge_rois,
            output_size=self.roi_output_size,
            spatial_scale=spatial_scale,
            aligned=True,
        )

        return self.node_visual_cnn(node_patches), self.edge_visual_cnn(edge_patches)

    def forward(self, data, return_attention=False, return_embeddings=False, attribution_mode=False):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # attribution_mode: detach inputs and re-enable grad so callers can run
        # per-edge backward passes to compute gradient × input attributions.
        if attribution_mode:
            x = x.detach().clone().requires_grad_(True)
            edge_attr = edge_attr.detach().clone().requires_grad_(True)

        batch_vec = data.batch if getattr(data, 'batch', None) is not None \
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        edge_batch = batch_vec[edge_index[0]]

        if self.use_visual_features:
            node_visual, edge_visual = self._extract_visual(data)
            if attribution_mode:
                node_visual = node_visual.detach().clone().requires_grad_(True)
                edge_visual = edge_visual.detach().clone().requires_grad_(True)
            x_in = self.node_fusion(x, node_visual)
            edge_in = self.edge_fusion(edge_attr, edge_visual)
        else:
            x_in = x
            edge_in = edge_attr

        # Skip connections use PRE-fusion raw features (see Design Choices).
        x_orig = x
        edge_attr_orig = edge_attr

        # GCN layer 1
        x_out, alpha1 = self.conv1(x_in, edge_index, edge_in)
        x_out = torch.cat([x_out, x_orig], dim=-1)
        x_out = self.norm_x1(x_out, batch_vec)

        # Edge update 1
        edge_out = self.edge_updater(x_out, edge_index, edge_in)
        edge_out = torch.cat([edge_out, edge_attr_orig], dim=-1)
        edge_out = self.norm_e1(edge_out, edge_batch)

        # GCN layer 2
        x_out, alpha2 = self.conv2(x_out, edge_index, edge_out)
        x_out = torch.cat([x_out, x_orig], dim=-1)
        x_out = self.norm_x2(x_out, batch_vec)

        # Edge update 2
        edge_out = self.edge_updater_1(x_out, edge_index, edge_out)
        edge_out = torch.cat([edge_out, edge_attr_orig], dim=-1)
        edge_out = self.norm_e2(edge_out, edge_batch)

        if return_embeddings:
            out, emb = self.classifier(x_out, edge_out, edge_index, return_embeddings=True)
        else:
            out = self.classifier(x_out, edge_out, edge_index)

        if attribution_mode:
            attr_tensors = {'x': x, 'edge_attr': edge_attr}
            if self.use_visual_features:
                attr_tensors['node_visual'] = node_visual
                attr_tensors['edge_visual'] = edge_visual
            if return_embeddings:
                return out, emb, attr_tensors
            return out, attr_tensors

        if return_embeddings and return_attention:
            return out, emb, (alpha1, alpha2)

        if return_embeddings:
            return out, emb

        if return_attention:
            return out, (alpha1, alpha2)

        return out