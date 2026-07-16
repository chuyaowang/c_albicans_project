import os
import os.path as osp
from pathlib import Path

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree, to_undirected
import torch_geometric.transforms as T

# Allowlist numpy's array reconstruction primitives so `InMemoryDataset.load` can
# deserialize `Data` objects that carry numpy attributes (e.g. `data.image`) under
# PyTorch's default `weights_only=True` safe-load path. Without all three, PyTorch
# falls back to the unsafe pickle loader and emits a UserWarning.
torch.serialization.add_safe_globals([
    np._core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
])


def augment_rotate(image):
    """
    Rotates a 2D numpy array image in 4 directions (0, 90, 180, 270 degrees).

    Args:
        image (np.ndarray): The 2D input image.

    Returns:
        list: A list containing 4 rotated numpy arrays.
    """
    return [np.rot90(image, k=i) for i in range(4)]


def _normalize_edge_labels(edge_label_input, edge_index_tensor):
    """Accept either a per-edge 0/1 label list or a list of true-edge (u, v) pairs.

    When the input is a list of length-2 integer tuples (or the empty list), it is
    interpreted as the set of TRUE edges; every candidate edge not in that set is
    labeled 0. Matching is undirected so `(u, v)` and `(v, u)` are equivalent.
    When the input is a flat list of 0/1 values with length matching the number of
    candidate edges, it is used as-is (legacy format).

    This keeps the old notebooks working while letting new ones skip the
    tedious full-length label list for graphs with many candidate edges.
    """
    num_edges = edge_index_tensor.size(1)

    is_pair_format = (
        len(edge_label_input) == 0
        or all(
            isinstance(e, (tuple, list))
            and len(e) == 2
            and all(isinstance(x, (int, np.integer)) for x in e)
            for e in edge_label_input
        )
    )

    if is_pair_format:
        true_pairs = {(int(min(u, v)), int(max(u, v))) for (u, v) in edge_label_input}
        labels = torch.zeros(num_edges, dtype=torch.float32)
        if num_edges > 0:
            u_arr = edge_index_tensor[0].tolist()
            v_arr = edge_index_tensor[1].tolist()
            for i, (u, v) in enumerate(zip(u_arr, v_arr)):
                if (min(u, v), max(u, v)) in true_pairs:
                    labels[i] = 1.0
        return labels

    return torch.tensor(edge_label_input, dtype=torch.float32)


class MicrosamData(Data):
    """Data subclass that knows how to batch per-graph microsam feature maps.

    PyG's default collation assumes all tensor-valued attributes can be concatenated
    along dim 0. `microsam_embedding` is a per-graph 3D tensor of shape
    (256, Hf, Wf), which has no node/edge dimension to cat along. Returning `None`
    from `__cat_dim__` tells PyG to `torch.stack` instead, producing
    (B, 256, Hf, Wf) - the exact layout torchvision's RoIAlign expects.
    This requires all graphs in a batch to share (Hf, Wf); for heterogeneous sizes
    a future version can switch to a list-valued attribute.
    """

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'microsam_embedding':
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'microsam_embedding':
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def create_pyg_data(
    edge_indices,
    nuclei_features_list,
    path_features_list,
    edge_labels_list,
    images_list=None,
    centroids_list=None,
    node_bboxes_list=None,
    microsam_paths_list=None,
    ais_labels_list=None,
    gt_labels_list=None,
    fragment_labels_list=None,
    node_types_list=None,
    max_edge_length_neg=None,
    length_col_idx=1,
):
    """
    Converts lists of raw graph components into a list of PyTorch Geometric Data objects.

    Args:
        edge_indices (list): A list of edge indices (2 x num_edges) for each graph.
        nuclei_features_list (list): A list of node feature arrays (num_nodes x num_node_features).
        path_features_list (list): A list of edge feature arrays (num_edges x num_edge_features).
        edge_labels_list (list): Per-graph edge labels. Each entry can be either:
            - Legacy: a flat 0/1 list of length `num_edges` matching `edge_indices` order.
            - Compact: a list of `(u, v)` tuples naming the TRUE edges; every candidate
              edge not in the list is labeled 0. Matching is undirected. Preferred for
              large graphs where listing every candidate edge is tedious.
        images_list (list, optional): List of original numpy images corresponding to the dataset.
        centroids_list (list, optional): List of node centroids [(row, col), ...] corresponding to the dataset.
            Stored as a (num_nodes, 2) float tensor so PyG can concatenate centroids across a batch.
        microsam_paths_list (list, optional): Per-graph path to a `*_microsam_features.npz` file
            produced by `compute_microsam_features`. When given, attaches
            `data.microsam_embedding` of shape (256, Hf, Wf) and scalar
            `data.pixels_per_feature`, and uses the `MicrosamData` subclass so the
            embedding batches correctly through PyG's DataLoader.
        ais_labels_list (list, optional): Per-graph AIS instance label map (H, W).
            Attached as `data.ais_labels` so `_log_figures` can relabel fragments
            into merged cells. Stored as uint16 to halve the memory of an int32 map.
        gt_labels_list (list, optional): Per-graph ground-truth whole-cell label map
            (H, W), attached as `data.gt_labels` for the merge comparison figure.
        fragment_labels_list (list, optional): Per-graph (num_nodes,) array giving
            each node's AIS label, in node order. Attached as `data.fragment_labels`;
            required alongside `ais_labels_list` to map nodes back onto pixels.
        node_types_list (list, optional): Per-graph (num_nodes,) int array of node-type
            class ids in regionprops order, attached as `data.node_type` and consumed by
            the node classification head. Absent means edge-only training.
        max_edge_length_neg (float, optional): If set, NEGATIVE edges whose length
            (column `length_col_idx` of `edge_attr`, already normalized by
            avg_nucleus_length upstream) exceeds this value are dropped to sparsify
            the candidate graph. Positive edges are retained regardless of length.
        length_col_idx (int): Column index of the normalized length feature in edge_attr.
            Default 1 matches the ordering produced by `extract_graph`.

    Returns:
        list: A list of torch_geometric.data.Data (or MicrosamData) objects representing undirected graphs.
    """
    use_microsam = microsam_paths_list is not None
    data_cls = MicrosamData if use_microsam else Data

    pyg_data_list = []
    for i, (edge_index, nuclei_df, path_df, edge_label) in enumerate(zip(edge_indices, nuclei_features_list, path_features_list, edge_labels_list)):

        # 1. Remove the ID columns so only the pure features remain
        node_features = nuclei_df.drop(columns=['node_id'])
        edge_features = path_df.drop(columns=['source_node', 'target_node'])

        # 2. Convert the feature DataFrames into PyTorch tensors (typically float32)
        x = torch.tensor(node_features.values, dtype=torch.float32)
        edge_attr = torch.tensor(edge_features.values, dtype=torch.float32)

        # 3. Convert the edge index list into a PyTorch tensor (typically long/int64)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)

        # 4. Convert the edge labels to a PyTorch tensor. Accepts either the legacy
        # flat 0/1 list or a compact list of true-edge (u, v) pairs (see
        # `_normalize_edge_labels`).
        edge_label_tensor = _normalize_edge_labels(edge_label, edge_index_tensor)

        # 5. Trim overly long negative edges to sparsify the candidate graph.
        # Positive edges are preserved so no ground-truth connections are lost
        # even if a true edge happens to exceed the threshold.
        if max_edge_length_neg is not None and edge_index_tensor.numel() > 0:
            lengths = edge_attr[:, length_col_idx]
            drop_mask = (edge_label_tensor == 0) & (lengths > max_edge_length_neg)
            if drop_mask.any():
                keep_mask = ~drop_mask
                edge_index_tensor = edge_index_tensor[:, keep_mask]
                edge_attr = edge_attr[keep_mask]
                edge_label_tensor = edge_label_tensor[keep_mask]

        # 6. Calculate true node degrees from the undirected positive edges so both
        # endpoints of a hyphal connection are counted.
        # For a connection A--B, undirected is A->B and A<-B, both A and B have a degree of 1
        true_edges_mask = edge_label_tensor == 1
        true_edge_index = edge_index_tensor[:, true_edges_mask]
        num_nodes = node_features.shape[0]
        true_edge_index_undirected = to_undirected(true_edge_index, num_nodes=num_nodes)
        true_degrees = degree(true_edge_index_undirected[0], num_nodes=num_nodes, dtype=torch.float)

        # 7. Create pyg data object
        data = data_cls(
            x=x,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr,
            edge_label=edge_label_tensor,
            true_degree=true_degrees
        )
        data = T.ToUndirected()(data)

        # Attach original image and centroids for visualization later.
        # Centroids are promoted to a (num_nodes, 2) float tensor so PyG can
        # cat them across a batch (needed for RoIAlign in the visual branch).
        if images_list is not None:
            data.image = images_list[i]
        if centroids_list is not None:
            centroids_arr = np.asarray(centroids_list[i], dtype=np.float32)
            data.centroids = torch.from_numpy(centroids_arr)

        if node_bboxes_list is not None:
            bbox_arr = np.asarray(node_bboxes_list[i], dtype=np.float32)
            data.node_bboxes = torch.from_numpy(bbox_arr)

        # Label maps for the merge figure. Kept as numpy (like data.image) rather
        # than tensors: they are never batched, only read at figure time. uint16
        # is ample -- fragment/cell counts are in the hundreds, not the thousands.
        if ais_labels_list is not None:
            data.ais_labels = np.asarray(ais_labels_list[i]).astype(np.uint16)
        if gt_labels_list is not None:
            data.gt_labels = np.asarray(gt_labels_list[i]).astype(np.uint16)
        if fragment_labels_list is not None:
            data.fragment_labels = np.asarray(fragment_labels_list[i]).astype(np.int32)
        if node_types_list is not None:
            # int64: a class index for CrossEntropyLoss, never a float target.
            data.node_type = torch.as_tensor(node_types_list[i], dtype=torch.long)

        if use_microsam:
            npz_path = Path(microsam_paths_list[i])
            with np.load(npz_path) as npz:
                feat_map = npz['feature_map'].astype(np.float32)
                ppf = float(npz['pixels_per_feature'])
            data.microsam_embedding = torch.from_numpy(feat_map)
            # Shape (1,) so PyG cat along dim 0 yields (B,) after batching.
            data.pixels_per_feature = torch.tensor([ppf], dtype=torch.float32)

        pyg_data_list.append(data)

    return pyg_data_list


class DapiTracingDataset(InMemoryDataset):
    """
    Thin InMemoryDataset wrapper around a pre-built list of PyG Data objects.

    Persists the output of `create_pyg_data` to `root/processed/data.pt` so
    subsequent runs skip raw-feature reconstruction. Accepts an optional
    `transform` applied per `__getitem__`, which is the correct slot for
    stochastic training-time augmentations (they re-roll each epoch).
    """

    def __init__(self, root, data_list=None, transform=None):
        self._data_list_to_save = data_list
        super().__init__(root, transform=transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        if self._data_list_to_save is None:
            raise RuntimeError(
                f"No processed file at {self.processed_paths[0]} and no "
                f"data_list was provided to build one. Call save_pyg_dataset "
                f"first, or pass data_list= to this dataset."
            )
        self.save(self._data_list_to_save, self.processed_paths[0])


def save_pyg_dataset(data_list, root):
    """
    Persist a list of PyG Data objects (e.g. output of `create_pyg_data`) to
    disk as a `DapiTracingDataset`. Writes to `root/processed/data.pt`,
    overwriting any existing file so the saved content always matches
    `data_list`.

    Args:
        data_list (list): List of torch_geometric.data.Data objects.
        root (str): Directory the processed dataset will be stored under.

    Returns:
        DapiTracingDataset: The freshly written dataset instance.
    """
    processed_path = osp.join(root, 'processed', 'data.pt')
    if osp.exists(processed_path):
        os.remove(processed_path)
    return DapiTracingDataset(root, data_list=data_list)


def load_pyg_dataset(root, transform=None):
    """
    Load a previously saved `DapiTracingDataset` from disk.

    Args:
        root (str): Directory the dataset was written to via `save_pyg_dataset`.
        transform (callable, optional): Per-sample transform applied on
            `__getitem__`. Intended slot for stochastic augmentations.

    Returns:
        DapiTracingDataset: The loaded dataset.
    """
    return DapiTracingDataset(root, transform=transform)


class StripHeavyAttrs:
    """Per-sample transform that drops heavy attributes before batching.

    The saved dataset keeps `microsam_embedding`, `pixels_per_feature`, and
    `image` so they are available for visual-branch training and for plotting.
    When training without the visual branch, loading them wastes host→device
    copies every batch. Pass this as the dataset `transform` to omit them on
    the fly without mutating the stored `.pt` file.
    """

    DEFAULT_KEYS = ('microsam_embedding', 'pixels_per_feature')

    def __init__(self, keys=None):
        self.keys = tuple(keys) if keys is not None else self.DEFAULT_KEYS

    def __call__(self, data):
        for k in self.keys:
            if k in data:
                del data[k]
        return data


def create_data_loader(dataset, batch_size, shuffle=True):
    """
    Creates a PyTorch Geometric DataLoader for batching graphs.

    Args:
        dataset (list or Dataset): The list of PyG Data objects.
        batch_size (int): The number of graphs per batch.
        shuffle (bool): Whether to shuffle the dataset. Defaults to True.

    Returns:
        DataLoader: The PyTorch Geometric DataLoader instance.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)