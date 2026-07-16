"""Assemble a single PyG graph from an AIS label map (and optional GT + embeddings).

Glue over extract_cell_graph (nodes/edges/features), cell_merge_labels (MST training
labels from GT), and create_pyg_data (PyG assembly). Collect a list of the outputs and
pass it to save_pyg_dataset to build a DapiTracingDataset.
"""
import numpy as np

from image_processing_tools.scene_graph_network.cell_mask_graph import extract_cell_graph
from image_processing_tools.scene_graph_network.cell_merge_labels import cell_merge_labels
from image_processing_tools.scene_graph_network.gnn_data import create_pyg_data


def build_cell_graph_data(ais_labels, intensity_image, gt_labels=None,
                          microsam_npz_path=None, display_image=None,
                          min_overlap_frac=0.5, **extract_kwargs):
    """Build one PyG graph from an AIS label map.

    Args:
        ais_labels: AIS instance label map (H, W); the node set.
        intensity_image: single 2D channel driving the node/edge intensity
            features. Build it with `ImageContainer([[*channel_paths]], config).merge()`
            so the channels are summed and stretched to full contrast; a channel
            stack is rejected by `extract_cell_graph`.
        gt_labels: optional GT whole-cell label map; when given, per-cell MST merge
            labels are attached. Without it every candidate edge is labeled 0.
        microsam_npz_path: optional path to a stitched SAM feature `.npz` for the
            visual branch.
        display_image: optional image stored as `data.image` for the prediction
            overlays in `gnn_train._log_figures` (2D grayscale or (H, W, 3) RGB).
        min_overlap_frac: GT-overlap threshold for fragment assignment.
        **extract_kwargs: forwarded to `extract_cell_graph` (e.g. k, contact_tau).
    """
    node_df, centroids, node_bboxes, edge_df, edge_index = extract_cell_graph(
        ais_labels, intensity_image, **extract_kwargs
    )
    if gt_labels is not None:
        true_edges = cell_merge_labels(ais_labels, gt_labels, min_overlap_frac)
    else:
        true_edges = []                          # empty compact list -> all-negative labels

    # Node index -> AIS label, needed to paint predicted merges back onto pixels.
    # regionprops (which drives node order in extract_cell_graph) yields regions in
    # ascending label order, so np.unique reproduces that order without the cost.
    fragment_labels = np.unique(ais_labels)
    fragment_labels = fragment_labels[fragment_labels != 0]

    data_list = create_pyg_data(
        edge_indices=[edge_index],
        nuclei_features_list=[node_df],
        path_features_list=[edge_df],
        edge_labels_list=[true_edges],
        images_list=None if display_image is None else [display_image],
        centroids_list=[centroids],
        node_bboxes_list=[node_bboxes],
        microsam_paths_list=None if microsam_npz_path is None else [microsam_npz_path],
        ais_labels_list=[ais_labels],
        gt_labels_list=None if gt_labels is None else [gt_labels],
        fragment_labels_list=[fragment_labels],
    )
    return data_list[0]