import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
import torch
from image_processing_tools.scene_graph_network.gnn_data import create_pyg_data


def _tiny_graph():
    node_df = pd.DataFrame({"node_id": [0, 1], "f0": [0.1, 0.2], "f1": [0.3, 0.4]})
    edge_df = pd.DataFrame({"source_node": [0], "target_node": [1], "e0": [0.5], "e1": [0.6]})
    return node_df, edge_df


def test_node_bboxes_attached_and_shaped():
    node_df, edge_df = _tiny_graph()
    bboxes = [np.array([[0, 0, 5, 5], [6, 6, 10, 10]], dtype=np.float32)]
    data_list = create_pyg_data(
        edge_indices=[[[0], [1]]],
        nuclei_features_list=[node_df],
        path_features_list=[edge_df],
        edge_labels_list=[[(0, 1)]],
        node_bboxes_list=bboxes,
    )
    d = data_list[0]
    assert hasattr(d, "node_bboxes")
    assert d.node_bboxes.shape == (2, 4)
    assert d.node_bboxes.dtype == torch.float32


def test_node_bboxes_optional_backward_compatible():
    node_df, edge_df = _tiny_graph()
    data_list = create_pyg_data(
        edge_indices=[[[0], [1]]],
        nuclei_features_list=[node_df],
        path_features_list=[edge_df],
        edge_labels_list=[[(0, 1)]],
    )
    assert not hasattr(data_list[0], "node_bboxes")