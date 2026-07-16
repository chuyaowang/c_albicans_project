import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
import torch
from image_processing_tools.scene_graph_network.simple_gnn import Model
from image_processing_tools.scene_graph_network.build_cell_dataset import build_cell_graph_data


def _data():
    ais = np.zeros((20, 60), dtype=np.int32)
    ais[8:12, 2:14] = 1
    ais[8:12, 16:28] = 2
    ais[8:12, 30:42] = 3
    gt = np.zeros((20, 60), dtype=np.int32)
    gt[8:12, 2:42] = 10
    dic = np.random.default_rng(2).random((20, 60)).astype(np.float32)
    return build_cell_graph_data(ais, dic, gt_labels=gt, k=2)


def test_model_forward_runs_at_cell_dims():
    data = _data()
    model = Model(hidden_channels=16, node_feature_dim=8, edge_feature_dim=10,
                  use_visual_features=False)
    model.eval()
    with torch.no_grad():
        out = model(data)
    # one probability per directed candidate edge, in [0, 1]
    assert out.shape[0] == data.edge_index.shape[1]
    assert torch.all(out >= 0) and torch.all(out <= 1)