import numpy as np
import torch
from torch_geometric.data import Data

from image_processing_tools.scene_graph_network.gnn_train import evaluate_node_types
from image_processing_tools.scene_graph_network.simple_gnn import Model


def _graph(n=12, seed=0, with_types=True):
    g = torch.Generator().manual_seed(seed)
    e = 30
    d = Data(
        x=torch.rand(n, 8, generator=g),
        edge_index=torch.randint(0, n, (2, e), generator=g),
        edge_attr=torch.rand(e, 10, generator=g),
        edge_label=(torch.rand(e, generator=g) > 0.5).float(),
    )
    if with_types:
        d.node_type = torch.randint(0, 3, (n,), generator=g)
    return d


def test_returns_metrics_for_a_typed_dataset():
    model = Model(hidden_channels=16, dropout_p=0.0, predict_node_type=True)
    device = torch.device("cpu")
    m = evaluate_node_types(model, [_graph()], device)
    assert m is not None
    assert "accuracy" in m and "per_class" in m
    assert 0.0 <= m["accuracy"] <= 1.0


def test_returns_none_without_node_types():
    model = Model(hidden_channels=16, dropout_p=0.0, predict_node_type=True)
    m = evaluate_node_types(model, [_graph(with_types=False)], torch.device("cpu"))
    assert m is None


def test_returns_none_without_the_head():
    model = Model(hidden_channels=16, dropout_p=0.0)   # predict_node_type=False
    m = evaluate_node_types(model, [_graph()], torch.device("cpu"))
    assert m is None


def test_only_reports_classes_present_in_the_dataset():
    d = _graph()
    d.node_type = torch.tensor([0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2])   # no epithelial
    model = Model(hidden_channels=16, dropout_p=0.0, predict_node_type=True)
    m = evaluate_node_types(model, [d], torch.device("cpu"))
    assert "epithelial" not in m["per_class"]
    assert set(m["present"]) == {"background", "hyphal"}
