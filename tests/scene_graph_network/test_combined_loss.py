import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from image_processing_tools.scene_graph_network.gnn_train import train_model
from image_processing_tools.scene_graph_network.simple_gnn import Model


def _graph(n=12, seed=0):
    """A graph with paired forward/reverse edges, as build_cell_graph_data produces.

    train_model calls enforce_symmetric_predictions, which requires every edge to have its
    reverse present -- it sorts by an undirected hash and pairs [0::2] with [1::2]. A raw
    torch.randint edge_index has unpaired edges and crashes it, which has nothing to do with
    the node loss under test. Real graphs always carry both directions.
    """
    g = torch.Generator().manual_seed(seed)
    pairs = sorted({(min(u, v), max(u, v))
                    for u, v in torch.randint(0, n, (60, 2), generator=g).tolist()
                    if u != v})[:15]
    src = [u for u, v in pairs] + [v for u, v in pairs]
    dst = [v for u, v in pairs] + [u for u, v in pairs]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    lbl = torch.zeros(len(pairs))
    lbl[:5] = 1.0                                  # both classes present for neg sampling
    return Data(
        x=torch.rand(n, 8, generator=g),
        edge_index=edge_index,
        edge_attr=torch.rand(edge_index.shape[1], 10, generator=g),
        edge_label=torch.cat([lbl, lbl]),          # symmetric, as the real labels are
        node_type=torch.tensor([0] * 3 + [1] * 4 + [2] * 5, dtype=torch.long),
    )


def _fit(model, **kw):
    loader = DataLoader([_graph()], batch_size=1)
    opts = [torch.optim.AdamW(model.parameters(), lr=1e-3)]
    return train_model(model, loader, opts, torch.nn.BCELoss(), **kw)


def test_node_loss_is_reported_and_nonzero_when_enabled():
    model = Model(hidden_channels=16, dropout_p=0.0, predict_node_type=True)
    out = _fit(model, node_loss_weight=1.0)
    assert len(out) == 8
    node_loss = out[7]
    assert node_loss > 0.0


def test_node_loss_is_zero_when_disabled():
    model = Model(hidden_channels=16, dropout_p=0.0)
    out = _fit(model)
    assert len(out) == 8
    assert out[7] == 0.0


def test_graph_without_node_type_still_trains():
    """Existing datasets have no node_type; enabling the weight must not crash on them."""
    model = Model(hidden_channels=16, dropout_p=0.0, predict_node_type=True)
    data = _graph()
    del data.node_type
    loader = DataLoader([data], batch_size=1)
    opts = [torch.optim.AdamW(model.parameters(), lr=1e-3)]
    out = train_model(model, loader, opts, torch.nn.BCELoss(), node_loss_weight=1.0)
    assert out[7] == 0.0


def test_node_head_receives_gradient():
    model = Model(hidden_channels=16, dropout_p=0.0, predict_node_type=True)
    loader = DataLoader([_graph()], batch_size=1)
    opts = [torch.optim.AdamW(model.parameters(), lr=1e-3)]
    # Initialise the lazy layers, then snapshot.
    train_model(model, loader, opts, torch.nn.BCELoss(), node_loss_weight=1.0)
    before = model.node_classifier.mlp_head.weight.detach().clone()
    train_model(model, loader, opts, torch.nn.BCELoss(), node_loss_weight=1.0)
    assert not torch.allclose(before, model.node_classifier.mlp_head.weight)


def test_node_loss_weight_scales_the_contribution():
    """Total loss must actually include node_loss_weight * node_ce."""
    torch.manual_seed(0)
    m1 = Model(hidden_channels=16, dropout_p=0.0, predict_node_type=True)
    torch.manual_seed(0)
    m2 = Model(hidden_channels=16, dropout_p=0.0, predict_node_type=True)
    torch.manual_seed(0)
    a = _fit(m1, node_loss_weight=0.0)
    torch.manual_seed(0)
    b = _fit(m2, node_loss_weight=1.0)
    assert a[0] != b[0]     # total loss differs


def test_node_loss_weight_without_the_head_raises():
    """A model with no node head would report node_loss 0.0 forever, which is
    indistinguishable in the logs from a converged head. Fail loudly instead.
    """
    model = Model(hidden_channels=16, dropout_p=0.0)   # no predict_node_type
    with pytest.raises(ValueError, match="predict_node_type"):
        _fit(model, node_loss_weight=1.0)
