import torch
from torch_geometric.data import Data

from image_processing_tools.scene_graph_network.simple_gnn import Model, NodeClassifier


def _graph(n=6, e=10, nf=8, ef=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    return Data(
        x=torch.rand(n, nf, generator=g),
        edge_index=torch.randint(0, n, (2, e), generator=g),
        edge_attr=torch.rand(e, ef, generator=g),
    )


def test_node_classifier_returns_logits_per_node():
    head = NodeClassifier(hidden_channels=16, num_classes=3, dropout_p=0.0)
    out = head(torch.rand(7, 24))
    assert out.shape == (7, 3)


def test_node_classifier_output_is_not_a_probability_distribution():
    """CrossEntropyLoss applies log-softmax itself; a softmaxed head would double it."""
    head = NodeClassifier(hidden_channels=16, num_classes=3, dropout_p=0.0)
    out = head(torch.rand(50, 24))
    assert not torch.allclose(out.sum(dim=-1), torch.ones(50), atol=1e-3)


def test_model_returns_node_logits_when_asked():
    model = Model(hidden_channels=16, dropout_p=0.0, predict_node_type=True)
    data = _graph()
    model.eval()
    with torch.no_grad():
        edge_out, node_logits = model(data, return_node_logits=True)
    assert edge_out.shape == (data.edge_index.shape[1],)
    assert node_logits.shape == (data.num_nodes, 3)


def test_model_without_the_flag_is_unchanged():
    """Existing callers must see exactly the old single-tensor return."""
    model = Model(hidden_channels=16, dropout_p=0.0)
    data = _graph()
    model.eval()
    with torch.no_grad():
        out = model(data)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (data.edge_index.shape[1],)


def test_node_logits_come_last_alongside_other_returns():
    model = Model(hidden_channels=16, dropout_p=0.0, predict_node_type=True)
    data = _graph()
    model.eval()
    with torch.no_grad():
        edge_out, emb, attns, node_logits = model(
            data, return_embeddings=True, return_attention=True, return_node_logits=True
        )
    assert node_logits.shape == (data.num_nodes, 3)
    assert len(attns) == 2


def test_requesting_node_logits_without_the_head_raises():
    """Silently returning nothing would surface as a confusing unpack error later."""
    model = Model(hidden_channels=16, dropout_p=0.0)   # predict_node_type=False
    data = _graph()
    model.eval()
    try:
        with torch.no_grad():
            model(data, return_node_logits=True)
    except (RuntimeError, ValueError) as exc:
        assert "predict_node_type" in str(exc)
    else:
        raise AssertionError("expected an error naming predict_node_type")
