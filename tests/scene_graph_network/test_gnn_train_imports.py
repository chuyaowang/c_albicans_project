import inspect
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


def test_gnn_train_imports_from_own_package():
    # importorskip skips gracefully if gnn_train's deps (e.g. torch.optim.Muon) are absent
    gnn_train = pytest.importorskip("image_processing_tools.scene_graph_network.gnn_train")
    src = inspect.getsource(gnn_train)
    assert "image_processing_tools.dapi_tracing" not in src
    assert "image_processing_tools.scene_graph_network.simple_gnn" in src


def test_log_figures_passes_node_bboxes():
    gnn_train = pytest.importorskip("image_processing_tools.scene_graph_network.gnn_train")
    src = inspect.getsource(gnn_train._log_figures)
    # node bboxes must be threaded into both box helpers for the overlay
    assert "node_bboxes=" in src