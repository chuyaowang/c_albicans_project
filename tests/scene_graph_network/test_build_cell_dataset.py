import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from image_processing_tools.scene_graph_network.build_cell_dataset import (
    build_cell_graph_data,
)


def _ais_gt_dic():
    ais = np.zeros((20, 60), dtype=np.int32)
    ais[8:12, 2:14] = 1
    ais[8:12, 16:28] = 2
    ais[8:12, 30:42] = 3
    gt = np.zeros((20, 60), dtype=np.int32)
    gt[8:12, 2:42] = 10                          # all three fragments are one cell
    dic = np.random.default_rng(1).random((20, 60)).astype(np.float32)
    return ais, gt, dic


def test_build_data_has_expected_tensors():
    ais, gt, dic = _ais_gt_dic()
    data = build_cell_graph_data(ais, dic, gt_labels=gt, k=2)
    assert data.x.shape == (3, 8)                # 3 nodes, 8 node features
    assert data.edge_attr.shape[1] == 10         # 10 edge features
    assert data.node_bboxes.shape == (3, 4)
    # MST labels (0-1, 1-2) → at least the positive candidate edges are labeled 1
    assert data.edge_label.sum().item() >= 2


def test_build_data_without_gt_has_no_positive_labels():
    ais, gt, dic = _ais_gt_dic()
    data = build_cell_graph_data(ais, dic, gt_labels=None, k=2)
    assert float(data.edge_label.sum().item()) == 0.0


def test_display_image_attached_when_given():
    ais, gt, dic = _ais_gt_dic()
    disp = np.stack([dic, dic, dic], axis=-1)                 # (H, W, 3) display image
    data = build_cell_graph_data(ais, dic, gt_labels=gt, display_image=disp, k=2)
    assert hasattr(data, "image")
    assert np.asarray(data.image).shape == disp.shape
    # Omitting it leaves no image (prediction overlay stays off)
    data2 = build_cell_graph_data(ais, dic, gt_labels=gt, k=2)
    assert not hasattr(data2, "image")