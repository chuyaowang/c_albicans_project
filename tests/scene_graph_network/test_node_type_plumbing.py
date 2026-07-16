import numpy as np

from image_processing_tools.scene_graph_network.build_cell_dataset import build_cell_graph_data
from image_processing_tools.scene_graph_network.cell_type_labels import NODE_CLASSES


def _scene():
    """Three fragments: two on a filament, one on background."""
    gt = np.zeros((80, 80), dtype=np.int32)
    gt[10:13, 5:70] = 1                       # thin filament
    gt[40:65, 20:45] = 2                      # fat blob

    ais = np.zeros((80, 80), dtype=np.int32)
    ais[10:13, 5:30] = 1                      # filament, left half
    ais[10:13, 35:70] = 2                     # filament, right half
    ais[40:65, 20:45] = 3                     # blob
    ais[70:78, 70:78] = 4                     # background

    rng = np.random.default_rng(0)
    img = rng.integers(0, 65535, size=(80, 80), dtype=np.uint16)
    return ais, gt, img


RULE = {"metric": "mean_width", "threshold": 10.0}


def test_node_type_attached_when_rule_given():
    ais, gt, img = _scene()
    data = build_cell_graph_data(ais, img, gt_labels=gt, k=3,
                                 cell_type_rule=RULE, min_overlap_frac=0.1)
    assert hasattr(data, "node_type")
    assert data.node_type.shape == (data.num_nodes,)
    assert data.node_type.dtype.is_floating_point is False
    assert data.node_type.tolist() == [NODE_CLASSES["hyphal"],
                                       NODE_CLASSES["hyphal"],
                                       NODE_CLASSES["epithelial"],
                                       NODE_CLASSES["background"]]


def test_no_rule_means_no_node_type():
    """Existing datasets have no node_type; absence must stay the default."""
    ais, gt, img = _scene()
    data = build_cell_graph_data(ais, img, gt_labels=gt, k=3)
    assert getattr(data, "node_type", None) is None


def test_no_gt_means_no_node_type_even_with_a_rule():
    """Inference-only graphs have no GT to derive a type from."""
    ais, _, img = _scene()
    data = build_cell_graph_data(ais, img, gt_labels=None, k=3, cell_type_rule=RULE)
    assert getattr(data, "node_type", None) is None


def test_node_type_aligns_with_node_features():
    ais, gt, img = _scene()
    data = build_cell_graph_data(ais, img, gt_labels=gt, k=3,
                                 cell_type_rule=RULE, min_overlap_frac=0.1)
    assert data.node_type.shape[0] == data.x.shape[0]
    assert data.node_type.shape[0] == data.centroids.shape[0]
