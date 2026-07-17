import numpy as np

from image_processing_tools.scene_graph_network.build_cell_dataset import build_cell_graph_data
from image_processing_tools.scene_graph_network.cell_type_labels import NODE_CLASSES


def _scene():
    """Five fragments over two GT cells: two on a filament, one on a blob, one on
    background, and one deliberately STRADDLING the filament's edge.

    Fragment 5 is what gives `min_overlap_frac` something to decide. Every other fragment
    lies either wholly inside a GT cell or wholly outside one, so any cutoff in (0, 1)
    treats them identically -- a fixture where the parameter could be hardcoded and no test
    would notice. Fragment 5 sits 30% on the filament, so it is a cell fragment at 0.1 and
    background at 0.5.
    """
    gt = np.zeros((80, 80), dtype=np.int32)
    gt[10:13, 5:70] = 1                       # thin filament
    gt[40:65, 20:45] = 2                      # fat blob

    ais = np.zeros((80, 80), dtype=np.int32)
    ais[10:13, 5:30] = 1                      # filament, left half   -> 100% on gt 1
    ais[10:13, 35:70] = 2                     # filament, right half  -> 100% on gt 1
    ais[40:65, 20:45] = 3                     # blob                  -> 100% on gt 2
    ais[70:78, 70:78] = 4                     # background            ->   0% on any
    # 50 px, of which rows 10-12 x cols 30-34 = 15 px lie on gt 1 -> 30%. It fills the gap
    # the two filament fragments leave (cols 30-34) and hangs off the filament downwards,
    # so it collides with no other label -- the blob is not usable here, since ais 3 covers
    # gt 2 exactly and leaves it no free area.
    ais[10:20, 30:35] = 5                     # straddles gt 1's edge ->  30%

    rng = np.random.default_rng(0)
    img = rng.integers(0, 65535, size=(80, 80), dtype=np.uint16)
    return ais, gt, img


STRADDLER = 4          # index of fragment 5 in regionprops (label-ascending) order


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
                                       NODE_CLASSES["background"],
                                       NODE_CLASSES["hyphal"]]     # straddler, at 0.1


def test_min_overlap_frac_decides_the_straddling_fragment():
    """The caller's min_overlap_frac must actually reach the node-type labels.

    Without this, the parameter can be hardcoded anywhere in the chain and every other test
    still passes -- every other fragment is 100% or 0% on its GT cell, so no cutoff can be
    told from another. Fragment 5 is 30% on the filament, so the two thresholds must
    disagree about it. The value is load-bearing: it feeds the node types AND the edge
    labels, and decoupling them silently relabels fragments while deleting their true
    merge edges.
    """
    ais, gt, img = _scene()
    lo = build_cell_graph_data(ais, img, gt_labels=gt, k=3,
                               cell_type_rule=RULE, min_overlap_frac=0.1)
    hi = build_cell_graph_data(ais, img, gt_labels=gt, k=3,
                               cell_type_rule=RULE, min_overlap_frac=0.5)

    assert lo.node_type[STRADDLER] == NODE_CLASSES["hyphal"]        # 0.30 >= 0.1 -> gt 1's type
    assert hi.node_type[STRADDLER] == NODE_CLASSES["background"]    # 0.30 <  0.5 -> rejected
    # and nothing else moves: the other four are 100%/0% and cannot straddle any cutoff
    assert lo.node_type.tolist()[:STRADDLER] == hi.node_type.tolist()[:STRADDLER]


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
