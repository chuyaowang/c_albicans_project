import numpy as np
from image_processing_tools.scene_graph_network.cell_merge_labels import (
    assign_fragments_to_gt, cell_merge_labels,
)


def _three_fragment_hypha():
    # ais: three collinear fragments (labels 1,2,3) + one separate fragment (4).
    ais = np.zeros((20, 60), dtype=np.int32)
    ais[8:12, 2:14] = 1
    ais[8:12, 16:28] = 2
    ais[8:12, 30:42] = 3
    ais[8:12, 50:58] = 4      # a correctly-segmented separate cell
    # gt: fragments 1,2,3 are ONE cell (label 10); fragment 4 is its own cell (20).
    gt = np.zeros((20, 60), dtype=np.int32)
    gt[8:12, 2:42] = 10
    gt[8:12, 50:58] = 20
    return ais, gt


def test_assignment_maps_fragments_to_majority_gt_cell():
    ais, gt = _three_fragment_hypha()
    gt_of_node = assign_fragments_to_gt(ais, gt)
    # regionprops order → node 0..3 correspond to ais labels 1..4
    assert gt_of_node.tolist() == [10, 10, 10, 20]


def test_mst_labels_form_a_chain_not_a_clique():
    ais, gt = _three_fragment_hypha()
    edges = set(cell_merge_labels(ais, gt))
    # fragments 1,2,3 (nodes 0,1,2) form a chain 0-1-2, NOT the clique (no 0-2)
    assert edges == {(0, 1), (1, 2)}


def test_background_fragment_excluded():
    ais, gt = _three_fragment_hypha()
    # add a fragment (label 5) that overlaps no gt cell → background
    ais[2:4, 2:6] = 5
    gt_of_node = assign_fragments_to_gt(ais, gt)
    assert gt_of_node[-1] == -1                       # node 4 = ais label 5
    # background node appears in no true edge
    assert all(4 not in e for e in cell_merge_labels(ais, gt))