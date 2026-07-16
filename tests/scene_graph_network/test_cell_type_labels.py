import numpy as np
import pytest

from image_processing_tools.scene_graph_network.cell_type_labels import (
    CELL_TYPE_METRICS, NODE_CLASSES, gt_cell_types, node_type_labels,
)


def _two_cell_gt():
    """GT with one thin filament (label 1) and one fat blob (label 2)."""
    gt = np.zeros((60, 60), dtype=np.int32)
    gt[10:12, 5:55] = 1        # 2 px thick, 50 long  -> mean_width ~2
    gt[30:50, 20:40] = 2       # 20x20 blob           -> mean_width ~20
    return gt


def test_mean_width_separates_filament_from_blob():
    gt = _two_cell_gt()
    types = gt_cell_types(gt, {"metric": "mean_width", "threshold": 8.0})
    assert types[1] == "hyphal"
    assert types[2] == "epithelial"


def test_all_rule_ignores_shape():
    """A hyphae-only image has no epithelial population to threshold against."""
    gt = _two_cell_gt()
    types = gt_cell_types(gt, {"all": "hyphal"})
    assert types == {1: "hyphal", 2: "hyphal"}


def test_direction_is_carried_by_the_metric():
    """axis_ratio is hyphal-when-high; mean_width is hyphal-when-low. Same cell, both agree."""
    gt = _two_cell_gt()
    by_width = gt_cell_types(gt, {"metric": "mean_width", "threshold": 8.0})
    by_ratio = gt_cell_types(gt, {"metric": "axis_ratio", "threshold": 5.0})
    assert by_width[1] == by_ratio[1] == "hyphal"
    assert by_width[2] == by_ratio[2] == "epithelial"


def test_every_metric_declares_a_valid_direction():
    for name, (fn, direction) in CELL_TYPE_METRICS.items():
        assert direction in ("low", "high"), f"{name} has direction {direction!r}"


def test_background_fragment_gets_class_zero():
    gt = _two_cell_gt()
    ais = np.zeros((60, 60), dtype=np.int32)
    ais[10:12, 5:30] = 1       # on the filament
    ais[30:50, 20:40] = 2      # on the blob
    ais[0:5, 0:5] = 3          # overlaps no GT at all -> background

    out = node_type_labels(ais, gt, {"metric": "mean_width", "threshold": 8.0},
                           min_overlap_frac=0.1)
    assert out.tolist() == [NODE_CLASSES["hyphal"],
                            NODE_CLASSES["epithelial"],
                            NODE_CLASSES["background"]]


def test_all_fragments_of_one_gt_cell_share_its_type():
    """The wiring invariant: a fragment's type is looked up from the GT cell it was
    assigned to, so the fragment's own shape never enters the computation.

    Both fragments below belong to one fat epithelial cell, so both must be epithelial.
    Reading the type off the fragment instead would split them -- the sliver's own
    mean_width is 1.73 (hyphal), the chunk's is 12.16 (epithelial) -- which is exactly
    the failure this guards: the node head would collapse into a restatement of shape
    features the model already has as inputs, while still appearing to work.

    This is the real case, not a contrived one: AIS oversegments a fat epithelial cell
    into pieces, and some of those pieces are slivers.
    """
    gt = np.zeros((90, 90), dtype=np.int32)
    gt[30:58, 20:48] = 1                       # one fat epithelial cell, mean_width 24.26

    ais = np.zeros((90, 90), dtype=np.int32)
    ais[30:58, 22:24] = 1                      # a thin sliver of it,  own mean_width 1.73
    ais[30:44, 30:44] = 2                      # a fat chunk of it,    own mean_width 12.16

    out = node_type_labels(ais, gt, {"metric": "mean_width", "threshold": 8.0},
                           min_overlap_frac=0.1)
    assert out.tolist() == [NODE_CLASSES["epithelial"], NODE_CLASSES["epithelial"]]


def test_node_order_matches_regionprops_ascending_labels():
    gt = _two_cell_gt()
    ais = np.zeros((60, 60), dtype=np.int32)
    ais[30:50, 20:40] = 7      # blob, high label
    ais[10:12, 5:30] = 2       # filament, low label

    out = node_type_labels(ais, gt, {"metric": "mean_width", "threshold": 8.0},
                           min_overlap_frac=0.1)
    # regionprops yields label 2 then label 7, so filament first.
    assert out.tolist() == [NODE_CLASSES["hyphal"], NODE_CLASSES["epithelial"]]


def test_unknown_metric_raises():
    with pytest.raises(KeyError):
        gt_cell_types(_two_cell_gt(), {"metric": "not_a_metric", "threshold": 1.0})
