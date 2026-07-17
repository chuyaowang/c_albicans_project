import numpy as np

from image_processing_tools.scene_graph_network.node_metrics import (
    aggregate_node_metrics, node_type_metrics,
)


def test_only_classes_present_in_y_true_are_reported():
    """Images 0/1 have no epithelial nodes; scoring epithelial 0 there would punish the
    model for a class the fold cannot contain."""
    y_true = np.array([0, 0, 2, 2, 2])          # no epithelial
    y_pred = np.array([0, 0, 2, 2, 2])
    m = node_type_metrics(y_true, y_pred)
    assert set(m["present"]) == {"background", "hyphal"}
    assert "epithelial" not in m["per_class"]


def test_perfect_prediction_scores_one():
    y = np.array([0, 1, 2, 1, 2])
    m = node_type_metrics(y, y)
    assert m["accuracy"] == 1.0
    for name in ("background", "epithelial", "hyphal"):
        assert m["per_class"][name]["f1"] == 1.0


def test_present_but_never_predicted_class_scores_zero_not_undefined():
    """A real miss, not an artefact -- unlike an absent class."""
    y_true = np.array([0, 1, 1, 2])
    y_pred = np.array([0, 2, 2, 2])             # epithelial never predicted
    m = node_type_metrics(y_true, y_pred)
    assert m["per_class"]["epithelial"]["recall"] == 0.0
    assert m["per_class"]["epithelial"]["precision"] == 0.0
    assert m["per_class"]["epithelial"]["f1"] == 0.0


def test_support_counts_y_true():
    y_true = np.array([0, 0, 0, 1, 2])
    y_pred = np.array([0, 0, 0, 1, 2])
    m = node_type_metrics(y_true, y_pred)
    assert m["per_class"]["background"]["support"] == 3
    assert m["per_class"]["epithelial"]["support"] == 1


def test_aggregate_averages_each_class_only_over_folds_where_present():
    per_fold = [
        node_type_metrics(np.array([0, 2]), np.array([0, 2])),        # no epithelial
        node_type_metrics(np.array([0, 2]), np.array([0, 2])),        # no epithelial
        node_type_metrics(np.array([0, 1, 2]), np.array([0, 1, 2])),
        node_type_metrics(np.array([0, 1, 2]), np.array([0, 1, 2])),
    ]
    agg = aggregate_node_metrics(per_fold)
    assert agg["epithelial"]["n_folds"] == 2      # NOT 4
    assert agg["background"]["n_folds"] == 4
    assert agg["epithelial"]["f1_mean"] == 1.0


def test_aggregate_does_not_dilute_with_absent_folds():
    """The bug this guards: averaging a missing class in as 0 across all folds."""
    per_fold = [
        node_type_metrics(np.array([0, 2]), np.array([0, 2])),        # epithelial absent
        node_type_metrics(np.array([1]), np.array([1])),              # epithelial perfect
    ]
    agg = aggregate_node_metrics(per_fold)
    assert agg["epithelial"]["f1_mean"] == 1.0    # not 0.5
    assert agg["epithelial"]["n_folds"] == 1


def test_class_absent_from_every_fold_is_absent_from_the_aggregate():
    per_fold = [node_type_metrics(np.array([0, 2]), np.array([0, 2]))]
    agg = aggregate_node_metrics(per_fold)
    assert "epithelial" not in agg
