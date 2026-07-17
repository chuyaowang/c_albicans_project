"""Per-class node-type metrics that survive a fold with a class missing entirely.

Under leave-one-out CV a fold can be tested on an image that contains no epithelial cells at
all. Recall is then undefined for that class, and scoring it 0 would punish the model for a
class the fold cannot contain. So: report only the classes present in `y_true`, and average
each class across folds only where it was present.
"""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from image_processing_tools.scene_graph_network.cell_type_labels import NODE_CLASS_NAMES


def node_type_metrics(y_true, y_pred):
    """Accuracy plus per-class precision/recall/F1, for the classes present in `y_true`.

    Presence is judged on `y_true`, never on predictions: a class absent from the labels has
    no defined recall. A class that IS present but never predicted scores precision 0 --
    that is a real miss, not an artefact, hence zero_division=0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    present = sorted(np.unique(y_true).tolist())

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=present, zero_division=0,
    )
    per_class = {
        NODE_CLASS_NAMES[c]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, c in enumerate(present)
    }
    return {
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else float("nan"),
        "per_class": per_class,
        "present": [NODE_CLASS_NAMES[c] for c in present],
    }


def aggregate_node_metrics(per_fold):
    """Average each class's F1 only over the folds where that class was present.

    Args:
        per_fold: list of `node_type_metrics` results, one per fold.

    Returns:
        dict: name -> {"f1_mean", "f1_std", "n_folds"}. `n_folds` is how many folds actually
        contained the class, so a thinner average reads as thinner rather than as a full one.
        A class present in no fold is absent from the result.
    """
    scores = {}
    for m in per_fold:
        for name, s in m["per_class"].items():
            scores.setdefault(name, []).append(s["f1"])
    return {
        name: {"f1_mean": float(np.mean(v)),
               "f1_std": float(np.std(v)),
               "n_folds": len(v)}
        for name, v in scores.items()
    }
