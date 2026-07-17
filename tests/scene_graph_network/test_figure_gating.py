"""The prediction overlays and the merge figure are gated on `data.image` existing.

Omitting `display_image=` from `build_cell_graph_data` is a one-word mistake whose only
symptom is two figures quietly missing from TensorBoard hours later. These tests pin the
gate's behaviour and make sure it announces itself.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
from torch_geometric.data import Data

from image_processing_tools.scene_graph_network.build_cell_dataset import build_cell_graph_data
from image_processing_tools.scene_graph_network.gnn_train import _log_figures
from image_processing_tools.scene_graph_network.simple_gnn import Model


def _scene():
    gt = np.zeros((80, 80), dtype=np.int32)
    gt[10:13, 5:70] = 1
    gt[40:65, 20:45] = 2
    ais = np.zeros((80, 80), dtype=np.int32)
    ais[10:13, 5:30] = 1
    ais[10:13, 35:70] = 2
    ais[40:65, 20:45] = 3
    rng = np.random.default_rng(0)
    img = rng.integers(0, 65535, size=(80, 80), dtype=np.uint16)
    return ais, gt, img


def test_display_image_reaches_data_image():
    """The kwarg that gates the figures. Without it `data.image` never exists."""
    ais, gt, img = _scene()
    with_img = build_cell_graph_data(ais, img, gt_labels=gt, k=2, display_image=img)
    without = build_cell_graph_data(ais, img, gt_labels=gt, k=2)

    assert hasattr(with_img, "image")
    assert not hasattr(without, "image")


def test_missing_image_warns_rather_than_skipping_silently(capsys):
    """A silent skip is indistinguishable from 'the figures were logged and are boring'.

    This is the exact mistake that shipped: notebook 12 omitted display_image= and lost
    both the overlays and the 2x2 merge figure with no warning anywhere.
    """
    ais, gt, img = _scene()
    data = build_cell_graph_data(ais, img, gt_labels=gt, k=2)   # no display_image
    assert not hasattr(data, "image")

    model = Model(hidden_channels=8, dropout_p=0.0)
    _log_figures(model, [data], [0], _NullWriter(), 0.5, torch.device("cpu"))

    out = capsys.readouterr().out
    assert "image" in out and "display_image" in out, (
        f"expected a warning naming the missing attribute and the fix, got: {out!r}"
    )


def test_no_warning_when_the_image_is_present(capsys):
    ais, gt, img = _scene()
    data = build_cell_graph_data(ais, img, gt_labels=gt, k=2, display_image=img)

    model = Model(hidden_channels=8, dropout_p=0.0)
    _log_figures(model, [data], [0], _NullWriter(), 0.5, torch.device("cpu"))

    assert "[warn] Graphs carry no" not in capsys.readouterr().out


class _NullWriter:
    """Swallows the SummaryWriter calls `_log_figures` makes; we assert on the gate, not
    on what TensorBoard received."""
    log_dir = "/tmp"

    def add_figure(self, *a, **k):
        pass

    def add_text(self, *a, **k):
        pass

    def add_scalar(self, *a, **k):
        pass
