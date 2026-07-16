import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from image_processing_tools.scene_graph_network.gnn_train import _imshow_microscopy


def _render(image):
    """Draw `image` and return what matplotlib actually rasterizes, as uint8 RGBA.

    Asserting on the rendered pixels rather than the input is the whole point:
    imshow silently clips out-of-range RGB, so the input can look fine while the
    output is blown out.
    """
    fig, ax = plt.subplots()
    try:
        im = _imshow_microscopy(ax, image) or ax.images[-1]
        return im.make_image(fig.canvas.get_renderer())[0]
    finally:
        plt.close(fig)


def _channel(rng, shape, lo, hi):
    return rng.integers(lo, hi, size=shape, dtype=np.uint16)


def test_three_channel_uint16_is_not_blown_out():
    """Channels arrive stretched to 0..65535; imshow clips RGB at 255.

    Passed through raw, essentially every pixel saturates to white. Regression
    test for the 3-channel display path.
    """
    rng = np.random.default_rng(0)
    img = np.stack([_channel(rng, (64, 64), 0, 65535) for _ in range(3)], axis=-1)

    rgba = _render(img)

    saturated = np.mean(np.all(rgba[..., :3] == 255, axis=-1))
    assert saturated < 0.10, f"{saturated:.1%} of pixels rendered pure white"


def test_three_channel_stretches_each_channel_independently():
    """A dim channel must stay visible next to a bright one.

    Per-channel stretch, matching the 2-channel branch: a channel occupying a
    narrow slice of the range still spans the output range.
    """
    rng = np.random.default_rng(1)
    bright = _channel(rng, (64, 64), 30000, 65535)
    dim = _channel(rng, (64, 64), 0, 800)
    img = np.stack([bright, dim, dim], axis=-1)

    rgba = _render(img).astype(np.float64)

    for c, name in enumerate(("bright", "dim", "dim")):
        spread = rgba[..., c].max() - rgba[..., c].min()
        assert spread > 200, f"channel {c} ({name}) collapsed to a flat {spread:.0f} of range"


def test_two_channel_composite_still_normalizes():
    rng = np.random.default_rng(2)
    img = np.stack([_channel(rng, (64, 64), 0, 800),
                    _channel(rng, (64, 64), 30000, 65535)], axis=-1)

    rgba = _render(img)

    assert np.mean(np.all(rgba[..., :3] == 255, axis=-1)) < 0.10


@pytest.mark.parametrize("shape", [(64, 64), (64, 64, 1)])
def test_single_channel_uses_colormap_autoscaling(shape):
    """2D and (H,W,1) go through Normalize, so uint16 needs no help."""
    rng = np.random.default_rng(3)
    img = _channel(rng, shape, 0, 65535)

    rgba = _render(img)

    assert np.mean(np.all(rgba[..., :3] == 255, axis=-1)) < 0.10


def test_flat_channel_does_not_divide_by_zero():
    img = np.full((32, 32, 3), 40000, dtype=np.uint16)

    rgba = _render(img)

    assert np.isfinite(rgba.astype(np.float64)).all()