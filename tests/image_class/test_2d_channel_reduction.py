"""The 2D channel-reduction contract the cell-fragment pipeline depends on.

Grouping channels in the constructor structure reduces them via _sum_channels;
leaving them ungrouped keeps them separate for the display image.
"""
import numpy as np
import tifffile

from image_processing_tools.image_class.image_container import ImageContainer

H = W = 24
FULL = np.iinfo(np.uint16).max


def _config():
    return {"preprocessing": {"resize_image": False, "max_dim": 1080,
                              "outlier_percentile": 0.35, "quantization": "16bit",
                              "correct_DIC_shift": [0, 0]}}


def _write(tmp_path, name, lo, hi, seed, speck=False):
    rng = np.random.default_rng(seed)
    img = (rng.random((H, W)) * (hi - lo) + lo)
    if speck:
        img[0, 0] = FULL  # hot pixel
    p = tmp_path / f"{name}.tif"
    tifffile.imwrite(p, img.astype(np.uint16), photometric="minisblack")
    return p


def test_grouped_channels_reduce_to_one_2d_channel(tmp_path):
    paths = [_write(tmp_path, "c0", 8000, 14000, 0),
             _write(tmp_path, "c1", 1000, 4000, 1),
             _write(tmp_path, "c2", 500, 2000, 2)]
    out = ImageContainer([paths], _config()).merge()
    assert out.shape == (H, W)
    assert out.min() == 0 and out.max() == FULL


def test_ungrouped_channels_stay_separate_for_display(tmp_path):
    paths = [_write(tmp_path, "c0", 8000, 14000, 0),
             _write(tmp_path, "c1", 1000, 4000, 1)]
    out = ImageContainer(paths, _config()).merge()
    assert out.shape == (H, W, 2)


def test_mono_and_multi_land_on_the_same_scale(tmp_path):
    """Summing a single channel is the identity, because the channel arrives
    already stretched to 0..65535. So a {dic} sample and a {dic, fluor, dapi}
    sample produce intensity images on the same scale, and the pooled
    training-fold z-score compares like with like."""
    mono = ImageContainer([[_write(tmp_path, "m", 20000, 30000, 3)]], _config()).merge()
    multi = ImageContainer([[_write(tmp_path, "a", 20000, 30000, 3),
                             _write(tmp_path, "b", 1000, 5000, 4)]], _config()).merge()
    assert mono.shape == multi.shape == (H, W)
    assert (mono.min(), mono.max()) == (0, FULL)
    assert (multi.min(), multi.max()) == (0, FULL)


def test_per_channel_clipping_stops_a_hot_pixel_setting_the_scale(tmp_path):
    """Each channel is percentile-clipped before the sum, so a speck cannot
    compress the real signal into a sliver of the range."""
    paths = [_write(tmp_path, "s0", 8000, 14000, 5, speck=True),
             _write(tmp_path, "s1", 8000, 14000, 6, speck=True)]
    out = ImageContainer([paths], _config()).merge().astype(np.float64)
    p5, p95 = np.percentile(out, [5, 95])
    # The body of the signal keeps most of the range rather than being crushed.
    assert (p95 - p5) / FULL > 0.5


def test_single_channel_container_merge_is_unreduced_2d(tmp_path):
    """The nuclei pipeline's path: one channel, ungrouped -> (H, W). Guards the
    historical DAPI-intensity behaviour against regression."""
    out = ImageContainer(_write(tmp_path, "dapi", 5000, 40000, 7), _config()).merge()
    assert out.shape == (H, W)
