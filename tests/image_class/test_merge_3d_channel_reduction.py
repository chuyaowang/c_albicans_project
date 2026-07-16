import numpy as np
import tifffile

from image_processing_tools.image_class.image_container import ImageContainer

Z, H, W = 4, 12, 12
FULL = np.iinfo(np.uint16).max


def _config(**over):
    cfg = {"resize_image": False, "max_dim": 1080, "outlier_percentile": 0.35,
           "quantization": "16bit"}
    cfg.update(over)
    return {"preprocessing": cfg}


def _volume(seed, lo, hi):
    rng = np.random.default_rng(seed)
    return (rng.random((Z, H, W)) * (hi - lo) + lo).astype(np.uint16)


def _write(tmp_path, n):
    """n z-stack channel files, each spanning a different, narrow band."""
    paths = []
    for i in range(n):
        p = tmp_path / f"ch{i}.tif"
        # photometric pinned so a z-stack is never read back as RGB planes.
        tifffile.imwrite(p, _volume(i, 8000 + i * 3000, 14000 + i * 3000),
                         photometric="minisblack")
        paths.append(p)
    return paths


def test_two_channels_are_summed_stretched_and_replicated(tmp_path):
    paths = _write(tmp_path, 2)
    out = ImageContainer(list(paths), _config()).merge()

    assert out.shape == (Z, H, W, 3)
    # All three planes are the same combined channel.
    assert np.array_equal(out[..., 0], out[..., 1])
    assert np.array_equal(out[..., 1], out[..., 2])
    # The combined channel is stretched to full contrast. Averaging two
    # mid-range channels without a stretch would leave it compressed.
    assert out[..., 0].min() == 0 and out[..., 0].max() == FULL


def test_two_channel_passthrough_mode_is_unchanged(tmp_path):
    paths = _write(tmp_path, 2)
    out = ImageContainer(list(paths), _config(two_channel_merge_mode="passthrough")).merge()
    assert out.shape == (Z, H, W, 2)


def test_three_channels_are_stacked_untouched(tmp_path):
    paths = _write(tmp_path, 3)
    out = ImageContainer(list(paths), _config()).merge()
    assert out.shape == (Z, H, W, 3)


def test_extra_channels_are_summed_and_stretched_into_ch3(tmp_path):
    paths = _write(tmp_path, 5)
    out = ImageContainer(list(paths), _config()).merge()

    assert out.shape == (Z, H, W, 3)
    # ch3 carries the reduction of channels 3..5 and is stretched to full range.
    assert out[..., 2].min() == 0 and out[..., 2].max() == FULL


def test_merge_3d_preserves_input_dtype(tmp_path):
    paths = _write(tmp_path, 2)
    out = ImageContainer(list(paths), _config()).merge()
    assert out.dtype == np.uint16
