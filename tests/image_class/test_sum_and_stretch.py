import numpy as np
import pytest

from image_processing_tools.image_class.image_container import _sum_and_stretch


def _channels(dtype=np.uint16):
    rng = np.random.default_rng(0)
    lo, hi = 0.15, 0.55
    a = (rng.random((16, 16)) * (hi - lo) + lo) * np.iinfo(dtype).max
    b = (rng.random((16, 16)) * (hi - lo) + lo) * np.iinfo(dtype).max
    return a.astype(dtype), b.astype(dtype)


def test_stretches_to_full_dtype_range():
    a, b = _channels()
    out = _sum_and_stretch([a, b], np.uint16)
    assert out.dtype == np.uint16
    assert out.min() == 0 and out.max() == np.iinfo(np.uint16).max


def test_sum_and_mean_are_interchangeable():
    """min-max is scale-invariant and sum == n * mean, so combining by either
    gives identical output. This is why summing is safe to use throughout."""
    a, b = _channels()
    summed = _sum_and_stretch([a, b], np.uint16)

    stacked = np.stack([a, b], axis=0).astype(np.float64)
    meaned = stacked.mean(axis=0)
    lo, hi = meaned.min(), meaned.max()
    meaned = ((meaned - lo) / (hi - lo) * np.iinfo(np.uint16).max).astype(np.uint16)

    assert np.array_equal(summed, meaned)


def test_single_channel_at_full_range_is_identity():
    """A channel already spanning the dtype range (as _get_high_contrast_16bit
    leaves it) passes through unchanged, so mono and multi-channel inputs land
    on the same scale."""
    rng = np.random.default_rng(1)
    ch = (rng.random((16, 16)) * 65535).astype(np.uint16)
    ch[0, 0], ch[0, 1] = 0, 65535  # pin the range
    assert np.array_equal(_sum_and_stretch([ch], np.uint16), ch)


def test_constant_image_returns_zeros_without_dividing_by_zero():
    flat = np.full((8, 8), 300, dtype=np.uint16)
    out = _sum_and_stretch([flat, flat], np.uint16)
    assert out.dtype == np.uint16
    assert np.all(out == 0)


def test_accumulates_in_float64_so_saturated_input_cannot_overflow():
    sat = np.full((8, 8), 65535, dtype=np.uint16)
    other = np.zeros((8, 8), dtype=np.uint16)
    other[0, 0] = 65535
    out = _sum_and_stretch([sat, other], np.uint16)
    # Without float64 accumulation the sum would wrap to 65534 and invert the
    # ordering; [0, 0] is the brightest pixel and must stay at the top.
    assert out[0, 0] == 65535
    assert out[1, 1] == 0


def test_respects_dtype_width():
    a, b = _channels(np.uint8)
    out = _sum_and_stretch([a, b], np.uint8)
    assert out.dtype == np.uint8
    assert out.max() == 255


def test_float_dtype_is_rejected():
    a, b = _channels()
    with pytest.raises(ValueError):
        _sum_and_stretch([a, b], np.float32)
