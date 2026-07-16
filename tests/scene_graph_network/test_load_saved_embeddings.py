import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
pytest.importorskip("nifty")     # _stitch_embeddings uses nifty + torch
pytest.importorskip("torch")

from image_processing_tools.scene_graph_network.precompute_microsam_feats import (
    load_and_stitch_saved_embeddings,
)


def _fake_micro_sam_store(path, shape=(512, 512), tile_shape=(512, 512), halo=(0, 0)):
    """Mimic the zarr layout micro-sam writes for tiled 2D embeddings."""
    f = zarr.open(str(path), mode="a")
    feats = f.require_group("features")
    feats.attrs["shape"] = list(shape)
    feats.attrs["tile_shape"] = list(tile_shape)
    feats.attrs["halo"] = list(halo)
    # single tile covering the whole image: (1, 256, 64, 64)
    grid = np.random.default_rng(0).random((1, 256, 64, 64)).astype("float32")
    feats["0"] = grid
    return f


def test_stitch_shape_and_scale(tmp_path):
    store = tmp_path / "emb.zarr"
    _fake_micro_sam_store(store)
    feature_map, ppf = load_and_stitch_saved_embeddings(str(store))
    assert feature_map.shape[0] == 256
    # one 512-px tile, halo 0 -> 512/64 = 8 px per feature
    assert ppf == pytest.approx(8.0)
    assert feature_map.shape[1] == 64 and feature_map.shape[2] == 64


def test_writes_npz(tmp_path):
    store = tmp_path / "emb.zarr"
    _fake_micro_sam_store(store)
    out = tmp_path / "out"
    load_and_stitch_saved_embeddings(str(store), save_path=str(out))
    npz = np.load(out / "microsam_features.npz")
    assert npz["feature_map"].shape[0] == 256
    assert float(npz["pixels_per_feature"]) == pytest.approx(8.0)