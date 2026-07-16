from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from nifty.tools import blocking as nifty_blocking

# `micro_sam` is imported lazily inside the functions that need it (`_get_encoder`,
# `compute_microsam_features`) so this module — and `load_and_stitch_saved_embeddings`
# in particular — can be used without micro_sam installed.
from image_processing_tools.image_class.image_container import ImageContainer


def _get_encoder(model_type: str, checkpoint_path: Union[os.PathLike, str]) -> util.SamPredictor:
    """Load the SAM image encoder (predictor).

    Args:
        model_type: The SAM model variant (e.g. 'vit_l_lm').
        checkpoint_path: Path to the model checkpoint.
        device: Torch device string. Defaults to the best available device.

    Returns:
        The SAM predictor with the encoder loaded.
    """
    from micro_sam import util

    device = util.get_device()
    predictor, _ = util.get_sam_model(
        model_type=model_type, device=device, checkpoint_path=checkpoint_path, return_state=True,
    )
    return predictor


def _prepare_image(image_container: ImageContainer, tile_shape: Tuple[int, int]) -> np.ndarray:
    """Merge channels, pad to 3-channel RGB, and pad spatial dims to a tile multiple.

    Args:
        image_container: Source image container.
        tile_shape: Tile size used for tiled embedding. Spatial dims are padded to a multiple of this.

    Returns:
        Padded uint8 RGB array of shape (H_padded, W_padded, 3).
    """
    img = image_container.merge()  # (H, W, C)

    # Map to exactly 3 channels (SAM expects RGB). For sub-3-channel inputs,
    # average the available channels and triplicate the result - the MicroSAM
    # paper reports that zero-padding a missing channel distorts image statistics
    # enough to hurt encoder output, while duplicating the channel mean preserves
    # the expected intensity distribution at the input of the pretrained ViT.
    if len(img.shape) == 2: # For one-channel images, the shape is just (H,W)
        img = np.repeat(img[..., None], 3, axis=-1)
    else:
        C = img.shape[2]
        if C < 3:
            avg = img.mean(axis=-1, keepdims=True).astype(img.dtype)
            img = np.repeat(avg, 3, axis=-1)
        elif C > 3:
            img = img[..., :3]

    # Pad spatial dims to a multiple of tile_shape so all tiles have symmetric halos
    H, W = img.shape[:2]
    pad_h = (tile_shape[0] - H % tile_shape[0]) % tile_shape[0]
    pad_w = (tile_shape[1] - W % tile_shape[1]) % tile_shape[1]
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)))

    return img


def _stitch_embeddings(
    emb: dict,
    padded_shape: Tuple[int, int],
    tile_shape: Tuple[int, int],
    halo: Tuple[int, int],
) -> Tuple[np.ndarray, float]:
    """Stitch per-tile SAM embeddings into a single spatial feature map.

    Args:
        emb: Output of precompute_image_embeddings (tiled mode).
        padded_shape: Spatial shape (H, W) of the padded input image.
        tile_shape: Tile size used during embedding.
        halo: Halo size used during embedding.

    Returns:
        feature_map: Array of shape (256, feature_map_H, feature_map_W).
        pixels_per_feature: Image pixels spanned by one feature map location.
    """
    tile_shape = list(tile_shape)
    halo = list(halo)
    tiling = nifty_blocking([0, 0], list(padded_shape[:2]), tile_shape)

    # One feature map location spans (tile + 2*halo) / 64 image pixels (non-edge tile)
    pixels_per_feature = (tile_shape[0] + 2 * halo[0]) / 64

    feature_map_H = int(np.ceil(padded_shape[0] / pixels_per_feature))
    feature_map_W = int(np.ceil(padded_shape[1] / pixels_per_feature))
    feature_map = np.zeros((256, feature_map_H, feature_map_W), dtype=np.float32)

    for tile_id in range(tiling.numberOfBlocks):
        tile = tiling.getBlockWithHalo(tile_id, halo)
        inner, outer = tile.innerBlock, tile.outerBlock

        # SAM output for this tile: (256, 64, 64) feature grid
        tile_features = np.array(emb["features"][str(tile_id)])[0]

        # Crop the halo region out of the 64x64 feature grid using this tile's actual scale
        outer_h = outer.end[0] - outer.begin[0]
        outer_w = outer.end[1] - outer.begin[1]
        halo_offset_h = inner.begin[0] - outer.begin[0]
        halo_offset_w = inner.begin[1] - outer.begin[1]
        inner_h = inner.end[0] - inner.begin[0]
        inner_w = inner.end[1] - inner.begin[1]

        feat_beg_h = int(round(halo_offset_h * 64 / outer_h))
        feat_end_h = int(round((halo_offset_h + inner_h) * 64 / outer_h))
        feat_beg_w = int(round(halo_offset_w * 64 / outer_w))
        feat_end_w = int(round((halo_offset_w + inner_w) * 64 / outer_w))
        inner_features = tile_features[:, feat_beg_h:feat_end_h, feat_beg_w:feat_end_w]

        # Target slot in the output feature map
        out_beg_h = int(round(inner.begin[0] / pixels_per_feature))
        out_beg_w = int(round(inner.begin[1] / pixels_per_feature))
        out_end_h = min(int(round(inner.end[0] / pixels_per_feature)), feature_map_H)
        out_end_w = min(int(round(inner.end[1] / pixels_per_feature)), feature_map_W)
        target_h = out_end_h - out_beg_h
        target_w = out_end_w - out_beg_w

        # Edge tiles have asymmetric halos so the cropped feature grid may not match
        # the target slot size exactly — resize with bilinear interpolation
        if inner_features.shape[1] != target_h or inner_features.shape[2] != target_w:
            inner_features = F.interpolate(
                torch.from_numpy(inner_features).unsqueeze(0),
                size=(target_h, target_w), mode='bilinear', align_corners=False
            )[0].numpy()

        feature_map[:, out_beg_h:out_end_h, out_beg_w:out_end_w] = inner_features

    return feature_map, pixels_per_feature


def compute_microsam_features(
    image_container: ImageContainer,
    config: dict,
    save_path: Optional[Union[str, Path]] = None,
    predictor: Optional[util.SamPredictor] = None,
) -> Tuple[np.ndarray, float]:
    """Compute tiled SAM image embeddings and stitch into a whole-image feature map.

    Args:
        image_container: Source image. Channels are merged and padded to RGB internally.
        config: Dictionary with the following keys:
            - model_type (str): SAM model variant, e.g. 'vit_l_lm'.
            - checkpoint_path (str | Path): Path to the model checkpoint.
            - tile_shape (tuple[int, int]): Spatial size of each tile fed to the encoder.
            - halo (tuple[int, int]): Overlap added around each tile to reduce edge artifacts.
        save_path: Directory to save the output .npz file. Defaults to the parent
            directory of the first source image in image_container.
        predictor: Optional pre-loaded SAM predictor. Pass this when processing multiple
            images to avoid reloading the model for each call. If None, the model is
            loaded from config on every call.

    Returns:
        feature_map: Stitched feature map of shape (256, feature_map_H, feature_map_W).
        pixels_per_feature: Image pixels spanned by one feature map location.
            Use 1/pixels_per_feature as spatial_scale in torchvision RoIAlign.
    """
    model_type = config["model_type"]
    checkpoint_path = Path(config["checkpoint_path"]).expanduser()
    tile_shape = config["tile_shape"]
    halo = config["halo"]

    if save_path is None:
        save_path = image_container._source_paths[0].parent
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if predictor is None:
        predictor = _get_encoder(model_type, checkpoint_path)
    img_padded = _prepare_image(image_container, tile_shape)

    from micro_sam.util import precompute_image_embeddings

    emb = precompute_image_embeddings(
        predictor, img_padded, ndim=2, tile_shape=tile_shape, halo=halo,
    )

    feature_map, pixels_per_feature = _stitch_embeddings(
        emb, img_padded.shape[:2], tile_shape, halo,
    )

    out_file = save_path / f"{image_container._source_paths[0].parent}/DAPI_DIC_microsam_features.npz"
    np.savez(out_file, feature_map=feature_map, pixels_per_feature=np.float32(pixels_per_feature))

    return feature_map, pixels_per_feature


def load_and_stitch_saved_embeddings(embedding_path, save_path=None,
                                     npz_name="microsam_features.npz"):
    """Stitch micro-sam's saved *tiled* embedding store into a whole-image feature map.

    Reads the zarr container micro-sam writes when precomputing tiled 2D embeddings
    (a `features` group of per-tile (1,256,64,64) datasets with `shape`/`tile_shape`/
    `halo` attrs) and reuses `_stitch_embeddings`. No SAM model or predictor needed.

    Args:
        embedding_path: Path to the saved embedding zarr container.
        save_path: If given, directory to write an `.npz` (`feature_map`,
            `pixels_per_feature`).
        npz_name: File name for the saved `.npz`.

    Returns:
        (feature_map (256, Hf, Wf) float32, pixels_per_feature float).
    """
    import zarr

    # open_group (not zarr.open) so this works whether the store is a v2 or v3
    # group; some zarr builds' `zarr.open` tries to open an array first and errors
    # on a group.
    f = zarr.open_group(str(embedding_path), mode="r")
    feats = f["features"]

    # Non-tiled fallback: a single (1,256,64,64) or (256,64,64) dataset, no tiles.
    tile_shape = feats.attrs.get("tile_shape", None) if hasattr(feats, "attrs") else None
    if tile_shape is None:
        raise ValueError(
            f"Embedding store at {embedding_path} is not tiled (tile_shape is None); "
            "this loader expects the tiled store micro-sam writes for tiled AIS."
        )

    shape = tuple(feats.attrs["shape"])
    halo = tuple(feats.attrs["halo"])
    tile_shape = tuple(tile_shape)

    feature_map, pixels_per_feature = _stitch_embeddings(
        {"features": feats}, shape, tile_shape, halo,
    )

    if save_path is not None:
        out_dir = Path(save_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / npz_name,
                 feature_map=feature_map,
                 pixels_per_feature=np.float32(pixels_per_feature))

    return feature_map, pixels_per_feature


def load_or_stitch_embeddings(embedding_path, save_path, npz_name, env="microsam"):
    """Return a path to a stitched ``.npz`` feature map for ``embedding_path``.

    Tries to stitch in the current process. If the current env's zarr cannot read
    the store — e.g. a zarr **v3** store under an older zarr, which happens when the
    store was written by micro-sam in a newer env than the one running the GNN — it
    falls back to running the identical stitch in another conda env (default
    ``'microsam'``) via ``conda run``, which then writes the ``.npz``.

    Results are cached: if ``save_path/npz_name`` already exists it is returned
    directly, so the (possibly slow) subprocess stitch runs at most once per image.

    Args:
        embedding_path: Path to the saved micro-sam embedding zarr store.
        save_path: Directory the ``.npz`` is written to.
        npz_name: File name for the ``.npz``.
        env: Conda env name to fall back to (must have a zarr that can read the store
            and have ``image_processing_tools`` importable).

    Returns:
        Path (str) to the stitched ``.npz``.
    """
    out_npz = Path(save_path) / npz_name
    if out_npz.exists():
        return str(out_npz)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    try:
        load_and_stitch_saved_embeddings(
            embedding_path, save_path=str(save_path), npz_name=npz_name)
        return str(out_npz)
    except Exception as exc_inproc:
        import subprocess
        conda_exe = os.environ.get("CONDA_EXE", "conda")
        code = (
            "from image_processing_tools.scene_graph_network.precompute_microsam_feats "
            "import load_and_stitch_saved_embeddings as f; "
            f"f({str(embedding_path)!r}, save_path={str(save_path)!r}, "
            f"npz_name={npz_name!r})"
        )
        proc = subprocess.run(
            [conda_exe, "run", "-n", env, "python", "-c", code],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not out_npz.exists():
            raise RuntimeError(
                f"Could not stitch embeddings for {embedding_path}.\n"
                f"In-process stitch failed: {type(exc_inproc).__name__}: {exc_inproc}\n"
                f"Fallback `{conda_exe} run -n {env}` exit={proc.returncode}; "
                f"stderr tail:\n{proc.stderr[-2000:]}"
            )
        return str(out_npz)