# MicroSAMPipeline

`MicroSAMPipeline` (in `image_class/micro_sam_pipeline.py`) wraps the whole microSAM
segmentation workflow behind one object. It takes one or more `ImageContainer`s and a
config dict, loads the SAM predictor (and, for automatic mode, the instance-segmentation
decoder), runs segmentation in the requested mode, and stores, saves, and visualizes the
results. It supports 2D and 3D data and three segmentation modes.

## Construction and config

`MicroSAMPipeline(image_containers, config)` accepts a single `ImageContainer` or a list,
and a config dict.

**Required config keys:** `model_type` (e.g. `vit_l_lm`), `checkpoint_path` (SAM
checkpoint), `base_input_dir` (used to structure visualization output paths).

**Optional / conditional keys:**

- `segmentation_mode` — `prompted` (default), `automatic`, or `combined`.
- `decoder_checkpoint_path` — required for `automatic` and `combined`; the AIS decoder.
- `ndim` — `2` (default) or `3`.
- `tiling` — `{tile_shape, halo}`; when both are set the run is tiled.
- `preprocessing`, `prompting` — passed through to the `ImageContainer` (see
  [ImageContainer](ImageContainer.md)).
- `segmentation_3d`, `ais_generate`, `debug_mode` — mode-specific parameters.

`_initialize_models` loads the predictor for prompted mode, and additionally attaches the
decoder (`get_decoder` + `get_instance_segmentation_generator`) for automatic and combined
modes. The device is CUDA when available, else CPU.

## The image is always a merged `ImageContainer`

Every processing path starts by calling `image_container.merge()` to obtain the array fed
to microSAM. This is the same reduction documented in
[ImageContainer](ImageContainer.md#channel-reduction-merge-and-_sum_and_stretch): a 2D
container yields `(H, W)` or `(H, W, C)`, a 3D z-stack yields `(Z, H, W, 3)`. The pipeline
never manipulates raw channels itself; all preprocessing, resizing, and channel-to-RGB
handling live in the container. The **channel-to-RGB note** (averaging vs. padding a
2-channel image, and the authors' current recommendation of padding) is documented in
[ImageContainer](ImageContainer.md#mapping-channels-to-the-3-channel-rgb-that-microsam-expects).

## Segmentation modes

### Prompted (`segmentation_mode: prompted`)

The container generates prompts from its DAPI channel (`generate_prompts` →
points / bbox / mask), and `batched_inference` (or `batched_tiled_inference` when tiling
is set) segments one object per prompt. `_process_single_image` handles 2D;
`_process_3d_image` handles 3D by generating prompts on a seed slice, precomputing 3D
embeddings, running seed-slice inference, then propagating each mask through the volume
with `segment_mask_in_volume`. Result data holds `masks` (a list of per-object binary
masks), `prompts`, `prompt_type`, `scores`, and `logits`.

### Automatic instance segmentation (`segmentation_mode: automatic`)

Uses the decoder-based `automatic_instance_segmentation` (AIS), which needs no prompts.
`_run_ais_on_image` handles 2D and `run_3d_ais` handles 3D. In addition to the instance
`masks`, AIS exposes the decoder's raw maps — `foreground`, `center_distances`, and
`boundary_distances` — which are the inputs the seeded-watershed post-processing turns
into instance labels, and which the downstream graph pipeline reads. The 3D path segments
each slice, offsets the per-slice labels so ids stay unique, then merges them into a
coherent 3D instance mask with `merge_instance_segmentation_3d`.

### Combined (`segmentation_mode: combined`, 3D only)

`_run_combined_3d` runs AIS on a seed slice to get automatic instance masks, then
propagates each instance through the volume with `segment_mask_in_volume`. This pairs
AIS's prompt-free detection with volumetric propagation. It requires `ndim = 3`.

## Embeddings and caching

`_get_embedding_path` returns a per-container `.zarr` path under `microsam_outputs/`, with
a `_tiled` suffix when tiling is on so a tiled and a non-tiled run never reuse each other's
incompatible cache. 3D paths precompute embeddings once with `precompute_image_embeddings`
and reuse them for inference and propagation. These are microSAM's segmentation
embeddings; the graph pipeline's visual branch computes its own encoder features
separately (`scene_graph_network/precompute_microsam_feats`).

## Running, retrieving, saving

- `run()` dispatches over the containers by mode and `ndim`, storing each result in
  `self.results` keyed by `container.name`.
- `get_masks(name)` returns the stored masks; `extract_objects(name)` crops each
  segmented object to its bounding box and zeros the background.
- `save_results()` writes to `<source_parent>/microsam_outputs/`, embedding the mode in
  every filename so AIS and prompted runs on the same image never overwrite each other:
  `{name}_{mode}_masks.tif` (instance labels at original resolution),
  `{name}_ais_raw.tif` (`(3, …)` float32 `[foreground, center_dist, boundary_dist]`),
  and, for prompted runs, `{name}_prompted_prompts.npy` and `{name}_prompted_viz.tif`.
  `_rescale_to_original` maps masks (nearest) and float maps (linear) back to the input
  resolution before writing.
- `visualize_results(mode)` saves either one figure per image (`single`) or one combined
  figure across the run's channels (`channel_comparison`).