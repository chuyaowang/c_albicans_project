# ImageContainer

`ImageContainer` (in `image_class/image_container.py`) is the single entry point for
loading, preprocessing, composing, and prompting microscopy images. Every downstream
component — `MicroSAMPipeline`, the microSAM feature precompute, and the graph
pipelines — receives its image as an `ImageContainer` and calls `merge()` to obtain a
processed array. The class holds one or more channels, applies a lazy preprocessing
chain to each, and reduces them to a single array on demand.

## Structure: channels and the constructor

An `ImageContainer` wraps a list of `_SingleChannel` objects. The constructor accepts
either a single `Path` (a mono-channel image) or a nested list that describes how to
compose several files:

- `ImageContainer(path, config)` — one channel.
- `ImageContainer([c1, c2], config)` — two channels kept separate.
- `ImageContainer([[c1, c2], c3], config)` — a **summed group** `c1 + c2` reduced to one
  channel via `_sum_channels`, plus a separate channel `c3`.

Items in the structure may be `Path`s or other `ImageContainer`s. A group written as an
inner list is summed and contrast-stretched into one channel at construction time
(`_sum_channels` → `_sum_and_stretch`); items left at the top level stay as separate
channels. `container_a + container_b` (`__add__`) is sugar for
`ImageContainer([container_a, container_b], config)`, concatenating their channels.

`name` builds a descriptive composite label from the source file names, parsing the
`_C<n>_` channel tags and the trailing `CY5,DAPI`-style suffix so a multi-channel
container reports something like `MAX_CET145_CY5+FITC,DAPI`.

## Preprocessing chain (`_SingleChannel`)

Each channel loads and preprocesses lazily; nothing touches disk until a pixel array is
requested, and every stage is cached. The chain, in order:

1. **Lazy load** (`image_16bit`) — reads the TIFF as `uint16`. If `correct_DIC_shift` is
   set in the config and the file name contains `DIC`, the image is translated by the
   given `[dy, dx]` shift (per z-slice for 3D). This registers the DIC channel against
   the fluorescence channels, which are offset on the microscope.
2. **Percentile clipping** (`_get_high_contrast_16bit`) — clips to the
   `outlier_percentile` / `100 - outlier_percentile` range and rescales to full 16-bit,
   so a few hot pixels do not dominate the contrast.
3. **Quantization** — `image_8bit` divides the high-contrast 16-bit image by 257 to
   `uint8`. The `quantization` config key (`"8bit"` / `"16bit"`) selects which depth
   downstream steps use.
4. **Resizing** (`resized_8bit` / `resized_16bit`) — when `resize_image` is true, the
   longest edge is scaled to `max_dim` and `scale_factor` is recorded so prompt
   coordinates and masks can be mapped back to the original resolution.

`get_image_for_processing` returns the array at the configured depth and resolution, and
is what `merge()` reads. `set_processed_image` injects an already-computed array (used
by `_sum_channels` to store a summed channel).

## Channel reduction: `merge()` and `_sum_and_stretch`

`merge()` reduces the container's channels to a single NumPy array. It **combines**
channels; it does not reduce them unless they were grouped for summing in the
constructor.

- **2D, one channel** → `(H, W)`.
- **2D, more than one channel** → `(H, W, C)` via `cv2.merge`.
- **3D z-stack** (auto-detected) → `(Z, H, W, 3)` via `_merge_3d`.

`_sum_and_stretch(images, dtype)` is the shared reduction: it sums the images in
`float64` (so saturated inputs cannot overflow), then min-max stretches the result to
the full range of `dtype`. Summing and averaging are interchangeable here because the
min-max stretch is invariant to a positive scale factor; the stretch is the part that
matters, since combining channels compresses everything toward mid-range and the stretch
restores contrast. Callers pass channels that have already been percentile-clipped, so
the min/max are not driven by hot pixels.

To get a single 2D channel out of several, group them in the constructor
(`ImageContainer([[c1, c2]], config)`), which sums them at build time; `merge()` then
returns that one channel as `(H, W)`.

### Mapping channels to the 3-channel RGB that microSAM expects

The microSAM ViT encoder expects a 3-channel RGB image. `_merge_3d` maps a z-stack to
`(Z, H, W, 3)` by channel count:

- **1 channel** → replicated across all three.
- **2 channels** → depends on `two_channel_merge_mode` (see the note below). The default
  `average_replicate` sums + stretches the two channels and replicates the result ×3.
  `passthrough` stacks them as `(Z, H, W, 2)` and leaves the third channel for micro_sam.
- **3 channels** → used as-is.
- **4+ channels** → channels 1 and 2 unchanged, channels 3..N summed + stretched into
  channel 3.

For 2D, `merge()` returns `(H, W, 2)` for a two-channel container and hands that to
micro_sam directly; micro_sam is then responsible for producing the third channel.

> **Note — averaging vs. padding the missing channel (microSAM 2-channel input).**
> Two strategies exist for turning a 2-channel image into the 3-channel RGB the encoder
> needs, and this codebase currently uses different ones in different places:
>
> - **Average-and-replicate** — `_merge_3d`'s default (`average_replicate`) and the
>   visual-feature precompute (`scene_graph_network/precompute_microsam_feats._prepare_image`)
>   both average the available channels and replicate the mean across all three RGB
>   channels. This was chosen on the strength of the microSAM paper, which reports that
>   averaging and replicating preserves the encoder's expected intensity distribution
>   better than zero-padding an empty channel.
> - **Zero-padding** — the microSAM library itself, when it segments a 2-channel image,
>   pads a zero third (blue) channel rather than averaging. `_merge_3d`'s `passthrough`
>   mode reproduces this by handing micro_sam a 2-channel array.
>
> These disagree. The paper's recommendation (averaging) and the library's actual
> behaviour (padding) are not the same, and **the microSAM authors have since confirmed
> that padding is now the recommended approach.** Anyone revisiting the visual-feature
> channel handling should prefer padding to match both the library and the current
> recommendation, and be aware that `_prepare_image` and `_merge_3d`'s default still
> average. This note is a coding record only; it is deliberately kept out of the thesis.

## Prompt generation

`generate_prompts()` produces prompts for microSAM's prompted mode from the container's
own DAPI channel, storing them in `self.prompts` / `self.prompt_type`. It finds the DAPI
channel automatically (`find_dapi_channel_file`, or a manually set `dapi_channel_index`),
picks a seed slice for 3D volumes (`seed_slice`, defaulting to the middle slice), and
delegates the work to `_PromptGeneratorHelper`.

`_PromptGeneratorHelper` first detects cell centers from the DAPI channel
(`_find_cell_centers`): Otsu threshold, small-object removal by `min_mask_area`, then a
distance-transform watershed that separates touching nuclei, taking the centroid of each
labelled region. It also estimates a `median_radius` from the watershed blobs. From
these it produces one of three prompt types, selected by `prompt_mode`:

- **`points`** — the DAPI centroids directly.
- **`bbox`** — either boxes from a seeded watershed over the target channels
  (`use_seeded_watershed_for_bbox`, default) or fixed-size boxes sized by
  `bbox_radius_multiplier × median_radius` around each centroid.
- **`mask`** — an instance mask from a seeded watershed, using the DAPI centroids as
  seeds over a 3-channel source (mono replicated, 2-channel padded with a zero blue
  channel, or the first three channels).

## Display

`display()` renders the merged image for inspection, mapping a 2-channel image to red and
green (with a zero blue channel) and showing 1- or 3-channel images directly.
`_normalize_for_display` scales to a float display range.

## Config keys used

`ImageContainer` reads its behaviour from the `preprocessing` and `prompting` blocks of
the config dict:

- `preprocessing`: `resize_image`, `max_dim`, `outlier_percentile`, `quantization`,
  `correct_DIC_shift`, `two_channel_merge_mode`.
- `prompting`: `prompt_mode`, `min_mask_area`, `use_seeded_watershed_for_bbox`,
  `bbox_radius_multiplier`.