# Cellpose vs. Omnipose in the `omnipose` conda environment

This note documents how Cellpose and Omnipose models coexist in the `omnipose`
conda environment, what their network outputs actually mean, and how those
outputs get turned into masks. It's based on tracing the installed package
source (`omnipose` 0.1.dev1, and its vendored `cellpose_omni` fork) rather
than the public docs, so line references point at the installed code:

```
/opt/miniconda3/envs/omnipose/lib/python3.10/site-packages/omnipose/
/opt/miniconda3/envs/omnipose/lib/python3.10/site-packages/cellpose_omni/
```

## 1. There is no separate "cellpose" package here

The `omnipose` conda env does not have a `cellpose` package installed
(`pip show cellpose` → not found). Instead, Omnipose vendors its own fork of
Cellpose as **`cellpose_omni`**, and both model families are loaded through
the exact same class: `cellpose_omni.models.CellposeModel`.

`cellpose_omni` uses one U-Net architecture (`resnet_torch.py`) for every
model — `cyto`, `cyto2`, `bact_phase_cp`, `bact_fluor_cp`, `bact_phase_omni`,
`bact_phase_affinity`, etc. What differs per model is:

- how many output channels the head has (`nclasses`)
- how the training targets were generated (Cellpose-style vs. Omnipose-style)
- which post-processing function turns raw network output into masks

That last point is a **runtime choice**, not a property baked into the
checkpoint — see §4.

## 2. Network output shape is shared

`CellposeModel.__init__` (`cellpose_omni/models.py:398`) sets `nclasses`
based on the model name:

| Model falls in...            | nclasses | Meaning of channels |
|---|---|---|
| default                      | `dim + 1` (2D → 3) | flow (2) + scalar field (1) |
| `BD_MODEL_NAMES` (boundary)  | `dim + 2` (2D → 4) | flow (2) + scalar field (1) + boundary field (1) |

`BD_MODEL_NAMES = C2_BD_MODELS + C1_BD_MODELS` (`models.py:45`), e.g.
`bact_phase_omni`, `bact_fluor_omni`, `plant_omni`.

So both Cellpose and Omnipose checkpoints emit the same tensor layout: a
flow field plus a scalar "mask strength" channel, optionally plus a
boundary channel. This is why the same `_run_cp` codepath can read out
`dP`, `cellprob`/`dist`, and `bd` regardless of which family the loaded
model belongs to (`models.py:1326-1337`).

### The flow field is predicted directly, not derived at inference time

It's tempting to assume `dP` is computed from the scalar field at
inference (e.g. `dP = gradient(dist)`), since that's how Omnipose's
training targets are built (see §3). It isn't — both are read straight out
of the network's own output channels, in the same forward pass
(`cellpose_omni/models.py:1326-1329`):

```python
if self.nclasses>1:
    cellprob[i] = yf[...,self.dim]      # scalar field always after the vector field output
    order = (self.dim,)+tuple([k for k in range(self.dim)])
    dP[:, i] = yf[...,:self.dim].transpose(order)
```

`yf` is the raw U-Net output; its first `dim` channels are unpacked as
`dP` and the channel right after them as `cellprob`/`dist`. No gradient
operation happens here — `dP` is just sliced out.

The gradient relationship exists only on the **training-target side**.
`omnipose/core.py:masks_to_flows` (§3, line 308) builds the *ground-truth*
flow field by first solving for a smooth ground-truth distance field from
the mask labels, then taking its gradient — that gradient-of-distance
field is what the network is trained to reproduce as `dP`. So the network
learns to predict "the gradient of a distance field" as a target, but at
eval time it outputs `dP` and `dist` as two independent sibling channels
from one pass, not one computed from the other.

## 3. Same channels, different meaning

Even though the tensors look identical, what they represent depends on how
the model was *trained*, which in turn depends on the `omni` flag used
during training (`omnipose/core.py:308`, `masks_to_flows` docstring):

> "First, we find the scalar field. In Omnipose, this is the distance
> field. In Cellpose, this is diffusion from center pixel."

| | Cellpose-trained (`_cp`, `cyto`, `cyto2`, `nuclei`) | Omnipose-trained (`_omni`, `bact_phase_affinity`) |
|---|---|---|
| Scalar channel | **Cell probability** (sigmoid logit, interior vs. exterior) | **Distance field** (distance to nearest boundary, via `edt` / eikonal relaxation) |
| Flow field | Gradient of a simulated **heat-diffusion field seeded at the mask centroid** — every interior pixel points toward one central attractor | Gradient of the **distance field** — follows the medial axis/skeleton of the cell, no single center point |
| Handles elongated / branched shapes | Poorly — a single centroid attractor merges or mis-splits non-star-convex cells | This is the whole point of Omnipose — works for filamentous/branching morphology |

This is directly relevant to *C. albicans* work, since hyphal/filamentous
morphology is exactly the non-star-convex case Cellpose's centroid flow
breaks on.

## 4. Mask reconstruction: two different algorithms, selected at eval() time

Regardless of which model is loaded, `CellposeModel.eval(..., omni=<bool>)`
picks the reconstruction algorithm via a **runtime parameter**, independent
of `self.omni` (the attribute set at model-load time from the checkpoint
name). Traced in `cellpose_omni/models.py`:

```python
# eval() forwards its own `omni` kwarg (default False) to _run_cp,
# NOT self.omni from __init__:
masks, ... = self._run_cp(x, ..., omni=omni, ...)   # models.py:1174-1203

# _run_cp branches purely on that local `omni` value:
if not (omni and OMNI_INSTALLED):
    masks, ... = dynamics.compute_masks(dP, cellprob, ...)       # Cellpose route
else:
    masks, ... = omnipose.core.compute_masks(dP, cellprob, ...)  # Omnipose route
```
(`models.py:1363-1405` for 3D, `models.py:1411-1458` for 2D)

### Cellpose route — `dynamics.compute_masks` (`cellpose_omni/dynamics.py:764`)

1. Foreground = `cellprob > mask_threshold`.
2. `follow_flows`: Euler-integrate every foreground pixel along the flow
   field for a **fixed** number of steps (default `niter=200`).
3. `get_masks`: histogram the final pixel positions, seed masks at
   histogram peaks, grow regions to include converged neighbors.
4. `remove_bad_flow_masks`: drop masks whose predicted-vs-recomputed flow
   error exceeds `flow_threshold`.
5. Remove small objects / fill holes.

### Omnipose route — `omnipose.core.compute_masks` (`omnipose/core.py:1300`)

1. Foreground = **hysteresis threshold on the distance field** (better for
   thin/tapering structures than a hard cutoff).
2. Iteration count is **dynamically scaled** to object size from the
   distance field (`niter=None` lets Omnipose pick, usually <20) instead of
   a fixed 200.
3. Endpoint grouping can use **DBSCAN clustering** (`cluster=True`) or a
   **pixel affinity graph** (`affinity_seg=True`) instead of histogram-peak
   seeding.
4. Optional boundary-field-based refinement (`boundary_seg=True`) and
   skeleton-spur removal (`despur=True`) to correctly split touching or
   non-convex cells.

The `omni` eval flag also switches image normalization
(`transforms.normalize_img(..., omni=omni)`) and, during training, which
function builds ground-truth targets (`dynamics.labels_to_flows` vs.
`omnipose.core.labels_to_flows`, `models.py:1624`).

### How `mask_threshold` drives the hysteresis window

`omnipose.core.compute_masks` doesn't expose separate low/high hysteresis
thresholds — `mask_threshold` **is** the high threshold, and the low
threshold is derived from it with a hardcoded offset of `1.0`
(`omnipose/core.py:1408-1414`):

```python
if (omni and SKIMAGE_ENABLED) or override:
    iscell = filters.apply_hysteresis_threshold(dist, mask_threshold - 1, mask_threshold)
else:
    iscell = dist > mask_threshold  # analog to original iscell = (cellprob > cellprob_threshold)
```

`skimage.filters.apply_hysteresis_threshold(image, low, high)` marks a pixel
as foreground if it's above `high` (a "seed"), **or** if it's above `low` and
connected to a seed pixel. Concretely, for a given `mask_threshold`:

- **Seed region**: distance-field pixels `> mask_threshold` — the
  confidently-foreground core.
- **Extended region**: pixels in `(mask_threshold - 1, mask_threshold]` that
  are connected to a seed pixel — lets thin, tapering parts of a cell (e.g. a
  hyphal tip that dips below `mask_threshold` right at its narrowest point)
  stay attached to the mask instead of getting clipped off, as a hard
  `dist > mask_threshold` cutoff would do.
- Raising or lowering `mask_threshold` slides **both** bounds by the same
  amount (since `low = mask_threshold - 1` always), which is exactly the
  "erode or dilate masks with higher or lower values" behavior already
  called out in the notebooks' inline `mask_threshold` comment — the whole
  hysteresis window moves up or down together.
- **The window width itself is not tunable via `mask_threshold`** — it's
  fixed at `1.0` in whatever units the scalar field happens to be in (pixels
  of distance-to-boundary, for a genuine Omnipose distance field). Changing
  `mask_threshold` repositions the window; it can't widen or narrow how much
  "low-confidence but connected" tissue gets pulled in. That's only
  adjustable by editing the `- 1` in the Omnipose source itself.
- This also means running the Omnipose route (`omni=True`) on a
  **Cellpose-trained** model (§6) applies this same fixed 1.0-wide hysteresis
  window to a cellprob logit rather than a pixel-distance field — the offset
  no longer corresponds to "1 pixel of erosion tolerance," which is part of
  why that combination is "not advised" rather than simply unsupported.
- This whole branch only runs when `iscell` isn't already supplied. Custom
  reconstruction code that computes `iscell` itself and passes it in — e.g.
  `7_omnipose_3d.ipynb`'s `reconstruct_masks_with_rf_cellprob`, which builds
  `iscell = rf_cellprob > mask_threshold` (a **plain**, non-hysteresis
  threshold) before calling `compute_masks` — bypasses this hysteresis logic
  entirely, even though it reuses the same `mask_threshold` name and even
  though `omni=True` is passed for the flow/mask step.

## 5. What happens if `omni` is omitted from `params`

If the `'omni'` key is left out of (or commented out of) the `params` dict
passed to `model.eval(imgs, **params)`, Python falls back to the `eval()`
signature default:

```python
def eval(self, x, ..., omni=False, calc_trace=False, ...):   # models.py:559
```

This default is **not** derived from `self.omni` or the loaded checkpoint —
it's just a fixed `False`. Concretely, that means:

1. **Mask reconstruction silently switches to the Cellpose route**
   (`dynamics.compute_masks`, `models.py:1411`), for *every* model type,
   including Omnipose-trained ones. No error or warning is raised — the
   distance-field output of an Omnipose model just gets treated as if it
   were a cellprob map.
2. **`niter=None` resolves differently.** With `omni=False`,
   `niter = 200 if (do_3D and not resample) else (1/rescale*200)`
   (`models.py:1359-1360`) is used instead of Omnipose's dynamic,
   distance-field-based iteration count.
3. **Image normalization reverts** to the non-Omnipose percentile
   normalization (`transforms.normalize_img(..., omni=False)`).
4. **The Omnipose-only knobs become dead weight**: `cluster`,
   `affinity_seg`, `boundary_seg`, `despur`, `hdbscan` are never passed to
   `dynamics.compute_masks` (its signature doesn't have them), so they're
   silently ignored rather than erroring.

Practical implications:

- For any Omnipose-trained model (`bact_phase_omni`, `bact_phase_affinity`,
  `plant_omni`, etc.), omitting `omni` silently runs the **mismatched**
  Cellpose reconstruction on a distance-field output. Expect degraded
  masks, especially on elongated/filamentous cells, since none of the
  mechanisms Omnipose adds to handle those shapes (hysteresis thresholding,
  dynamic niter, clustering/affinity grouping) run at all.
- Rule of thumb: always set `omni` explicitly to match the loaded model's
  family. The notebook now does this everywhere — explicitly in the main
  eval cell, and automatically via the model registry in the "run all
  models" section (§9) — so this footgun no longer applies in practice, but
  keep it in mind when adding new `eval()` calls.

## 6. `bact_phase_affinity`

`bact_phase_affinity` is an **Omnipose** model (not Cellpose), and is
currently the default bacterial phase-contrast model
(`cellpose_omni/gui/__init__.py:27`, `DEFAULT_MODEL = 'bact_phase_affinity'`),
superseding the older `bact_phase_omni`.

Model registry (`omnipose/core.py:28-49`):

```python
C2_BD_MODELS = ['bact_phase_omni', 'bact_fluor_omni', ...]   # 2-channel input + boundary field
C2_MODELS    = ['bact_phase_cp', 'bact_fluor_cp', ...]         # 2-channel input, Cellpose-trained
C1_BD_MODELS = ['plant_omni']                                   # 1-channel + boundary field
C1_MODELS    = ['bact_phase_affinity']                           # "the affinity seg models"
```

It's single-channel (`nchan=1`), has no boundary-field output
(`nclasses=2`, since it's not in `BD_MODEL_NAMES`), and is paired with
Omnipose's newer **`affinity_seg`** reconstruction mode: instead of
Euler-integrating to convergence points and clustering endpoints, it builds
a pixel **affinity graph** directly from the flow/distance field and
extracts masks via connected components on that graph, skipping the
Euler-suppression step (`suppress = omni and not affinity_seg`,
`omnipose/core.py:1429`).

The notebook currently sets `'affinity_seg': False` (commented "new
feature, stay tuned..."). If `bact_phase_affinity` is loaded with
`affinity_seg` left `False`, it still runs through the standard
distance-field/clustering Omnipose path rather than the affinity-graph path
it was designed for.

## 7. Notebook refactor: `ChannelImage`/`CompositeImage` → `ImageContainer`

`Qualitative Evaluation omnipose.ipynb` originally loaded images through
`prompt_generation.ChannelImage`/`CompositeImage`, which no longer exist as
standalone modules — that file-I/O logic was consolidated into the installed
`image_processing_tools` package. The notebook was refactored to use
`image_processing_tools.image_class.image_container.ImageContainer` instead,
which is a drop-in replacement for the old classes: same `+` composition
operator, same lazy per-channel loading. The main API differences that
required call-site changes:

- `composite.get_channels()` → `composite.channels` (plain attribute, no
  longer a method)
- No manual `cv2.merge([ch.resized_8bit for ch in channels])` — `ImageContainer.merge()`
  does this natively (and returns 2D grayscale for a single channel rather
  than converting to 3-channel BGR, unlike the old `ChannelImage` path)
- Per-channel `.path` moved from the container itself onto `container.channels[i].path`

Two latent bugs surfaced once the pipeline actually ran end-to-end after the
refactor (both pre-existed the refactor, they just hadn't been exercised):

1. **`_SingleChannel.image_16bit` crashed on any config missing
   `correct_DIC_shift`** (`image_container.py:43`): `self.proc_config.get("correct_DIC_shift")`
   had no default, so `sum(None)` raised unconditionally (Python evaluates
   the left side of `and` regardless of the right side). This affected
   *every* caller of `ImageContainer` that doesn't set that key — including
   `micro_sam_config.json`, not just this notebook. Fixed in the shared
   library: `self.proc_config.get("correct_DIC_shift") or (0, 0)`.
2. **`prepare_analysis_image` returned a `(image, scale_factor)` tuple that
   no call site ever unpacked** — `imgs = [prepare_analysis_image(i) for i in composite_images]`
   fed a list of tuples straight into `model.eval()`, which crashed on
   `x[i].squeeze()`. `scale_factor` (the resize ratio `max_dim / longest_edge`,
   used elsewhere in `micro_sam_pipeline.py` to map prompt/mask coordinates
   back to original resolution) was never actually consumed anywhere in this
   notebook, so the fix was to simply return the image directly.

## 8. Visualization: why `plot.show_segmentation` couldn't save, and the fix

The original visualization cells called `cellpose_omni.plot.show_segmentation(...)`,
which renders its own internal figure rather than returning axes a caller can
hand to `plt.savefig()` — that's why nothing ever reached disk, and why
several increasingly complex workaround cells (manually re-deriving outlines,
overlays, colorized labels) were needed just to approximate what
`show_segmentation` already draws.

The fix follows the pattern already working in `3. GNN/7_omnipose_3d.ipynb`
(cell that iterates `model_names`, builds `plt.subplots(n_models, 3, ...)`,
and calls `imshow` + `savefig` directly): skip `show_segmentation` entirely
and build the figure by hand from the raw `masks`/`flows` arrays. The
notebook's `flows` list layout (`models.py:1210-1213`) is:

```
flows[0] = RGB flow visualization
flows[1] = dP (flow field components)
flows[2] = cellprob / distance field
flows[3] = p (pixel coordinates after Euler integration)
flows[4] = bd (boundary field)
flows[5] = tr (pixel trajectories, if calc_trace)
flows[6] = affinity graph
flows[7] = bounds (binary boundary map)
```

The per-composite figure (one row per image, 4 columns) uses:
1. `_composite_to_rgb(img)` + `imshow(masks, cmap='nipy_spectral', alpha=0.4)` — image + mask overlay
2. `flows[idx][0]` — flow field
3. `flows[idx][2]` — cell probability / distance field
4. `flows[idx][2] > mask_threshold` — thresholded cell probability, using
   `params['mask_threshold']` (the value actually driving mask reconstruction
   for that run, not the function's raw hardcoded default of `0.0`)

Saved via plain `plt.savefig(...); plt.close(fig)`, filename and suptitle
both include the model name.

## 9. Auto-detecting `omni` and channel count from the model name

The "run every model on one image" section needs, per model, (a) whether to
pass `omni=True/False` to `eval()`, and (b) whether the model expects a
mono-channel or 2-channel image. Both are derived from `cellpose_omni.models`'s
own registry constants rather than string-matching the model name:

```python
from cellpose_omni.models import CP_MODELS, C2_MODELS, C2_BD_MODELS, C1_BD_MODELS, C1_MODELS, C2_MODEL_NAMES

CELLPOSE_MODEL_NAMES = set(CP_MODELS + C2_MODELS)                       # Cellpose-trained
OMNIPOSE_MODEL_NAMES = set(C2_BD_MODELS + C1_BD_MODELS + C1_MODELS)     # Omnipose-trained

is_omni  = lambda name: name in OMNIPOSE_MODEL_NAMES
is_mono  = lambda name: name not in C2_MODEL_NAMES   # nchan only becomes 2
                                                       # when model_type is in
                                                       # C2_MODEL_NAMES (models.py:458)
```

This matters because the obvious shortcut — `'omni' in model_name` — is what
`CellposeModel.__init__` itself uses as a *default* (`models.py:478`), but
it's wrong for `bact_phase_affinity`, which is Omnipose-trained with no
"omni" substring in its name. The registry-based check gets all 15 known
models right with no overlap (verified empirically — see §10 table).

**Mono-channel models get a summed, rescaled image, not the raw composite.**
Feeding a 2-channel `(H, W, 2)` composite into an `nchan=1` model
(`bact_phase_affinity` in this notebook's list) doesn't error immediately —
`transforms.convert_image` truncates to the first `nchan` channels
(`transforms.py:520-524`) — but produced a degenerate result (a progress bar
counting up to the image's pixel dimension instead of a normal small tile
count, and `masks` coming back as an empty list rather than an array),
consistent with the reshape logic ending up in `core.py`'s z-stack code path
(`core.py:592-616`, `if imgi.ndim==4 and self.dim==2`) meant for genuine 3D
stacks run through a 2D model. The fix: sum the two channels and rescale back
to the input dtype's range before feeding mono-channel models, mirroring the
exact normalization `ImageContainer._sum_channels` already uses:

```python
def _sum_channels_rescaled(img):
    summed = img.astype(np.float64).sum(axis=-1)
    min_val, max_val = summed.min(), summed.max()
    d_info = np.iinfo(img.dtype)
    return ((summed - min_val) / (max_val - min_val) * d_info.max).astype(img.dtype)
```

`plant_omni` is deliberately excluded from the "run every model" list — it's
a native 3D model (`dim=3`, whole z-stack input, ~20GB VRAM per
`7_omnipose_3d.ipynb`'s own notes), not comparable to a single-2D-image eval
loop.

**The reverse mismatch — mono image into an `nchan=2` model — is handled
gracefully, unlike the truncation case above.** `3. GNN/7_omnipose_3d.ipynb`'s
"run every model on one image" section feeds a single-channel DIC image
(`img = imgs[0]`, from a mono `ImageContainer.merge()`) to every model in its
list, including several `nchan=2` ones (`bact_phase_omni`, `bact_fluor_omni`,
`worm_*_omni`, `cyto2_omni`, `bact_phase_cp`, `bact_fluor_cp`, `cyto2`,
`cyto`). Unlike feeding *too many* channels (truncates — see above, and can
break for `nchan=1` models), feeding *too few* channels is the well-supported
path: `transforms.convert_image`'s `channels is None` branch pads with a
zero-filled channel to reach `nchan` (`transforms.py:530-533`):

```python
if x.shape[-1] < nchan:
    x = np.concatenate((x,
                        np.tile(np.zeros_like(x), (1,1,nchan-1))),
                        axis=-1)
```

So a mono `(H, W)` image becomes `(H, W, nchan)` with the real image in
channel 0 and the rest zero-filled — exactly the standard Cellpose/Omnipose
"grayscale" convention already referenced elsewhere in these notebooks via
the `channels=[0,0]` comment ("always define this if using older models,
e.g. `[0,0]` with `bact_phase_omni`"). This is why
`7_omnipose_3d.ipynb`'s all-models loop runs without error despite the
channel-count mismatch, while the *opposite* mismatch in this notebook
(§9 above) produced a broken result and required the explicit
`_sum_channels_rescaled` fix instead of relying on `convert_image` to cope on
its own.

## 10. Complete model reference

All 15 models known to `cellpose_omni.models.MODEL_NAMES`, classified by the
registry constants in `cellpose_omni/models.py` (`CP_MODELS`, `C2_MODELS`,
`C2_BD_MODELS`, `C1_BD_MODELS`, `C1_MODELS`) — verified to partition cleanly
with no overlap and no gaps:

| Model | Family | Channels |
|---|---|---|
| `cyto` | Cellpose | 2 |
| `nuclei` | Cellpose | 2 |
| `cyto2` | Cellpose | 2 |
| `bact_phase_cp` | Cellpose | 2 |
| `bact_fluor_cp` | Cellpose | 2 |
| `plant_cp` | Cellpose | 2 |
| `worm_cp` | Cellpose | 2 |
| `bact_phase_omni` | Omnipose | 2 |
| `bact_fluor_omni` | Omnipose | 2 |
| `worm_omni` | Omnipose | 2 |
| `worm_bact_omni` | Omnipose | 2 |
| `worm_high_res_omni` | Omnipose | 2 |
| `cyto2_omni` | Omnipose | 2 |
| `plant_omni` | Omnipose | 1 (mono) — also 3D-only, see §9 |
| `bact_phase_affinity` | Omnipose | 1 (mono) |

## Summary table

| Model | Family | `nclasses` | Scalar channel meaning | Intended reconstruction |
|---|---|---|---|---|
| `cyto`, `cyto2`, `nuclei` | Cellpose | 3 | cellprob | `dynamics.compute_masks` (`omni=False`) |
| `bact_phase_cp`, `bact_fluor_cp` | Cellpose | 3 | cellprob | `dynamics.compute_masks` (`omni=False`) |
| `bact_phase_omni`, `bact_fluor_omni`, `cyto2_omni`, `worm_omni` | Omnipose | 4 (has boundary field) | distance field | `omnipose.core.compute_masks` (`omni=True`) |
| `plant_omni` | Omnipose | 4 | distance field | `omnipose.core.compute_masks` (`omni=True`) |
| `bact_phase_affinity` | Omnipose | 3 (no boundary field) | distance field | `omnipose.core.compute_masks` with `affinity_seg=True` |