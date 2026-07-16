# GCN Data Flow

![[Media/GCN Design 2026-03-19 17.12.45.excalidraw]]
> What features are created and why. Also about normalization
additional node features: color moments?

> **Scope — mixed; the nuclei-specific parts are historical.** The **nuclei** pipeline described here (node = nucleus, fully-connected candidate edges, manual labels, 6 node / 6 edge features) is **no longer run** — the live pipeline makes each node an **AIS cell-fragment mask** and learns to merge oversegmented fragments (kNN candidate edges by boundary distance, per-cell MST labels, mask-bbox visual branch): see [Cell Mask Graph Data Flow](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Cell%20Mask%20Graph%20Data%20Flow.md).
>
> Read this page accordingly:
> - **Historical** — [Node features](#Node%20features) and [Edge features](#Edge%20features) (the nuclei schema; the fragment schema supersedes it) and the `max_edge_length_neg` bullet under [Data preprocessing](#Data%20preprocessing) (replaced by `dist_cap_factor` at kNN build time).
> - **Shared and live** — the normalization rationale (global z-score, angles ÷ `π/2`), self-loops, `T.ToUndirected()`, symmetry enforcement, [Dataset persistence](#Dataset%20persistence), [Train-test split](#Train-test%20split), [Batching](#Batching) and [Pre-training data preparation](#Pre-training%20data%20preparation). These apply unchanged to the fragment pipeline.
>
> Full breakdown: [Nuclei vs. cell-fragment](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Cell%20Mask%20Graph%20Data%20Flow.md#Nuclei%20vs.%20cell-fragment%20—%20what%20carries%20over). For *why* nuclei were tried first and what preceded them, see [Approach History](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Approach%20History.md).

## Source images — which image feeds what

Three different images are in play, and only one of them drives the handcrafted features. Keeping them straight matters: `data.image` is a channel stack, and it is easy to assume the features are read from it.

| Image | Built by | Used for |
| --- | --- | --- |
| **DAPI, single channel** | `ImageContainer(dapi_files, config).merge()` — one channel, so `merge()` returns `(H, W)` | **Nuclei segmentation and every intensity feature** — node `average_intensity`, edge `average_intensity` |
| **DAPI + DIC, 2 channels** | `dapi + dic` → `ImageContainer.__add__` → `merge()` → `(H, W, 2)` | **Display only** — stored as `data.image` for the prediction overlays, where `plot_edge_predictions` composites DAPI in blue over DIC in grayscale. **Never read by any feature.** |
| **MicroSAM embeddings** | `compute_microsam_features` → `(256, H_f, W_f)` | The [visual branch](#Visual%20features%20from%20MicroSAM) (RoIAlign), not the tabular features |

**The DAPI channel is contrast-normalized before anything reads it.** The config passed to `ImageContainer` sets `outlier_percentile: 0.35`, `quantization: "16bit"` and `resize_image: False`, which routes `get_image_for_processing()` to `_get_high_contrast_16bit()`: each channel is percentile-clipped at 0.35 / 99.65 and stretched to the full `0..65535` range. So the intensity features read a clipped, full-range channel, not raw sensor values — hot pixels cannot set the scale, and per-image brightness differences do not leak into the features. `resize_image: False` also keeps the image pixel-aligned with the label maps.

> The fragment pipeline reads a **summed** intensity channel rather than DAPI alone, but the normalization is identical — see [Which channel the intensity features read](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Cell%20Mask%20Graph%20Data%20Flow.md#Which%20channel%20the%20intensity%20features%20read).

## Node features

Node features define the morphological and visual characteristics of individual nuclei detected in the DAPI image.

**What they are:**
- **Average Intensity:** The mean pixel value within the nucleus mask. Indicates DAPI concentration.
- **Eccentricity & Circularity:** Geometric measures of how elongated or round the nucleus is. Hyphal cells tend to be highly elongated compared to yeast-form cells or artifacts.
- **Area & Perimeter:** Absolute size metrics in number of pixels
- **Major/Minor Axis Length:** The length and width of the bounding ellipse. Crucial for understanding the physical span and orientation of the cell.

**How they are normalized:**
- **Global Z-Score Normalization:** Node features (like Area and Intensity) are normalized using the global mean and standard deviation calculated *strictly from the training set*. 
- **Why NOT Within-Graph Normalization:** We deliberately avoid scaling features relative to the in-graph mean (e.g., `area / graph_mean_area`). Because our graphs are extremely small (3 to 6 nodes), a single outlier (like a huge merged artifact) would drastically warp the graph mean. This "Small N Variance" would cause identically sized true hyphae to have completely different scaled feature values depending on their neighbors. Global Z-scoring provides a stable, reliable mathematical anchor.

## Edge features

Edge features define the physical path and geometric relationship between two candidate nuclei. They act as the visual and structural evidence of a hyphal connection.

**What they are:**
- **Average Intensity:** Sampled via a `profile_line` drawn between the two centroids (with the raw nuclei pixels masked out to zero). This directly measures the presence of a fluorescent "bridge."
- **Length:** The Euclidean distance between the two cell centroids.
- **Angle Differences (`node1_angle_diff`, `node2_angle_diff`, `min_diff_angle`, `relative_angle`):** The angular deviation between the major axis of the cell and the proposed connection path.

**How they are normalized (The "Why"):**
- **Path Length (Biological Normalization):** Instead of Z-scoring, raw pixel distance is divided by the `avg_nucleus_length` found in the image. **Why:** This converts arbitrary pixels into a universal biological metric (e.g., "this gap is 1.5 nuclei long"). It perfectly standardizes distances across different microscope magnifications without warping the data statistically.
- **Angles (Geometric Normalization):** Angles are normalized strictly by dividing by $\pi/2$. **Why:** This bounds the angles between `0.0` (perfectly aligned) and `1.0` (perfectly perpendicular). Z-scoring would center the mean at 0 and stretch the variance, which mathematically destroys the bounded, physical meaning of an angle.
- **Intensity (Statistical Normalization):** Only the raw edge intensity receives standard Z-score normalization alongside the node features.

## Data preprocessing

- **Self-Loops (Omitted):** Self-loops (edges pointing from a node to itself) are deliberately not added to the graph connectivity matrix. **Why:** In early testing, introducing self-loops degraded model performance. The network likely struggled to differentiate between the internal feature updates and the crucial inter-cellular connection features, blurring the distinct role of the `EdgeUpdater`.
- **Undirected Graph Construction:** The raw candidate edges are mapped into a bidirectional format (using `T.ToUndirected()`). **Why:** Hyphal connections in static DAPI images do not have an inherent directional flow (causality). If nuclei A is connected to nuclei B, then nuclei B is physically connected to nuclei A. Forcing bidirectional information flow allows the network to evaluate the connection from both perspectives simultaneously.
- **Symmetry Enforcement:** During inference and loss calculation, the output predictions for $P(A \rightarrow B)$ and $P(B \rightarrow A)$ are averaged. This acts as an "AND" constraint, forcing the model to learn robust, viewpoint-agnostic representations of true edges.
- **Long-edge trimming (negative-only):** `create_pyg_data` accepts a `max_edge_length_neg` threshold (configurable, expressed in `avg_nucleus_length` units). Negative candidate edges whose normalized `length` exceeds the threshold are dropped before the graph is made undirected. Positive edges are retained regardless of length so no ground-truth connection is ever lost to trimming. **Why:** The candidate graph is fully connected, so long-range negatives explode quadratically with node count but carry little signal — the chain-like biology means plausible connections are short. Trimming sparsifies the graph before training without touching the learning targets and without requiring any threshold on positives.

## Visual features from MicroSAM

Visual context from the raw image is injected through a separate branch in the model (see [Visual branch](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Design%20Choices.md#Visual%20branch)). The data pipeline's job is to (a) persist the encoder output to disk so the training environment does not need the MicroSAM conda env, and (b) attach it to each PyG graph so it batches cleanly. For a step-by-step diagram with example tensor shapes, see [GCN Visual Feature Data Flow](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Visual%20Feature%20Data%20Flow.md).

- **Precomputation:** `precompute_microsam_feats.compute_microsam_features` tiles the image, runs the MicroSAM encoder on each tile, stitches the per-tile feature grids into a single `(256, H_f, W_f)` feature map, and records the `pixels_per_feature` scale factor. The output is saved as `<image_name>_microsam_features.npz` next to the source image. See the existing notebook `Get microsam Features.ipynb` for the end-to-end call.
  - **Reusing a saved embedding store:** if the AIS run already wrote a tiled embedding zarr, `precompute_microsam_feats.load_and_stitch_saved_embeddings(embedding_path)` stitches that store into the same `(256, H_f, W_f)` map + `pixels_per_feature` without rerunning the encoder (`micro_sam` is imported lazily, so this path needs only zarr + nifty + torch). See [Cell Mask Graph Data Flow](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Cell%20Mask%20Graph%20Data%20Flow.md).
  - Note: DIC images have a shift in pixel dimensions due to additional lenses. The shift is documented [in this note](C_Albicans%20Thesis%20Project/5.%20Results/3.%20Data%20Exploration/Scene%20Graph%20Data%20Generation%202026-01-30.md#DIC%20focus). The shift must be corrected before using the DIC image to extract features.
- **Attachment at graph-build time:** When `microsam_paths_list` is passed to `create_pyg_data`, each graph receives:
  - `data.microsam_embedding` — `(256, H_f, W_f)` float tensor
  - `data.pixels_per_feature` — length-1 float tensor; `1 / pixels_per_feature` is the RoIAlign `spatial_scale`
  - `data.centroids` — `(num_nodes, 2)` float tensor in `(y, x)` pixel coordinates (promoted from the legacy list-of-tuples form so PyG can concatenate centroids across a batch, which the visual branch needs for RoIAlign)
- **Custom `Data` subclass (`MicrosamData`):** The feature map has no natural node/edge dimension to concatenate along, so the default PyG collation would break. `MicrosamData` overrides `__cat_dim__` to return `None` for `microsam_embedding`, which tells PyG to `torch.stack` instead — producing `(B, 256, H_f, W_f)`, the exact NCHW layout torchvision's `roi_align` expects. `__inc__` returns `0` to keep the feature map from being offset like an index. The stacking strategy assumes all graphs in a batch share `(H_f, W_f)`; for heterogeneous image sizes a future version can switch to a list-valued attribute.
- **Spatial scale consistency:** Per-graph `pixels_per_feature` values are stacked into `(B,)` by default batching. The model asserts they are uniform across a batch so a single `spatial_scale` can drive `roi_align`. Same-size same-tile-config images satisfy this automatically.

## Dataset persistence

- **On-disk format:** `save_pyg_dataset(data_list, root)` writes the output of `create_pyg_data` to `root/processed/data.pt` via a thin `InMemoryDataset` subclass (`DapiTracingDataset`). `load_pyg_dataset(root, transform=None)` reads it back. **Why:** Labeling is currently manual in the notebook, and re-running `create_pyg_data` plus relabeling on every kernel restart is wasteful. Persisting a PyG `InMemoryDataset` (rather than a loose `torch.save` of a list) also gives us `pre_transform`/`transform` hooks, integer/slice/mask indexing, and native `DataLoader` integration.
- **`transform` as the augmentation slot:** The `transform` argument to `load_pyg_dataset` is applied per `__getitem__` call, so stochastic augmentations re-roll every epoch. `pre_transform`/`pre_filter` are reserved for deterministic preprocessing that should be cached alongside the data.
- **Non-tensor attributes:** `data.image` (numpy array) rides along through `InMemoryDataset.collate` as a per-graph non-tensor attribute. `data.centroids` is stored as a `(num_nodes, 2)` float tensor (not a list of tuples) so PyG can concatenate across a batch — this is required by the [Visual branch](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Design%20Choices.md#Visual%20branch) for RoIAlign, and the visualization helper converts back to numpy when plotting a single graph. To support PyTorch's default `weights_only=True` safe-load, `gnn_data.py` allowlists `numpy._core.multiarray._reconstruct` at import time; without that, `torch.load` falls back to an unsafe load and emits a warning.
- **Normalization is deliberately not baked in:** Z-score normalization is still computed per-fold inside `n_fold_validation` to avoid train/test leakage ([Feature Normalization Flow](#Batching)). The saved dataset stores raw (un-normalized) features.

## Train-test split

- **Graph-Level Splitting:** During the N-fold cross-validation, the dataset is split entirely at the graph (image) level, rather than subsampling nodes or edges within a single graph.
- **Why:** Randomly masking/subsampling edges within a single graph for a test set allows the model to "cheat" by using the remaining graph structure to infer the missing links (transductive learning). By separating the data at the image level (inductive learning), we mathematically guarantee that the model evaluates entirely unseen biological structures, providing a true measure of real-world generalization.

## Batching

- **PyG Internal Batching:** PyTorch Geometric (`DataLoader`) does not stack graphs into a new dimension like standard image batches (e.g., `[batch_size, nodes, features]`). Instead, it concatenates multiple distinct graphs into one giant, disconnected graph. The `edge_index` matrices are diagonally stacked, ensuring messages only pass within the boundaries of the original subgraphs. **Why:** This allows the model to process variable-sized graphs in parallel without needing computationally wasteful padding (zeros).
- **Feature Normalization Flow:** 
  1. The Global Mean and Standard Deviation are calculated **only** on the batched PyG Data objects within the active *training fold*.
  2. These training statistics are then applied to normalize both the training batch and the testing batch.
  3. **Why:** This strict isolation mathematically prevents "Data Leakage." The model is never allowed to have statistical priors about the distribution of the test set before attempting to classify it.

## Pre-training data preparation

Two optimizations run once, after the dataset is loaded / after the train/test split and Z-score normalization, before the epoch loop starts:

- **Strip heavy attributes when the visual branch is off:** `StripHeavyAttrs` (defined in `gnn_data.py`) is applied eagerly in the notebook against the loaded `pyg_data_list` (e.g. `pyg_data_list = [strip(d) for d in pyg_data_list]`), deleting `microsam_embedding` and `pixels_per_feature` from each `Data` object in memory. The saved `data.pt` stays untouched. **Why:** with `use_visual_features=False`, the `(256, H_f, W_f)` feature map is dead weight — it still gets deep-copied per fold and pushed to GPU even though no layer reads it. Stripping it cuts per-graph memory and transfer cost to near zero. It is run eagerly on the list rather than as a PyG `transform=` hook because `pyg_data_list` is a plain Python list, not an `InMemoryDataset` instance, so the transform hook is not invoked on `__getitem__`. Keep `image` unless VRAM is tight, since `plot_edge_predictions` needs it for the TensorBoard overlays.
- **One-time move to GPU:** Inside `n_fold_validation` / `train_overfit_test`, after `_apply_feature_zscore`, each `Data` object is moved to the device via `data.to(device)` *before* the `DataLoader` is constructed. The loader then concatenates GPU-resident tensors, so no host→device copy runs during training. **Why:** the dataset is tiny (6 graphs) and fits comfortably in VRAM. Keeping it resident eliminates per-epoch CPU→GPU transfers of the `microsam_embedding` / `image` payloads, which otherwise dominate epoch time when the rest of the graph is small. Existing `data.to(device)` calls inside `train_model` / `test_model` become no-ops (PyG returns `self` when already on the target device). Trade-off: `pin_memory`, `non_blocking=True`, and `num_workers>0` all become irrelevant — workers can't fork CUDA tensors anyway, and the transfer they were designed to hide no longer happens.
  - If training on larger datasets and VRAM is limited, the image can be stripped as well from the data stored on VRAM.
