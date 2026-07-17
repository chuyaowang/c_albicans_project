# GCN Model Interpretation

> How and why the trained model is inspected to understand its internal reasoning.
> Both analyses are computed at the best early-stopping epoch and logged to TensorBoard under `Interpretation/`.

> **Scope — live, cell-fragment first.** The feature names and counts on this page are the **cell-fragment** schema (8 node / 10 edge → 26 tabular columns). The **nuclei** pipeline is historical, and the interpretation code is deliberately *not* backward-compatible with its 6 / 6 schema (18 columns) — older figures will show the historical layout. The two analyses themselves are pipeline-agnostic. Full breakdown: [Nuclei vs. cell-fragment](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Cell%20Mask%20Graph%20Data%20Flow.md#Nuclei%20vs.%20cell-fragment%20—%20what%20carries%20over).

The model exposes two complementary windows into its learned behaviour:

1. **Pre-logit embedding space** — where in a low-dimensional projection do TP, TN, FP, and FN edges land, and how well does the model's final representation separate them?
2. **Per-edge attribution heatmap** — which input features drove each individual edge prediction, and how do the visual and handcrafted streams compare?

## 1. Pre-logit embedding space

### Extraction point

- **How:** The [Classifier](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Design%20Choices.md#Classifier%20Head) MLP is split into a body (`CustomLazyLinear → LayerNorm → ReLU → Dropout`) and a head (`Linear(1) → Sigmoid`). The body output — a `hidden_channels`-dimensional vector per edge — is extracted with `return_embeddings=True` on `Model.forward()`. This is the highest-level representation the model builds before collapsing to a scalar probability. It already encodes the full two-hop context gathered by both GCN layers and both edge updaters.
- **Why:** The representation immediately before the final linear layer is the most informative single tensor for understanding whether the model has learned a separable internal geometry for the task. A model that generalises well should cluster TPs and TNs into distinct regions of this space even on held-out graphs it has never seen.

### PCA

- **How:** `sklearn.decomposition.PCA` reduces the embedding matrix to 2 components. If training embeddings are supplied, PCA is fit on the combined (test + training) matrix so both sets share the same coordinate system; otherwise it is fit on the test set alone. Test edges are plotted as filled circles with a solid black border; training edges (when provided) are plotted as open circles with a dashed colored border, using the same TP / TN / FP / FN color coding. Axis labels report the fraction of variance explained by each component.
- **Why:** PCA makes no assumptions about class structure. It reveals the dominant directions of variance in the embedding space irrespective of labels. Overlapping TP and FP clusters in PCA space indicate the model encodes similar representations for both, which is a signal that the embedding has not yet disentangled true from false connections. Overlaying the training distribution shows whether the test graph embeddings are well within the training distribution (expected for generalisation) or sit in a distinct region (a sign of distribution shift or overfitting).

### PLS-DA

- **How:** `sklearn.cross_decomposition.PLSRegression` is fit with the binary true labels (0 / 1) as the response. If training embeddings are supplied, the model is fit on the combined (test + training) matrix and labels; otherwise it is fit on the test set alone. Training edges are overlaid using the same open-dashed-circle style as in the PCA plot. The two latent variables (LVs) that maximally co-vary with the label are used as axes. Edges are scatter-plotted with the same TP / TN / FP / FN colour coding. The percentage of variance in the embedding matrix X captured by each LV is reported on the axis label. This is the standard PLS $R^2_X(k) = \|t_k p_k^\top\|_F^2 / \mathrm{SS}(X_c)$, computed by re-running the NIPALS deflation with the unit-norm weight vectors $w_k$ stored in `x_weights_`: $t_k = X_{\text{res}} w_k$, $p_k = X_{\text{res}}^\top t_k / \|t_k\|^2$, then $R^2_X(k) = \|t_k\|^2 \|p_k\|^2 / \mathrm{SS}(X_c)$, and $X_{\text{res}} \leftarrow X_{\text{res}} - t_k p_k^\top$. This formula is guaranteed $\leq 1$ per component and telescopes correctly across components.
- **Why:** PCA is blind to labels and may orient the principal components along directions unrelated to classification. PLS-DA explicitly rotates the embedding space to maximise the separation between the positive class (true hyphal connections) and the negative class. It therefore highlights discriminative structure that PCA might bury. A model that generalises well should show good TP / TN separation along LV1. The explained variance percentage contextualises how much of the overall embedding variance each LV direction accounts for — a low percentage means PLS-DA has found a narrow but label-discriminative slice of the embedding space.
- **Caveat:** With only 20–30 test edges per fold (fully connected 3–6-node graphs), both projections are based on very few points. Patterns should be interpreted qualitatively and cross-checked against the full fold metrics (AUC, F1) reported in TensorBoard scalars.

## 2. Per-edge attribution heatmap

### What is gradient × input attribution?

- **How:** For each edge prediction $e$, the model is run in `attribution_mode=True`, which detaches and re-attaches `requires_grad=True` to all tracked input tensors. A separate `backward()` pass is then performed for each edge's scalar prediction $\hat{p}_e$:
$$\text{attr}_{e,f} = \left| \frac{\partial \hat{p}_e}{\partial \text{input}_f} \cdot \text{input}_f \right|$$
  The result is the absolute value of the elementwise product of the gradient and the input value. Large values indicate that a small change in that input feature would strongly affect this edge's prediction, and that the feature is actually present (non-zero) in the input.
- **Why:** This is a first-order approximation of feature importance. It captures sensitivity through the full forward pass including both GCN message-passing layers and both edge updater layers. Because the model uses skip connections that feed the raw input back at every depth, the gradient of `data.x` flows through both the 1-hop and 2-hop message-passing pathways as well as directly through the residual path, giving a holistic attribution that accounts for all mechanisms by which the feature influences the prediction.

### Input taxonomy

Each edge $e = (i, j)$ has six groups of tracked inputs:

| Group | Source tensor | Dimensions | Meaning |
| --- | --- | --- | --- |
| `Node (src)` | `data.x[i]` | 8 | Tabular morphology of the source fragment |
| `Node (tgt)` | `data.x[j]` | 8 | Tabular morphology of the target fragment |
| `Edge` | `data.edge_attr[e]` | 10 | Tabular junction features between the two fragments |
| `Visual (src)` | `node_visual[i]` (CNN output) | `d_vis` | Per-latent attribution from the source fragment's mask-bbox patch |
| `Visual (tgt)` | `node_visual[j]` (CNN output) | `d_vis` | Per-latent attribution from the target fragment's mask-bbox patch |
| `Visual (edge)` | `edge_visual[e]` (CNN output) | `d_vis` | Per-latent attribution from the edge's bounding box patch |

The visual streams (`Visual (src/tgt/edge)`) are only present when `use_visual_features=True`. Each latent dimension is kept as its own column (`vs_0 … vs_{d-1}`, `vt_0 … vt_{d-1}`, `ve_0 … ve_{d-1}`) rather than being aggregated. This makes the total column count `26 + 3 × d_vis` (e.g., 122 for `d_vis=32`), enabling direct comparison of which microsam latent dimensions co-vary with which manual features. The tabular 26 = 8 source-node + 8 target-node + 10 edge features (the cell-fragment schema; the historical nuclei schema gave 6 + 6 + 6 = 18). Switching to `column_order='clustered'` (see [Column ordering](#Column%20ordering)) is particularly useful here: if visual and manual features cluster together, it indicates that the CNN has learned a representation correlated with interpretable morphological properties.

The tabular features in column order are:

- **Node features (8):** `circ` (circularity), `ecc` (eccentricity), `sol` (solidity), `area` (normalised area), `maj` (major axis), `min` (minor axis), `int` (interior DIC intensity), `ctx` (context-ring intensity).
- **Edge features (10):** `gap` (junction intensity), `dist` (normalised boundary distance), `ang1` / `ang2` (per-fragment angle diffs), `min_ang` (minimum angle diff), `rel_ang` (relative orientation), `contact` (contact fraction), `area_r` (area ratio), `collin` (axis collinearity), `cont` (intensity continuity).

> **Historical note.** The nuclei pipeline used 6 node features (`circ, ecc, area, int, maj, min`) and 6 edge features (`e_int, e_len, ang1, ang2, min_ang, rel_ang`) — hence the 18 tabular columns quoted in older figures. The fragment schema is a positional superset: edge columns 0–5 keep the same roles in the same order (`gap`≡`e_int`, `dist`≡`e_len`, then the four angles), which is why the trainer's normalisation contract needed no change. See [Cell Mask Graph Data Flow](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Cell%20Mask%20Graph%20Data%20Flow.md#Nuclei%20vs.%20cell-fragment%20—%20what%20carries%20over).

### Heatmap layout

The heatmap has three panels (left to right):

1. **Class strip:** a narrow colour bar per row indicating TP (green), TN (blue), FP (red), FN (orange).
2. **Probability column:** the symmetrized predicted probability displayed as a 'Greens' colour bar with the numerical value overlaid. The probability shown here is the value that determined the TP / TN / FP / FN classification — it uses `enforce_symmetric_predictions` before thresholding, matching the training objective.
3. **Attribution heatmap:** the per-edge, per-feature attribution matrix displayed with the `hot` colormap. Values are log-transformed (`log(1 + attr)`) and then row-normalised so each row sums to 1, making cell values directly comparable within each row as each feature's proportional contribution to that edge's prediction. Column order depends on `column_order` (see [Column ordering](#Column%20ordering)): by **default (`'grouped'`)** columns stay in fixed source order with each group contiguous, separated by vertical dividers under colored group headers. Under `'clustered'` the columns are globally Ward-clustered, so groups are no longer contiguous and membership is instead indicated by **colored x-tick labels** plus a legend patch inside the heatmap. White horizontal lines mark class boundaries in both modes.

### Row sort order

Edges are sorted into four class groups in the order TP → TN → FP → FN. Within each group:

- **TP ascending** by predicted probability — borderline true positives (just above the threshold) appear first; high-confidence correct positives appear last.
- **TN descending** by predicted probability — uncertain true negatives (closest to the threshold) appear first; confident correct negatives appear last.
- **FP ascending** — the most borderline false positives appear first.
- **FN ascending** — the most borderline false negatives appear first.

This layout places the "boundary" edges near the top of each class group and the clearly correct predictions at the bottom, making it easier to spot what features distinguish confident predictions from uncertain ones.

### Column ordering

Selected by the `column_order` parameter, which is threaded through `plot_attribution_heatmap` / `plot_combined_figure` and exposed as `interpret_column_order` on both `n_fold_validation` and `train_overfit_test`.

**`'grouped'` (default)** — columns stay in fixed source order, each of the six groups contiguous: `Node (src)` → `Node (tgt)` → `Edge` → `Visual (src)` → `Visual (tgt)` → `Visual (edge)`.

- **How:** `_group_columns` emits the columns in group order and returns the group spans; `_fill_heatmap_axes` draws a vertical divider at each span boundary and a colored header above each block.
- **Why:** the question this heatmap is usually asked is "which *stream* drove this prediction — the source fragment, the target fragment, the junction, or their visual counterparts?" A fixed, contiguous layout answers that directly and is stable across folds and runs, so heatmaps from different experiments can be compared position-by-position. Clustered order re-permutes on every run, which makes that comparison impossible.

**`'clustered'`** — columns are globally reordered by similarity of attribution.

- **How:** After row-normalisation, the columns are reordered by Ward hierarchical clustering (`scipy.cluster.hierarchy.linkage`, Euclidean metric) applied to the transposed attribution matrix. The dendrogram leaf order from `leaves_list` permutes the columns. Group membership is stored per-column before permutation and carried through, so the colored x-tick labels and legend remain correct after reordering.
- **Why:** With 100+ columns, fixed order makes it hard to spot correlations *between* streams. Clustering places similarly-attributed features adjacent. If a visual latent clusters next to a tabular feature (e.g., `vs_7` next to `src_area`), that is a hypothesis that the CNN learned a representation tracking that morphological property.
- **Caveat:** Ward clustering minimises within-cluster variance in the row-normalised attribution space. With only 20–30 edges per fold the clustering rests on a small sample and is exploratory, not a definitive feature grouping.

> Within-group ordering is never clustered in either mode — it always follows the feature schema order.

### Row normalisation

- **How:** Attribution values are first log-transformed (`log(1 + attr)`) to compress the heavy tail of large gradients, then divided by the row sum so that all values in each row sum to 1. A cell value of 0.3 means that feature accounts for 30% of the total log-attribution for that edge.
- **Why:** The goal is to compare *which features drove a given edge's prediction most strongly*. This is a within-row comparison. Scaling each row to sum to 1 makes the cell values directly interpretable as each feature's proportional contribution to that individual edge's classification. A per-column normalisation would instead answer a different question ("which edges relied most heavily on a given feature") and would prevent cross-feature comparison within a single edge.
- **Caveat:** Because log-transformation is applied before normalisation, cell values represent shares of *log-attribution* rather than raw attribution. The log scale suppresses extreme dominance by single features and keeps smaller contributions visible, which is the right trade-off for visual diagnosis. To inspect raw attribution magnitudes, use the `attr_matrix` returned by `compute_per_edge_attributions` directly.

## Design decisions

### Test graphs only

The model has been trained on the training fold graphs: their embeddings and attributions reflect memorised patterns, not generalisation. Only the held-out test graph reveals how the model performs on unseen data. Training-graph interpretation would be misleading as a model quality signal, though it can be added by calling the same functions on `train_dataset` if needed.

### Best epoch only

Both analyses are expensive (attribution requires one backward pass per edge). Running them at every epoch would multiply wall-clock time by the number of edges per fold. The best early-stopping epoch is the only model state that matters for evaluation, so it is the right time to inspect.

### `model.eval()` and deterministic embeddings

All interpretation passes use `model.eval()`, which disables Dropout. This is essential: stochastic masking would produce different embeddings and different gradients on each call, making the results non-reproducible and mixing attribution signal with dropout noise. `LayerNorm` and `GraphNorm` have no running statistics, so their behaviour is identical in train and eval mode.

### Model parameter gradient suppression

During the attribution backward loop, model parameters are temporarily frozen (`model.requires_grad_(False)`) to avoid allocating and computing gradients for parameters that are not used. Only the four tracked input tensors (`x`, `edge_attr`, `node_visual`, `edge_visual`) need gradients. This reduces memory allocation and backward-pass time proportionally to the number of model parameters.

## Heatmap edge sampling

The heatmap is **one row per directed edge**, so it grows with the candidate graph. On the 157-fragment graph (1388 directed edges) the full figure renders at **28,660 px tall** — technically complete, practically unreadable. `sample_heatmap_edges` produces a balanced subset, and the sampled version comes in at ~1,700 px.

- **Sampled by ground-truth label, not by TP/TN/FP/FN.** Up to `heatmap_sample_size` (default 15) edges with label 1 and 15 with label 0. Balancing on what the edge *is* keeps the panel representative while letting the classes it resolves into stay visible. If a class has fewer than 15 edges, all of them are taken.
- **Directed edges are sampled**, so both directions of one pair can appear. They are not redundant rows: attribution is asymmetric, since the `src_*` and `tgt_*` column blocks swap between the two directions.
- **Seeded** (`heatmap_seed`, default 0) via `np.random.default_rng`, so the same run always yields the same rows.
- **Both versions are logged.** The full heatmap remains the complete record; the sampled one is the readable view.

> **Attributions are computed once, over every edge, before sampling.** Both heatmaps are drawn from that same `attr_matrix`, so the sampled figure is a literal subset of the full one and the two cannot disagree. Sampling before the backward loop would have been ~46× faster (30 passes instead of 1388) but would have made the full heatmap unavailable.

`heatmap_sample_size` and `heatmap_seed` are parameters of both `n_fold_validation` and `train_overfit_test`, alongside `interpret_column_order`.

## Attention parallel coordinates

`Attention/Graph_<id>` plots each edge's two GCN-layer attention weights on two vertical axes, joined by a line and colored by TP / TN / FP / FN. Built with `pandas.plotting.parallel_coordinates`.

- **Why this and not the overlay text.** The `Predictions` overlay prints `A1 | A2` per edge, which was readable for a handful of nuclei and is not for hundreds of fragments. The parallel-coordinates view shows the same two numbers for every edge at once, and makes the *relationship* between the layers legible: whether attention shifts between layer 1 and layer 2, and whether that shift differs by class.
- **Draw order is deliberate.** TN is drawn first at `alpha=0.12`, then FN / FP / TP at `alpha=0.55`. Negatives dominate the candidate graph and would otherwise bury the classes worth looking at.
- **The CSV is the same data.** `attention_graph_<id>.csv` is written from the DataFrame that backs the figure, so downstream analysis and the plot cannot diverge.

## Predicted-probability violins

`Probabilities/Graph_<id>` shows the distribution of predicted probability as two violins — **true edges (label 1) against false edges (label 0)** — with the fold's F1-maximizing threshold drawn across them, per-group μ and σ annotated, and the individual edges jittered on top. Colors follow the convention already used for the training overlays: positive → TP-green, negative → TN-blue.

**Why grouped by true label rather than TP / TN / FP / FN.** TP and FN are both label-1 edges, separated *by the threshold*. Plotting them as separate violins would show one distribution sliced at the cut, with the slice line as the boundary — an artifact of the threshold, not a fact about the model. Grouping by what each edge *is* and *drawing* the threshold shows the two things that matter: how far apart the classes sit, and where the cut actually falls.

**This gives [§10](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#10.%20Saturated%20probabilities%20under%20leave-one-out%20CV%20(in%20progress)) a picture.** Saturation is currently tracked only as the scalar `Diag/Pred_Std_Test < 0.05`. A violin shows it directly: collapsed predictions render as flat lines rather than distributions, and a fold whose positives and negatives overlap is visible at a glance rather than inferred from a number.

Built with matplotlib's `ax.violinplot` (seaborn is not a dependency). A single-valued group has zero variance, which `gaussian_kde` cannot fit, so the violin body is skipped in that case and the jittered scatter still shows where the mass sits.

### Per-fold and pooled

| Scope | Tag | Where |
| --- | --- | --- |
| One graph | `Probabilities/Graph_<id>` | each fold dir (CV), the run dir (overfit) |
| All folds pooled | `CV/Probabilities_all_folds` | `<repeat>/aggregate/` — **CV only** |

The pooled version concatenates every fold's held-out predictions, so each edge appears exactly once, scored by a model that never saw its graph. Because each fold chose its own threshold, the pooled figure draws the **mean** threshold and says so in the legend — there is no single cut that applies to all of them. The per-fold violins are kept precisely because the pooled one hides which fold contributed what, which is the failure §10 is about.

## Node-type figures

Two figures, logged only when the [node head](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Design%20Choices.md#Node%20Classifier%20Head%20(optional)) is on and the graph carries `data.node_type`. Both use one palette — **red = background, blue = epithelial, orange = hyphal** (`NODE_TYPE_COLORS` in `gnn_train.py`) — matching the notebook 11 label figures, so exploration and training figures are read with the same eyes.

### Node-type comparison (2×2)

`plot_node_type_comparison` — source image, AIS fragments, **ground-truth** types, **predicted** types.

- **GT and prediction sit diagonally opposite**, so a misclassified fragment reads as a **colour flip in the same place**. The eye compares position, not legend.
- **Deliberately mirrors `plot_merge_comparison`'s layout** — the two figures are meant to be opened side by side, and a shared layout makes "this fragment is mistyped *and* mis-merged" visible at a glance.
- Panel titles carry the class counts and the accuracy, so a figure is interpretable without the scalars.

### Edge outcome by node pair

`edge_outcome_by_node_pair` / `plot_edge_outcome_by_node_pair` (`gnn_interpret.py:971`) — each edge's TP / FN / FP / TN outcome, grouped by **the pair of node types it connects** (`bg-bg`, `bg-epi`, …).

**This is the figure that tests the hypothesis.** The whole argument for the node head is that a true edge cannot span background↔cell or epithelial↔hyphal, and that the model would learn this implicitly. This figure asks it directly: *do the model's edge mistakes line up with the node types it believes in?*

- **It groups by the model's own predicted types, not ground truth** — deliberately. The question is whether the model's type beliefs and its edge beliefs are consistent *with each other*; ground-truth grouping would answer a different question.
- **`EDGE_OUTCOME_ORDER = ['TP', 'FN', 'FP', 'TN']`** puts the two error classes **adjacent, in the middle**, so the error mass is one contiguous block rather than split across the ends.
- **Only pairs that actually occur are plotted.** A pair with no edges would render as an empty bar and invite reading it as a score of zero.

## TensorBoard log paths

Rendered once per graph at `global_step=0`, after the best-AUC snapshot is restored:

| Figure | TensorBoard tag |
| --- | --- |
| Combined interpretation figure — all edges | `Interpretation/Graph_<id>` |
| Combined interpretation figure — sampled heatmap | `Interpretation/Graph_<id>_sampled` |
| Attention parallel coordinates | `Attention/Graph_<id>` |
| Predicted-probability violin | `Probabilities/Graph_<id>` |
| Predicted-probability violin, folds pooled | `CV/Probabilities_all_folds` (in `<repeat>/aggregate/`) |
| Node-type comparison 2×2 | `NodeType/Graph_<id>` — node-head runs only |
| Edge outcome by node pair | `NodeType/Graph_<id>_edge_outcomes` — node-head runs only |

`<id>` is the original dataset index of the test graph (same identifier used in `Predictions/Graph_<id>` and `Merge/Graph_<id>`), so the prediction overlay, the merge and the interpretation figure can be opened side by side in TensorBoard.

The combined figure uses a two-row layout. The top row contains PCA (left) and PLS-DA (right) as square axes side by side. The bottom row contains the full-width attribution heatmap (class strip, probability column, and the clustered attribution matrix). This allows immediate cross-referencing: an edge that appears as an outlier in the scatter plots can be located by its class and probability in the heatmap rows below. **The scatter plots always show every edge** — only the heatmap is subset, since it is the panel that grows unreadable with edge count.
