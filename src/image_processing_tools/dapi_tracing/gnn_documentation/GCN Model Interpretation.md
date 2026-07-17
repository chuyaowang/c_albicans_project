# GCN Model Interpretation

> How and why the trained model is inspected to understand its internal reasoning.
> Both analyses are computed at the best early-stopping epoch and logged to TensorBoard under `Interpretation/`.

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
| `Node (src)` | `data.x[i]` | 6 | Tabular morphology of source nucleus |
| `Node (tgt)` | `data.x[j]` | 6 | Tabular morphology of target nucleus |
| `Edge` | `data.edge_attr[e]` | 6 | Tabular path features between the two nuclei |
| `Visual (src)` | `node_visual[i]` (CNN output) | `d_vis` | Per-latent attribution from source nucleus patch |
| `Visual (tgt)` | `node_visual[j]` (CNN output) | `d_vis` | Per-latent attribution from target nucleus patch |
| `Visual (edge)` | `edge_visual[e]` (CNN output) | `d_vis` | Per-latent attribution from the edge's bounding box patch |

The visual streams (`Visual (src/tgt/edge)`) are only present when `use_visual_features=True`. Each latent dimension is kept as its own column (`vs_0 … vs_{d-1}`, `vt_0 … vt_{d-1}`, `ve_0 … ve_{d-1}`) rather than being aggregated. This makes the total column count `18 + 3 × d_vis` (e.g., 114 for `d_vis=32`), enabling direct comparison of which microsam latent dimensions co-vary with which manual features. The column clustering (see [Column clustering](#Column%20clustering)) is particularly useful here: if visual and manual features cluster together, it indicates that the CNN has learned a representation correlated with interpretable morphological properties.

The tabular features in column order are:

- **Node features:** `circ` (circularity), `ecc` (eccentricity), `area`, `int` (average intensity), `maj` (major axis length), `min` (minor axis length).
- **Edge features:** `e_int` (path intensity), `e_len` (normalised length), `ang1` / `ang2` (per-nucleus angle diffs), `min_ang` (minimum angle diff), `rel_ang` (relative nucleus orientation).

### Heatmap layout

The heatmap has three panels (left to right):

1. **Class strip:** a narrow colour bar per row indicating TP (green), TN (blue), FP (red), FN (orange).
2. **Probability column:** the symmetrized predicted probability displayed as a 'Greens' colour bar with the numerical value overlaid. The probability shown here is the value that determined the TP / TN / FP / FN classification — it uses `enforce_symmetric_predictions` before thresholding, matching the training objective.
3. **Attribution heatmap:** the per-edge, per-feature attribution matrix displayed with the `hot` colormap. Values are log-transformed (`log(1 + attr)`) and then row-normalised so each row sums to 1, making cell values directly comparable within each row as each feature's proportional contribution to that edge's prediction. Columns are globally Ward-clustered (see [Column clustering](#Column%20clustering)) so groups are no longer contiguous; group membership is indicated by **colored x-tick labels** (one color per group) and a legend patch inside the heatmap. White horizontal lines mark class boundaries.

### Row sort order

Edges are sorted into four class groups in the order TP → TN → FP → FN. Within each group:

- **TP ascending** by predicted probability — borderline true positives (just above the threshold) appear first; high-confidence correct positives appear last.
- **TN descending** by predicted probability — uncertain true negatives (closest to the threshold) appear first; confident correct negatives appear last.
- **FP ascending** — the most borderline false positives appear first.
- **FN ascending** — the most borderline false negatives appear first.

This layout places the "boundary" edges near the top of each class group and the clearly correct predictions at the bottom, making it easier to spot what features distinguish confident predictions from uncertain ones.

### Column clustering

- **How:** After row-normalisation, the columns (features) are reordered by Ward hierarchical clustering (`scipy.cluster.hierarchy.linkage`, Euclidean metric) applied to the transposed attribution matrix. The dendrogram leaf order from `leaves_list` is used to permute the columns. Group membership is stored per-column before permutation and carried through, so the colored x-tick labels and legend remain correct after reordering.
- **Why:** With 100+ columns, natural column order (all `Node (src)` features, then `Node (tgt)`, then `Edge`, then three visual groups) makes it hard to spot correlations between streams. Clustering places features that are attributed similarly across edges adjacent to each other. If a visual latent dimension clusters next to a tabular feature (e.g., `vs_7` next to `src_area`), it is a hypothesis that the CNN has learned a representation tracking that morphological property.
- **Caveat:** Ward clustering minimises within-cluster variance in the row-normalised attribution space. With only 20–30 edges per fold, the clustering is based on a small sample and should be interpreted as an exploratory visualisation, not a definitive feature grouping.

### Row normalisation

- **How:** Attribution values are first log-transformed (`log(1 + attr)`) to compress the heavy tail of large gradients, then divided by the row sum so that all values in each row sum to 1. A cell value of 0.3 means that feature accounts for 30% of the total log-attribution for that edge.
- **Why:** The goal is to compare *which features drove a given edge's prediction most strongly*. This is a within-row comparison. Scaling each row to sum to 1 makes the cell values directly interpretable as each feature's proportional contribution to that individual edge's classification. A per-column normalisation would instead answer a different question ("which edges relied most heavily on a given feature") and would prevent cross-feature comparison within a single edge.
- **Caveat:** Because log-transformation is applied before normalisation, cell values represent shares of *log-attribution* rather than raw attribution. The log scale suppresses extreme dominance by single features and keeps smaller contributions visible, which is the right trade-off for visual diagnosis. To inspect raw attribution magnitudes, use the `attr_matrix` returned by `compute_per_edge_attributions` directly.

## 3. Predicted-probability violin

`plot_probability_violin` draws the distribution of predicted probability for each held-out
graph, split by ground-truth label, with the fold's decision threshold across it.

### Why split by true label, not by TP/TN/FP/FN

TP and FN are both label-1 edges, separated only by the threshold; FP and TN are both
label-0 edges, likewise. Plotting the four classes apart would draw each true-label
distribution twice, sliced at the cut — a picture of the threshold, not of the model.
Grouping by what each edge *is* shows the two things that matter together: how far apart the
model pushes true and false edges, and where the cut happens to land between them.

### What it diagnoses

Saturation reads directly. A model that has collapsed onto one prediction renders as a flat
line rather than a distribution, which is the visual counterpart of the `Diag/Pred_Std_Test`
scalar. Overlap between the two violins is the model's real error budget: no threshold can
separate what the model did not separate. Each violin is annotated with its mean, standard
deviation and n.

## 4. Attention parallel coordinates

`plot_attention_parallel_coords` draws two vertical axes — layer-1 and layer-2 attention —
with every directed edge a marker on each, joined by a line coloured by TP / TN / FP / FN.

### Why parallel coordinates

The question is whether the two GCN layers attend to the same edges, and whether attention
tracks correctness. That is a per-edge relationship between two variables plus a class, and a
scatter of `a1` vs `a2` loses the identity of the line; parallel coordinates keep each edge a
single readable object across both layers. A rising line means layer 2 weights that edge more
than layer 1 did.

### Draw order

Classes are drawn worst-populated-last, and TN is drawn faint. The candidate graph is
fully connected, so negatives dominate by construction and would otherwise bury the handful
of TP / FP / FN lines that carry the signal.

### The CSV

`attention_dataframe` backs both the figure and an `attention_graph_<id>.csv` written beside
the fold's event file, so the exported numbers and the plotted ones cannot drift apart. The
frame carries one row per directed edge: `edge_idx`, `src`, `tgt`, `a1`, `a2`, `prob`,
`true_label`, `edge_class`.

## Design decisions

### Test graphs only

The model has been trained on the training fold graphs: their embeddings and attributions reflect memorised patterns, not generalisation. Only the held-out test graph reveals how the model performs on unseen data. Training-graph interpretation would be misleading as a model quality signal, though it can be added by calling the same functions on `train_dataset` if needed.

### Best epoch only

Both analyses are expensive (attribution requires one backward pass per edge). Running them at every epoch would multiply wall-clock time by the number of edges per fold. The best early-stopping epoch is the only model state that matters for evaluation, so it is the right time to inspect.

### `model.eval()` and deterministic embeddings

All interpretation passes use `model.eval()`, which disables Dropout. This is essential: stochastic masking would produce different embeddings and different gradients on each call, making the results non-reproducible and mixing attribution signal with dropout noise. `LayerNorm` and `GraphNorm` have no running statistics, so their behaviour is identical in train and eval mode.

### Model parameter gradient suppression

During the attribution backward loop, model parameters are temporarily frozen (`model.requires_grad_(False)`) to avoid allocating and computing gradients for parameters that are not used. Only the four tracked input tensors (`x`, `edge_attr`, `node_visual`, `edge_visual`) need gradients. This reduces memory allocation and backward-pass time proportionally to the number of model parameters.

## TensorBoard log paths

All three panels (PCA, PLS-DA, attribution heatmap) are rendered into a single combined figure per fold, logged once at `global_step=0`:

| Figure | TensorBoard tag |
| --- | --- |
| Combined interpretation figure | `Interpretation/Graph_<id>` |
| Predicted-probability violin | `Probabilities/Graph_<id>` |
| Attention parallel coordinates | `Attention/Graph_<id>` |

Alongside the fold's event file, `attention_graph_<id>.csv` carries the same per-edge
attention table the parallel-coordinates figure is drawn from.

`<id>` is the original dataset index of the test graph (same identifier used in `Predictions/Graph_<id>`), so the prediction overlay and the interpretation figure can be opened side by side in TensorBoard.

The combined figure uses a two-row layout. The top row contains PCA (left) and PLS-DA (right) as square axes side by side. The bottom row contains the full-width attribution heatmap (class strip, probability column, and the clustered attribution matrix). This allows immediate cross-referencing: an edge that appears as an outlier in the scatter plots can be located by its class and probability in the heatmap rows below.
