# GCN Design Choices

> How and why each component is designed the way it is.
> See the visual mermaid diagram at [GCN Model Mermaid Diagram](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Mermaid%20Diagram.md)
>
> **Scope — applies to both pipelines.** Every design on this page is **shared and live**: the model is identical for the historical **nuclei** pipeline and the current **cell-fragment merge** pipeline. Verified by diff — the only differences in `simple_gnn.py` are the RoI box source ([§2 below](#2.%20Region-of-interest%20extraction%20—%20RoIAlign)) and the `node_feature_dim` / `edge_feature_dim` defaults (now **8 / 10**; historically 6 / 6). Where the prose below says "nuclei", read "nodes" — the wording is nuclei-era. Full breakdown: [Nuclei vs. cell-fragment](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Cell%20Mask%20Graph%20Data%20Flow.md#Nuclei%20vs.%20cell-fragment%20—%20what%20carries%20over).

## GCN Layer

A custom Graph Convolutional Network (GCN) layer is used to perform message passing. Unlike standard GCNs which only update node embeddings based on static edges, this layer dynamically scales messages using an attention mechanism and explicitly utilizes edge attributes. It consists of four main components:

### 1. Message Function

- **How:** Calculates the directional difference between the target node and source node ($x_j - x_i$) and concatenates it with the edge features ($edge\_attr$). This combined vector is passed through an [MLP Module](#MLP%20Module)  to generate the raw message.
- **Why:** 
  - Using the difference $x_j - x_i$ captures the relative change in morphology or intensity between connected nuclei, which is more informative for tracing paths than absolute values.
  - Concatenating the `edge_attr` grounds the message in the physical, visual evidence (path intensity, relative angles, normalized length) that physically connects the two cells.

### 2. Attention Mechanism (Attn-MLP)

- **How:** In parallel to the message function, the same concatenated vector $[x_j - x_i, edge\_attr]$ is passed through a single `CustomLazyLinear` layer to produce a scalar score. A `softmax` function is then applied over the local neighborhood (all edges pointing to the same target node) to convert these scores into normalized attention weights ($\alpha$) that sum to 1.0. The raw message is multiplied by this weight.
- **Why:** It acts as a dynamic, learned filtering mechanism. The model learns to assign high weights to biologically plausible hyphal connections and suppress false positives (noise or background edges) *before* the information is aggregated into the node's state. It explicitly models the "competition" between candidate edges.
- **Experimental basis:** [Dynamic edge weight (attention)](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#Dynamic%20edge%20weight%20(attention)).

### 3. Aggregation Function

- **How:** Uses a simple summation (`aggr='sum'`) of the attention-scaled messages from all neighbors.
- **Why:** Because the messages are already scaled by the softmax attention weights (which guarantee the total incoming weight equals 1.0), the sum aggregation naturally computes a perfectly balanced weighted average. This elegantly removes the need for the rigid, traditional degree normalization (e.g., $1/\sqrt{deg(i) deg(j)}$) used in standard GCNs, allowing the network to handle endpoints (degree 1) and continuous chains (degree 2) without mathematical bias.

### 4. Update Function

- **How:** Concatenates the node's current embedding with the aggregated messages from its neighborhood, and passes the result through an [MLP Module](#MLP%20Module) to produce the new node embedding.
- **Why:** By concatenating rather than just adding, the node retains a distinct representation of its own independent features while fusing them with the contextual information gathered from its neighbors.

## MLP Module

This specific Multi-Layer Perceptron (MLP) block is the workhorse of the architecture, used recurrently to transform embeddings:
*   `CustomLazyLinear` $\rightarrow$ `LayerNorm` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout` $\rightarrow$ `Linear`

- **CustomLazyLinear:** Automatically infers input dimensions, eliminating hardcoded shape tracking. The custom implementation explicitly applies **Kaiming (He) Normal initialization**, which prevents vanishing/exploding gradients specifically when paired with the subsequent ReLU activation.
- **LayerNorm:** Normalizes each item's activation vector across the feature dimension independently. **Why it replaced BatchNorm:** `LazyBatchNorm1d` was originally used here and did stabilize early training, but it maintains running mean/variance statistics that are computed from the training batch and applied at eval time. Under LOO cross-validation with a single held-out graph, this produces a per-channel offset between the training distribution and the held-out graph's activation distribution; large-magnitude weights (which develop as BCE drives toward zero) then amplify this offset through subsequent linear layers, saturating the sigmoid output. `LayerNorm` normalizes per item with no running statistics and identical train/eval behavior, eliminating this instability entirely. See [Saturated probabilities under leave-one-out CV](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#10.%20Saturated%20probabilities%20under%20leave-one-out%20CV%20(in%20progress)) for the full diagnosis.
- **ReLU:** Introduces non-linearity, allowing the model to learn complex, non-linear biological decision boundaries.
- **Dropout:** Randomly zeroes out elements of the tensor during training. Acts as robust regularization to prevent the network from memorizing specific training graphs and forces it to distribute its learned logic across multiple features.
- **Linear:** The final projection to the desired output dimension (using Xavier Uniform initialization).
- **Hidden dimension:** Fixed at 128 (upgraded from 64 in visual-feature experiments).
- **Experimental basis:** [Non-linearities and MLP depth](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#Non-linearities%20and%20MLP%20depth) (motivation for the MLP block and for dropout), [Weight initialization](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#Weight%20initialization) (Kaiming + Glorot), [Hidden layer size](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#Hidden%20layer%20size) (why 64 → 128), [Saturated probabilities under leave-one-out CV](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#10.%20Saturated%20probabilities%20under%20leave-one-out%20CV%20(in%20progress)) (BatchNorm → LayerNorm).

## The Edge Updater

- **How:** Between every GCN layer, the edge attributes are explicitly updated. The updater extracts the newly calculated source node embedding ($x[edge\_index[0]]$) and target node embedding ($x[edge\_index[1]]$), concatenates them with the current edge embedding, and passes them through an [MLP Module](#MLP%20Module) to output a new edge representation.
- **Why:** In standard GCNs, edge features are static and only used to assist node updates. In this tracing task, the *edges themselves* are the final targets. By updating the edge embeddings using the progressively context-aware node embeddings, the edges evolve to "understand" their structural role within the broader cell chain, rather than just representing local pixel intensity.
- **Experimental basis:** [Non-linearities and MLP depth](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#Non-linearities%20and%20MLP%20depth).

## Classifier Head

- **How:** The final module extracts the final source node, target node, and edge embeddings, concatenates them, and passes them through an [MLP Module](#MLP%20Module) that outputs a single logit. This logit is passed through a Sigmoid function to produce a probability between 0.0 and 1.0.
- **Why:** It mimics the human visual decision-making process. To classify if a connection is real, the network evaluates the complete local triad simultaneously: *What does the source cell look like? What does the target cell look like? And what does the physical path between them represent?*
- **Experimental basis:** [Non-linearities and MLP depth](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#Non-linearities%20and%20MLP%20depth).

## Node Classifier Head (optional)

A **second** head, predicting each node's type — `background` (0) / `epithelial` (1) / `hyphal` (2) — from the same final node embedding the edge classifier reads. Off by default; enabled with `Model(predict_node_type=True, num_node_classes=3)`.

- **How:** `NodeClassifier` (`simple_gnn.py:134`) is one `MLP body → Linear` over the post-residual node embedding `x_out`: `CustomLazyLinear(hidden) → LayerNorm → ReLU → Dropout → Linear(hidden, num_classes)`. It emits **raw logits**, never a softmax — `CrossEntropyLoss` applies log-softmax itself, and a softmax here would apply it twice and flatten the gradients.
- **Why a second head rather than a feature or a filter:** the two failure modes it targets — AIS calling background regions cells, and candidate edges spanning epithelial↔hyphal masks — are both statements about *nodes*, and both imply a constraint on edges (a true edge cannot span background↔cell or epithelial↔hyphal). Feeding the type in explicitly would require *knowing* it at inference, which is the very thing being predicted. Sharing the trunk and combining the losses instead means the constraint is learned **implicitly**: the representation must serve both tasks, so it must encode what a node is in order to score its edges. **Nothing is fed in explicitly.**
- **Where it reads from:** `x_out` — *after* the residual concatenation and both GCN layers, the same tensor the edge classifier consumes. So the two heads share every parameter below them, which is the entire point: the gradient from the node task shapes the trunk the edge task uses.
- **The visual branch is not optional for it.** A fragment's type is a property of the *whole cell* it belongs to, not of the fragment's own shape (see [Node Type Label Construction](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Node%20Type%20Label%20Construction.md)), so tabular geometry alone cannot decide it — the head needs image context and neighbourhood message passing.
- **Guard:** `return_node_logits=True` against a model built without the head raises rather than silently returning nothing (`simple_gnn.py:470`). Node logits are appended **last** in every return signature, including `attribution_mode`.
- **Results:** [GCN Model Experiments §11](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md). In short — it improved edge AUC on all six folds while its own predictions stayed mediocre; it earns its place as an **auxiliary task**, not as a working node classifier.

## Overall Model Flow

### Input data

- **Node features:** Morphological cell properties (area, eccentricity, etc.).
- **Edge features:** Path visuals (intensity, length, angles).
- **Edge index:** The graph connectivity matrix mapping the proposed topology.

### 1st GCN layer

- Nodes aggregate messages from their immediate (1-hop) neighbors, weighted by the attention mechanism. Nodes learn about their immediate surroundings.

### Edge updater

- Edges update their representations based on the newly context-aware 1-hop node embeddings.

### 2nd GCN layer

- Nodes aggregate messages again. Because their neighbors already contain information from *their* neighbors, this step expands the receptive field to 2 hops. 
- **Why 2 Layers:** Given the relatively small size of the individual hyphal chains in the dataset (3 to 6 nodes), a 2-hop receptive field is sufficient for a node to understand its position within the entire local chain without suffering from severe oversmoothing.

### Residual connection

- **How:** After every GCN Layer and Edge Updater, the output embeddings are concatenated with the *original, raw* input features (`x_orig`, `edge_attr_orig`). This concatenated vector is immediately passed through a `GraphNorm`.
- **Why:**
  1. **Preventing Information Loss:** Deep GCNs suffer from "oversmoothing," where repeated message passing causes all embeddings to blend together. Skip connections guarantee that the network always has direct access to the raw, unadulterated biological measurements (e.g., exact pixel intensity or absolute cell size) at every depth.
  2. **Normalization:** Fusing deep abstract embeddings with raw input data creates a tensor where the two components have very different scales. The subsequent norm re-centers and rescales this concatenation before the next layer processes it.
  3. **Why GraphNorm instead of BatchNorm:** The residual norms are called inside `Model.forward()` where `data.batch` (and the corresponding edge batch vector) are available, allowing `GraphNorm` to normalize each graph's node/edge representations independently. Unlike `LazyBatchNorm1d`, `GraphNorm` accumulates no running statistics and behaves identically at train and eval time, eliminating the train/eval asymmetry that was the root cause of prediction saturation. The learnable per-channel ratio `α` additionally allows the model to decide how much of the graph-level mean to suppress. See [Saturated probabilities under leave-one-out CV](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#10.%20Saturated%20probabilities%20under%20leave-one-out%20CV%20(in%20progress)).
- **Experimental basis:** [Residual connections](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#Residual%20connections), [Saturated probabilities under leave-one-out CV](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#10.%20Saturated%20probabilities%20under%20leave-one-out%20CV%20(in%20progress)).

### Classification

- The Classifier processes the final fused embeddings to output the absolute probability of a true hyphal connection.
- **When `predict_node_type=True`, the same `x_out` also goes to the [Node Classifier Head](#Node%20Classifier%20Head%20(optional))**, which emits `(N, num_node_classes)` raw logits alongside the edge probabilities. The two heads run in parallel off one trunk; only the losses combine (see [GCN Training Choices](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Training%20Choices.md#Loss)).

## Visual branch

The baseline model treats each node and edge as a small tabular feature vector (area, intensity, angles, normalized length, …). The visual branch adds a parallel stream that gives every node and every edge access to raw image content through the MicroSAM encoder, without re-running the ViT at training time.

### 1. Feature source — MicroSAM encoder

- **How:** The MicroSAM ViT is run **once per image**, offline, via `precompute_microsam_feats.compute_microsam_features`. The image is tiled with an overlapping halo, each tile is encoded to a `(256, 64, 64)` grid, and tiles are stitched into a whole-image feature map `(256, H_f, W_f)` with a scalar `pixels_per_feature`. The result is cached to disk next to the image, and attached to the PyG `Data` object at graph-build time (see [Visual features from MicroSAM](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Data%20Flow.md#Visual%20features%20from%20MicroSAM)).
- **Why:** MicroSAM is a strong microscopy-pretrained encoder, but it lives in a separate conda env from the GNN and the ViT pass is expensive. Precomputing once and caching decouples the two environments entirely: the GNN trains on plain tensors. It also makes RoIAlign over nuclei and edges near-free at training time.

### 2. Region-of-interest extraction — RoIAlign

- **How:** The **box source depends on the pipeline** — this is the one place the two differ (`_node_boxes` / `_edge_boxes` in `simple_gnn.py`):
  - **Cell fragment (live):** when `data.node_bboxes` is present, each node box is that fragment's **mask bbox** padded by `node_bbox_pad_frac` (default 10%), and each edge box is the **union** of its two endpoints' padded mask bboxes.
  - **Nuclei (historical):** with no `data.node_bboxes`, the code falls back to a fixed-size axis-aligned box (default 150×150 px) around each centroid, and an edge box spanning the source and target centroids padded by a fractional margin (default 15%) with a minimum pixel floor (default 20 px) so very short edges still get a reasonable receptive field.

  Either way the boxes are fed through torchvision's `roi_align` with `spatial_scale = 1 / pixels_per_feature`, producing a fixed `(256, roi, roi)` patch per node and per edge (default `roi=7`). Everything downstream of box construction is identical between the pipelines.
- **Why:** Different edges span wildly different pixel distances, and different objects have different sizes; a per-node box + a data-adaptive box per edge means the CNN downstream always receives the same-shape tensor regardless of the underlying geometry. The **mask bbox** is the better node box for fragments because a fragment's extent varies enormously (a long hyphal fragment vs. a small yeast one), so a fixed square would both crop long fragments and swamp small ones with background — whereas a nucleus is roughly constant-sized, which is why the fixed square sufficed historically. The edge box is deliberately loose (fractional overshoot / padded union) because the biologically relevant signal — the connecting cell body — often extends slightly outside the tight bounding rectangle.

### 3. Visual CNN per stream (`NodeVisualCNN`, `EdgeVisualCNN`)

- **How:** A small CNN (`Conv(256→64) → ReLU → Conv(64→32) → ReLU → GlobalAvgPool → Linear(32→d_visual)`) maps each patch to a `d_visual`-dimensional vector. Separate weights for nodes and edges.
- **Why:** Nodes and edges look for different things — a node patch is dominated by a single nucleus, while an edge patch must evaluate the path *between* two nuclei. Sharing weights would conflate the two questions. The CNN is intentionally shallow: the MicroSAM features already carry rich high-level semantics, so the CNN's job is to summarize a 7×7 patch, not re-learn a representation from scratch.
- **`d_visual` is configurable** so it can be swept alongside `hidden_channels`.

### 4. Fusion with tabular features (`FusionMLP`)

- **How:** Before the first GCN layer, node tabular features are concatenated with `NodeVisualCNN`'s output and projected through an [MLP Module](#MLP%20Module) down to `hidden_channels`; edge features are fused the same way with `EdgeVisualCNN`.
- **Why:** Passing both streams through an MLP with `LazyBatchNorm1d` normalizes the scale mismatch between z-scored tabular features and un-normalized CNN outputs, and lets the network learn how to weigh morphology vs. pixels per feature dimension. The fused output replaces the raw `x` / `edge_attr` as the input to the first GCN layer.

### 5. Skip-connection source — pre-fusion raw features

- **How:** The existing residual connections around each GCN layer and each edge updater continue to concat the *pre-fusion* raw `x` / `edge_attr` back in. The visual features are mixed into the computation path via the fused inputs but are **not** independently re-injected as a skip.
- **Why:** The residual connections were originally designed to guarantee access to clean, interpretable biological measurements (absolute area, raw intensity) at every depth. Keeping the skip source unchanged preserves that behavior and isolates the visual-feature experiment to the forward path. **Alternatives that are still on the table, to be decided by experiment:**
  1. Skip from the *post-fusion* embedding, so later layers also see pre-digested visual information.
  2. Treat the visual features as a third, separate skip stream concatenated alongside the raw-tabular skip.
- **Experimental basis:** pending — add an entry to `GCN Model Experiments.md` when the first visual-feature sweep lands so the chosen skip source is justified by evidence, not just by default.

### Configuration surface

All visual-branch knobs are model constructor arguments (plumbed through `model_params` in the notebook):

| Argument | Default | Meaning |
| --- | --- | --- |
| `use_visual_features` | `False` | Master switch; when `False` the model behaves identically to the baseline. |
| `d_visual` | `16` | Output dimension of each `VisualCNN`; the extra width fused into each stream. |
| `node_box_size` | `150` | Side length (pixels) of the square RoI centered on each nucleus centroid. |
| `edge_box_margin_frac` | `0.15` | Fractional padding added to the endpoint bounding box. |
| `edge_box_margin_floor` | `20` | Minimum pixel padding, to give very short edges enough context. |
| `roi_output_size` | `7` | RoIAlign output spatial size (`roi × roi`). |