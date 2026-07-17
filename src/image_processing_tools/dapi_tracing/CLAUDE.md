# C. Albicans DAPI Tracing GNN Project - Context & Architecture Document

## 1. Problem Statement

The goal of this project is to automatically trace unbranched, acyclic hyphal cell chains of *Candida albicans* from fluorescence microscopy images (DAPI channel). The system must identify distinct nuclei, evaluate the biological and visual plausibility of connections between them, and predict the exact graph structure representing the true cell chains.

Biological constraints dictate that the resulting graph must be:

1. **Unbranched:** Nodes generally have a maximum degree of 2 (endpoints have degree 1).
2. **Acyclic:** Cell chains do not form closed loops (rings/hexagons).
3. **Distinct:** Parallel or separate hyphal structures must not be artificially merged.

## 2. Background & Data Type

The input data originates from raw image processing (segmentation, filtering, and feature extraction).
The dataset is structured as **PyTorch Geometric (PyG) `Data` objects** representing undirected graphs.

* **Node Features (`data.x`):** Morphological properties extracted via `skimage.measure.regionprops`. 6 features in column order: `circularity`, `eccentricity`, `area`, `average_intensity`, `major_axis_length`, `minor_axis_length`.
* **Edge Features (`data.edge_attr`):** Visual and spatial features characterizing the path between two nuclei. 6 features in column order: `average_intensity` (mean of non-zero pixels along a line profile, width=3, excluding nucleus pixels), `length` (Euclidean distance divided by `avg_nucleus_length`), `node1_angle_diff` (angle between path and nucleus 1 major axis, normalized by π/2), `node2_angle_diff` (same for nucleus 2), `min_diff_angle` (minimum of the two per-node angle differences, normalized by π/2), `relative_angle` (absolute difference between the two nucleus orientations, folded to [0, π/2], normalized by π/2). All four angle features are bounded to `[0, 1]` and are excluded from Z-score normalization to preserve their geometric meaning.
  * **With-in Graph Normalization:** An approach to normalize intensity and area features within each graph using the mean feature value instead of z-score normalization in the training set was attempted. However, due to the small size of the dataset, the mean gets heavily influenced by any outliers in the data. This ended up degrading the model performance.
* **Graph Structure (`data.edge_index`):** A fully connected candidate graph where the GNN must classify true edges (`1.0`) vs false edges (`0.0`).
* **Edge Labeling (`data.edge_label`):** At the moment, these graphs are labeled manually in the jupyter notebook.
* **Dataset used:** a dataset containing 6 graphs with sizes varying from 3 to 6 nodes. The graphs are fully connected.

## 3. Project Structure

### `nuclei_detection.py` (Shared Image Segmentation & Filtering)

* **`detect_nuclei`:** Segments the DAPI image using Otsu thresholding with optional Watershed to split touching nuclei.
* **`detect_nuclei_rf`:** Segments using a trained Random Forest model as an alternative to Otsu.
* **`filter_nuclei`:** Filters detected objects by size and eccentricity, and computes `avg_nucleus_length` as the biological ruler.
* This module is the shared foundation imported by both the GNN pipeline and the greedy approach.

### `extract_graph.py` (GNN Feature Extraction)

* **`extract_graph`:** Compiles node and edge features into Pandas DataFrames, normalizing distances dynamically using `avg_nucleus_length`. This is the entry point for the GNN data pipeline.

### `deprecated/greedy_connectivity.py` (Deprecated Greedy Approach — Do Not Use in GNN Pipeline)

* Contains the old deterministic greedy algorithm for constructing the hyphal network.
* **`calculate_connectivity`:** Scores pairwise connections using intensity profiles, distance penalties, and orientation alignment.
* **`plot_nuclei_analysis`:** Visualizes the greedy connectivity network in a multi-panel figure.
* **`extract_and_plot_nuclei_axis`:** Main driver tying the greedy pipeline together.
* Since a greedy algorithm is not differentiable and has been shown to fail for images containing multiple hypha cells, these functions must never be used in the GNN approach. Do not make any attempt to do so.

### `gnn_data.py` (Data Pipeline & Training Loop)

* **Data Conversion:** Maps the Pandas DataFrames to PyG Tensors, calculates true node degrees from undirected labels, and structures the `Data` objects.
* **Cross Validation (`n_fold_validation`):** Implements rigorous K-Fold CV. Critically, Z-score normalization for node features and edge intensity is calculated *strictly on the training fold* to prevent data leakage. Edge lengths and angles are deliberately *not* Z-scored to preserve their universal biological scaling.
* **Symmetry Enforcement:** Applies `enforce_symmetric_predictions` (averaging $P(A \rightarrow B)$ and $P(B \rightarrow A)$) to ensure the network learns robust, direction-agnostic features for undirected connections.

### `simple_gnn.py` (Core GNN Architecture)

The model leverages a heavily customized Graph Convolutional Network designed to treat edges as first-class citizens alongside nodes.

* **Custom Layers:** Uses `CustomLazyLinear` with Kaiming Normal initialization paired with `ReLU` activations and `LazyBatchNorm1d`.
* **`GCNConv` (Message Passing):** Calculates messages by concatenating source/target node differences with edge attributes. It includes an internal `attn_mlp` that calculates and applies softmax-normalized attention scores over local neighborhoods.
* **`EdgeUpdater`:** Between every node-updating GCN layer, the edge features are explicitly updated by concatenating the newest source and target node embeddings with the previous edge features.
* **Skip Connections:** Concatenates original input features to the output of intermediate layers to prevent feature smoothing/vanishing over depth. The features are renormalized after concatenating the original features, preventing the original features from dominating the gradident flow.
* **`Classifier`:** A final MLP that evaluates the concatenated source node, target node, and updated edge embeddings to predict a binary probability.

### Jupyter notebooks

* Notebooks that do not live in this directory. They only import functions from here to run the experiments.
* The edge labels are also entered manually in the notebooks.

## 4. GNN Design Choices & Key Decisions

1. **Dual Optimizers:** The architecture separates parameters between two optimizers. The `Muon` optimizer handles 2D weight matrices (hidden layers) for superior gradient updates, while `AdamW` handles 1D parameters (biases, batch norms, classifier head).
2. **Biological Normalization vs. Statistical Normalization:** Raw distance in pixels does not generalize across magnifications. The `length` feature is normalized by the image's average nuclei length. Because this yields a universally stable biological metric (e.g., 2.5 cell lengths), it is explicitly excluded from Z-score normalization to preserve its intrinsic meaning.
3. **Average vs. Max Symmetry:** For the baseline symmetric GNN, `average` symmetry is used during training. This forces the model to learn reliable bidirectional features (an "AND" relationship). `Max` symmetry acts as an "OR" gate and is only useful when specific directional masking tricks are applied.
4. **Topological Sinks / Directed Constraints (Abandoned):** Attempting to force an undirected acyclic graph purely through directed topological potentials (Sinkhorn/Acyclic models) failed mathematically. The local `max` symmetrization bypassed directional masks, and the degree penalty had a blind spot for global rings (such as hexagons). Do not go further down this path.

## 5. Loss Function Design

The loss is a composite function targeting both individual edge accuracy and structural graph constraints:

1. **BCE Loss:**
    * The 'classification loss'. Evaluates the correctness of the classification against the ground truth
2. **Sparsity-Aware Degree Penalty:**
    * The 'degree' loss. Tried to make the model predict biologically plausible node degrees.
    * Calculates the mean squared error between the predicted degree and the true degree.
    * The predicted degree is composed of two components:
      * First term: The `top-k` highest probabilities are summed for a node whose true node degree is `k`.
      * Second term: If there are other probabilities remaining, those probabilities are averaged.
      * The second term is subtracted from the first term, and the result is evaluated against the true node degree. This forces the model to predict k high probabilities and close to 0 probabilities for other edges.
    * Remaining issue: despite the careful design, the model can still cheat sometimes by predicting all edges to have close to 0 probabilities. This will minimize the second term in the degree loss, even if the first term is still high.
3. **Negative edge sampling for loss calculation:**
   * The graph data can have class imbalance (number of positive vs. negative edges) depending on the graph size. Since the dataset contains graphs of various sizes, the class imbalance also has high variance.
   * This shifts the model's assumption about the underlying data distribution depending on which graphs are allocated in the training and which in testing during the cross validations.
   * Because of the high variance, a weighted BCE loss will not work since the weight calculated in the training set will not be suitable for the graph in the testing set. Do not suggest this approach in the future.
   * The working approach is sampling negative edges from the graph to maintain a consistent positive-negative ratio, and use only the sampled true and false edges to calculate the loss. This will prevent the model making wrong assumptions on the data distribution.
   * Also, the class imbalance means that when the test set has an overwhelming amount of negative edges, the model that predicts every edge to be negative will be kept by the early stopping mechanism. For this reason, the early stopping criteria is the AUC, not the accuracy. The threshold to classify edges as 1/0 from the probabilities is chosen dynamically to be the value that maximizes the F1 score.

## 6. Experiment Logging Standards

The project uses a combination of `tqdm` for real-time console monitoring and `TensorBoard` for comprehensive experiment tracking and visual diagnostics.

### Console Logging (`tqdm`)

* **Epoch Progress:** Training loops (`n_fold_validation` and `train_overfit_test`) are wrapped in `tqdm` progress bars to monitor epoch iterations.
* **Dynamic Postfix:** At the end of each epoch, the progress bar's postfix is dynamically updated with key metrics: Training Loss, Training Accuracy, Validation AUC, PR_AUC, and F1-Score. This provides immediate, real-time feedback on model convergence and early stopping conditions.

### Experiment Tracking (`TensorBoard`)

* **Directory Structure:** Logs are saved systematically under `output/cv_experiment/<root_experiment_name>/<repeat id>/fold_<k>` (or `output/overfit_experiment/...` for overfit testing).
* **Scalar Metrics:** The `SummaryWriter` records continuous metrics per epoch, which include:
  * **Training Components:** `Loss/Train_Total`, `Loss/Train_BCE`, `Loss/Train_DegreePenalty`, and `Accuracy/Train`.
  * **Validation Components:** `Loss/Test`, `Accuracy/Test`, `AUC/Test`, `PR_AUC/Test`, and `F1/Test`.
* **Text Summaries:** A high-level summary of each fold (or test run) is logged as text under the `Fold Summary` tag. This includes the exact train/test indices, best threshold found, and final evaluation metrics, allowing for quick lookups of fold configurations.
* **Graph Visualizations:** During the final evaluation of a fold, the model's unbatched predictions on the test set are visually plotted over the original microscopy image (`data.image`) using `plot_edge_predictions`. The background is **not assumed to be a single modality** — `data.image` may be a channel stack (e.g. DIC plus one or more fluorescence channels), so the renderer dispatches on shape: 2D or `(H, W, 1)` → grayscale; `(H, W, 2)` → composite with channel 0 (DAPI) in blue and channel 1 (DIC) in grayscale, each percentile-stretched independently; `(H, W, 3)` → shown as-is. These diagnostic figures highlight:
  * True Positives
  * False Positives
  * False Negatives
  * True Negatives
  * They also overlay the actual predicted probabilities and intermediate GCN layer attention scores (`A1`, `A2`) directly on the image. These generated figures are logged directly to TensorBoard under the `Predictions/Graph_<id>` tag.
* **Per-graph diagnostics:** two further figures are logged for each held-out graph at the best epoch, described in [GCN Model Interpretation](gnn_documentation/GCN%20Model%20Interpretation.md):
  * `Probabilities/Graph_<id>` — predicted probability split by **ground-truth label** (not by TP/TN/FP/FN, which would slice each true-label distribution at the threshold rather than show the model), with the fold's threshold drawn across. Saturation reads directly: a collapsed model renders as a flat line.
  * `Attention/Graph_<id>` — parallel coordinates of each directed edge's layer-1 and layer-2 attention, coloured by TP/TN/FP/FN, with TN faint since negatives dominate a fully connected candidate graph. The same table is exported beside the event file as `attention_graph_<id>.csv`, so figure and export cannot drift.

## 7. Claude interaction guidelines

* Always draft coding plan first and do not start coding unless the user gives explicit approval.
* Documentation management: the `gnn_documentation/` folder contains markdown files. It contains the following files and purposes:
  * GCN Data Flow: describes feature generation, normalization, pre-processing, train-test split strategy, and how PyG's batching mechanism works.
  * GCN Design Choices: describes model modules and why are they chosen.
  * GCN Model Mermaid Diagram: mermaid diagrams that visualizes the GCN structure.
  * GCN Training Choices: describes training protocol, loss choice, early stopping, performance tracking, and negative edge sampling.
  * GCN Model Experiments: what experiments were performed to improved the model, what worked, what did not work, and the reasoning. Model choices and training choices are backed by experiments.
  * GCN Model Interpretation: describes the two post-training interpretation analyses (pre-logit embedding PCA/PLS-DA and per-edge gradient × input attribution heatmap), their design decisions, and how to read the TensorBoard figures.
* Documentation management guideline:
  * Keep the documentation updated when new changes happen to the code.
  * The documents were imported from Obsidian and contain links to each other. Create new links when necessary following these examples:
    * `[Node features](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Data%20Flow.md#Node%20features)` for cross-file links. Preserve the original path so the link can still be resolved in Obsidian.
    * `[MLP Module](#MLP%20Module)` for with-in file links.
  * When adding new entries to training choices and model design choices, link the supporting section from the model experiments. If there is none, ask the user to update the experiment document.
