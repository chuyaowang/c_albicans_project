# GCN Training Choices

> How the model is trained to achieve the desired learning outcome. Every decision below links to its supporting evidence in [GCN Model Experiments](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md).
>
> **Scope — applies to both pipelines.** Every training decision on this page is **shared and live**: the loss, optimizers, cross-validation and normalization are identical for the historical **nuclei** pipeline and the current **cell-fragment merge** pipeline. Caveat: the supporting experiments were run on nuclei data and have not been re-measured on fragments (see [GCN Model Experiments](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md)). Full breakdown: [Nuclei vs. cell-fragment](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Cell%20Mask%20Graph%20Data%20Flow.md#Nuclei%20vs.%20cell-fragment%20—%20what%20carries%20over).

## Loss

The loss is **BCE with label smoothing, on sampled edges** — a pure classification objective. It was historically a *composite* of BCE + a structural degree penalty, but the degree term is now **disabled** (see [§2](#2.%20Sparsity-aware%20degree%20penalty%20(disabled))), so no structural constraint enters the loss.

### 1. BCE (classification) loss with label smoothing

- **How:** Binary cross-entropy between the predicted edge probability and a *smoothed* ground-truth label. Hard labels 0 and 1 are replaced by `ε` and `1 − ε` (currently `ε = 0.1`), giving soft targets 0.1 and 0.9. The formula is `smoothed = target × (1 − ε) + (1 − target) × ε`. Smoothing is applied only during training on the sampled loss indices; `test_model` always evaluates against hard labels.
- **Why label smoothing:** When train BCE falls well below 0.1, the optimizer grows weight magnitudes to push edge logits toward ±∞ (because `sigmoid(z) → 1` requires `z → +∞`). Even after fixing the BatchNorm instability with GraphNorm, the logit scale continues to grow, pushing the F1-maximizing threshold to extreme values (near 0 or 1). Label smoothing creates a finite BCE floor of `H(1−ε, ε) = −ε log ε − (1−ε) log(1−ε) ≈ 0.325` (at `ε = 0.1`) that the loss cannot go below regardless of logit magnitude. This keeps probabilities within a useful range and produces more calibrated, interpretable outputs.
- **Why not weighted BCE:** The BCE here is **unweighted** (`torch.nn.BCELoss()`, no `weight` / `pos_weight`). A positive weight $w^{+} = N_{\text{neg}}/N_{\text{pos}}$ derived from the training fold is itself a fold statistic, so it does not fix the train/test distribution mismatch — it *inverts* it: a fold of large graphs yields $w^{+} = 2$, and on a balanced test graph the model then over-predicts instead of under-predicting. Class balance is handled by [negative edge sampling](#Negative%20edge%20sampling) instead, which fixes the ratio to a constant rather than correcting a variable one. See [Weighted BCE](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#Weighted%20BCE%20—%20tried%20first,%20before%20negative%20sampling).
- **Experimental basis:** [Saturated probabilities under leave-one-out CV](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#10.%20Saturated%20probabilities%20under%20leave-one-out%20CV%20(in%20progress)).

### 2. Sparsity-aware degree penalty (disabled)

> ⚠️ **Disabled — do not reintroduce.** `degree_penalty_weight` defaults to `0.0` in `train_model`, `n_fold_validation` and `train_overfit_test`, and the penalty is only computed when the weight is `> 0`. It contributes nothing to the current objective. Retained here as a record of the formulation and why it was dropped.

- **How (v3, as implemented):** MSE between the true node degree $k$ and a *constructed* predicted degree:
  - Sum the top-$k$ predicted probabilities incident to the node.
  - Subtract the mean of the remaining probabilities.
  - Compare the difference to $k$.
- **Intent:** force wide separation between the $k$ edges the model believes in and the rest, encouraging predictions that respect the biological constraint of degree ≤ 2.
- **Why it was dropped** — two independent reasons, neither of which depends on the node type (nuclei or cell-mask fragment):
  - **The model can cheat, and the cheat was never eliminated.** Predicting near-zero probabilities everywhere minimizes the second (subtracted) term even though the first term stays high. The v3 subtraction coupling only *partially* mitigates this; it is a property of the formulation, not of the data.
  - **It measurably hurt.** At weight 2, AUC dropped 0.908 → 0.896 and fold 4 R1 returned to 0.5. With GraphNorm + label smoothing, BCE is already well-shaped and provides clear per-edge supervision; a competing degree penalty interferes. See [Findings after label smoothing](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#Findings%20after%20label%20smoothing).
- **Edge case (when it was on):** when true node degree is 0 the penalty was not evaluated, to avoid reinforcing the all-zero cheat.
- **If a structural constraint is ever needed**, it belongs in a **decode step over the predicted probabilities**, not in the loss.
- **Experimental basis:** See [Node degree loss](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#5.%20Node%20degree%20loss) for the evolution through three versions, why earlier formulations failed, and the disabling decision.

### 3. Acyclicity (implicit only)

- **How:** No explicit acyclicity term, and **no degree term either** (§2 is disabled) — the loss is BCE alone. What discourages cycles is the [visual features](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Design%20Choices.md#Visual%20branch): they give each edge much stronger *independent* evidence of being true or false, and cycles arise precisely when the model must guess between comparably-scored candidates.
- **Why:** Explicit acyclicity objectives (NOTEARS, topological potentials, Sinkhorn permutations) are either infeasible for variable-size graphs, too data-hungry for a 5-graph dataset, or mathematically self-defeating under a symmetric read-out — see [Topological DAG Constraint (Abandoned)](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Topological%20DAG%20Constraint%20(Abandoned).md). Note the degree penalty could never have forbidden a cycle anyway: degree is *local*, acyclicity is *global*, and a ring satisfies degree ≤ 2 at every node.
- **Occasional cycles** still appear but are rare. In the nuclei pipeline they were pruned manually downstream. ⚠️ For **cell fragments** cycles remain relevant — the fragment chain order encodes growth direction, and a cyclic subnetwork yields no order at all. The [inference merge](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Cell%20Mask%20Graph%20Data%20Flow.md#Inference%20merge) now **counts** them per run (`Merge/Graph_<id>_summary`) rather than preventing them; whether to add a structural decode is still open.
- **Experimental basis:** See [Acyclicity](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#7.%20Acyclicity).

## Negative edge sampling

- **How:** For every graph in every batch, sample negative (label-0) edges so the positive-to-negative ratio is fixed. Loss is computed only on the positive edges + the sampled negatives.
- **Why:** The graphs in the dataset have very different positive ratios (67% for 3-node graphs, 33% for 6-node graphs). Training on all edges means the loss landscape shifts depending on which graphs land in the training fold, making it difficult for the model to form a stable decision boundary. Fixed-ratio sampling gives the model a consistent class distribution regardless of graph mixture.
- **Why not weighted BCE — the approach this replaced:** Weighted BCE was tried *first*. It re-weights the loss by $w^{+} = N_{\text{neg}}/N_{\text{pos}}$ measured on the training fold, so the model's implied class prior remains a function of the fold's graph mixture — the mismatch changes sign rather than disappearing (train on 6-node graphs, over-predict on a balanced 3-node test graph). Sampling is different **in kind**: it does not correct a variable ratio with a coefficient, it makes the ratio a constant, leaving no fold-dependent prior to transfer. See [Weighted BCE](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#Weighted%20BCE%20—%20tried%20first,%20before%20negative%20sampling).
- **Experimental basis:** [Class imbalance](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#8.%20Class%20imbalance).

## Symmetric prediction enforcement

- **How:** Apply `enforce_symmetric_predictions` to average $P(A \rightarrow B)$ and $P(B \rightarrow A)$ before loss and metrics are computed.
- **Why:** Hyphal connections are undirected. Averaging forces the model to learn viewpoint-agnostic features because both directed predictions must agree for a true edge ("AND" constraint). The max alternative only requires one side to fire ("OR"), which is useful only when paired with directional masking tricks that are not used here.
- **Experimental basis:** [Symmetric predictions: average vs. max](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#6.%20Symmetric%20predictions:%20average%20vs.%20max).

## Optimizer setup

- **How:** Dual optimizer. `Muon` handles all 2D weight matrices (hidden-layer weights); `AdamW` handles all 1D parameters (biases, LayerNorm/GraphNorm affine parameters, classifier head's 1D params).
- **Why:** Muon provides better-conditioned updates on 2D weight matrices but is not defined for 1D tensors. Splitting parameters by dimensionality captures Muon's benefit without forcing a fallback.
- **Experimental basis:** [Optimizer choice](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#4.%20Optimizer%20choice).

## Early stopping

- **Criterion:** **Validation AUC** (not accuracy) — training AUC in the [overfit test](#Cross-validation%20vs.%20overfit%20test), which has no held-out set.
- **Why not accuracy:** Under class imbalance, a model that predicts every edge as negative can score high accuracy on the test set, and would be kept by an accuracy-based early-stopping rule. AUC is threshold-free and rewards correct *ranking* of edges by probability, so it cannot be gamed this way.
- **Mechanism:** Best validation AUC across epochs wins. The model state at that epoch is restored for final evaluation.
- **Minimum epoch floor (`min_epoch`):** The early-stopping check — both the best-model snapshot update and the patience counter — is suppressed until `epoch >= min_epoch` (currently 50). Before the floor, the model trains freely without any snapshot being saved. **Why:** In early training, the BatchNorm running stats (now replaced by GraphNorm, but the principle remains) are immature and AUC values on the small single-graph test set can be high by numerical chance — these are "fluke snapshots" where a barely-trained model's tiny logit noise happens to rank a small graph's edges in the right order. Setting a floor ensures early stopping only selects a genuinely trained model state. Removing fluke snapshots is a prerequisite for interpreting subsequent changes to the model or loss as signal rather than noise.
- **Experimental basis:** [Class imbalance](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#8.%20Class%20imbalance), [Saturated probabilities under leave-one-out CV](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#10.%20Saturated%20probabilities%20under%20leave-one-out%20CV%20(in%20progress)).

## Decision threshold

- **How:** The 0/1 decision threshold is not fixed at 0.5. It is chosen per fold to be the value that maximizes validation F1 on the best-AUC epoch.
- **Why:** Optimal operating point depends on the fold's class ratio. A fixed 0.5 threshold is miscalibrated for a fold dominated by negatives.
- **Experimental basis:** [Class imbalance](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#8.%20Class%20imbalance).

## Performance tracking

Training is monitored on two surfaces: `tqdm` for live console feedback and TensorBoard for persistent per-experiment diagnostics.

### Cross-validation vs. overfit test

Both entry points track the **same metrics** but under **different tag names**, and the held-out metrics mean different things in each:

- **`n_fold_validation`** evaluates on the held-out fold and suffixes its evaluation tags `/Test`.
- **`train_overfit_test`** builds its `eval_loader` from the *same* `train_dataset` it trains on (`gnn_train.py:793`) and suffixes its tags **`/Eval`**. There is no held-out set by design — the run answers "does the model have the capacity to fit this data at all?", so every `/Eval` metric measures **fit, not generalization**.

| | Cross-validation | Overfit test |
| --- | --- | --- |
| Evaluation tags | `Loss/Test`, `Accuracy/Test`, `AUC/Test`, `PR_AUC/Test`, `F1/Test` | `Loss/Eval`, `Accuracy/Eval`, `AUC/Eval`, `PR_AUC/Eval`, `F1/Eval` |
| Prediction diagnostics | `Diag/Pred_Mean_Test`, `Diag/Pred_Std_Test` | `Diag/Pred_Mean_Eval`, `Diag/Pred_Std_Eval` |
| Text summary | `Fold Summary` | `Overfit Test Summary` |
| Evaluated on | held-out graph | the training set itself |
| Early stopping selects on | validation AUC | **training** AUC |

Two consequences worth keeping in mind when reading an overfit run:

- **`AUC/Eval` is a training-set metric.** Early stopping still maximizes it (`gnn_train.py:816-823`), which for a capacity test means "stop once memorization plateaus" — it is not model selection against unseen data.
- **`Diag/Pred_Std_Eval` is not the saturation diagnostic.** The `_Test` version earns that role by measuring the held-out graph ([§10](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#10.%20Saturated%20probabilities%20under%20leave-one-out%20CV%20(in%20progress))); its `_Eval` counterpart cannot detect the same failure.

### Metrics and their interpretation

Tags below are given as *CV* / *overfit* where the two differ. Training-side tags (`Loss/Train_*`, `Diag/Pred_Mean`, `Accuracy/Train`, `EarlyStopping/*`) are identical in both modes.

| Metric | Definition | Why it's tracked |
| --- | --- | --- |
| `Loss/Train_Total` | `BCE + degree_penalty_weight × degree penalty`, averaged over training batches. With the degree penalty [disabled](#2.%20Sparsity-aware%20degree%20penalty%20(disabled)) (weight `0.0`) this equals `Loss/Train_BCE` | Primary optimization signal |
| `Loss/Train_BCE` | Sampled BCE component only (on the negative-sampled subset) | Diagnoses whether the classifier is learning the per-edge task |
| `Loss/Train_BCE_Unsampled` | BCE evaluated on **all** training edges (not just sampled) | Cross-checks whether the sampled loss generalizes; compare against constant-predictor floor `H(π_pos) ≈ 0.611` |
| `Loss/Train_DegreePenalty` | Sparsity-aware degree term. ⚠️ **Logs a flat `0.0`** while the penalty is [disabled](#2.%20Sparsity-aware%20degree%20penalty%20(disabled)) — it is only computed when `degree_penalty_weight > 0` | Inert in the current configuration; ignore the curve |
| `Diag/Pred_Mean`, `Diag/Pred_Std` | Mean and std of training predictions per epoch | Confirms training-side probability distribution is healthy (mean ≈ π_pos, std growing) |
| `Diag/Pred_Mean_Test`, `Diag/Pred_Std_Test`<br/>*(overfit:* `…_Eval`*)* | Same statistics on the held-out graph — or, in the overfit test, on the training set again | Key saturation diagnostic: `pred_std_test < 0.05` flags a collapse regardless of AUC. **CV only** — see [above](#Cross-validation%20vs.%20overfit%20test) |
| `EarlyStopping/Best_Epoch`, `EarlyStopping/Best_AUC` | Epoch and AUC of the saved snapshot | Cross-references the saved model against per-epoch curves; identifies fluke snapshots |
| `Accuracy/Train`, `Accuracy/Test`<br/>*(overfit:* `Accuracy/Eval`*)* | Fraction of edges correctly classified at the chosen threshold | Sanity check; misleading in isolation under class imbalance |
| `AUC/Test`<br/>*(overfit:* `AUC/Eval`*)* | Area under the ROC curve | Threshold-free ranking quality; **drives early stopping** in both modes |
| `PR_AUC/Test`<br/>*(overfit:* `PR_AUC/Eval`*)* | Area under the precision-recall curve | More informative than AUC when positives are rare; complements AUC |
| `F1/Test`<br/>*(overfit:* `F1/Eval`*)* | F1 at the threshold that maximizes validation F1 | Operating-point quality; used to pick the [decision threshold](#Decision%20threshold) |

### Console logging (`tqdm`)

- Training loops (`n_fold_validation`, `train_overfit_test`) wrap their epoch iterator in a `tqdm` bar.
- Each epoch updates the bar's postfix with: train loss, train accuracy, validation AUC, PR-AUC, F1.
- Purpose: catch obvious divergence or convergence without opening TensorBoard.

### TensorBoard logging

- **Directory layout:** the two modes differ by one level — cross-validation adds a per-fold directory, the overfit test writes a single run:
  - CV: `output/cv_experiment/<root_experiment_name>/<repeat_id>/fold_<k>`, plus a sibling `<repeat_id>/aggregate/`
  - Overfit: `output/overfit_experiment/<root_experiment_name>/<repeat_id>` — **no `fold_<k>` level**, and no `aggregate/` (there are no folds to pool)

  In both, `<root_experiment_name>` is the `experiment` string up to its first underscore, so repeats of one experiment group under a shared root.

- **`<repeat_id>/aggregate/` — the across-fold view**, written by `_log_cv_aggregate` once every fold has finished and its writer is closed:

    | Artifact | What it is |
    | --- | --- |
    | `CV/Probabilities_all_folds` | the probability violin over **every fold's held-out edges pooled**. Each edge appears exactly once, scored by a model that never trained on its graph — the honest CV picture. Folds each pick their own F1-maximizing threshold, so no single cut applies; the **mean** threshold is drawn dashed and labelled as such. |
    | `cv_summary.csv` | the per-fold metrics table — `best_epoch, auc, f1, pr_auc, threshold, pred_mean_train, pred_std_train, pred_mean_test, pred_std_test` — written automatically at the end of every CV run. |

  `cv_summary.csv` is produced by handing `summarize_cv_logs` the parent of this repeat and filtering back down to it, so the table is read from the fold event files rather than recomputed — one source of truth. Running `summarize_cv_logs.py` by hand still works and additionally pivots across repeats.

  **Why `aggregate/` is a sibling of the folds, not the repeat dir itself:** TensorBoard then lists it as a peer run (`fold_1`, `fold_2`, …, `aggregate`) instead of a confusing parent run rendered as `.`; and `summarize_cv_logs` skips it for free, because it matches fold directories on `fold_(\d+)$` and ignores anything else.
- **Scalars:** all metrics above, logged per epoch.
- **Text summaries:** one `Fold Summary` entry per fold with train/test indices, chosen threshold, and final metrics — lets you recover exact fold configuration later. The overfit test logs the equivalent under `Overfit Test Summary` (best epoch, final metrics, threshold; no indices, since it trains on everything).
- **Figures:** at final evaluation — after the best-AUC snapshot is restored — `_log_figures` renders six figures per graph. Both modes call it, so the overfit test produces them too, over its own training graphs.

    | Tag | What it shows |
    | --- | --- |
    | `Predictions/Graph_<id>` | `plot_edge_predictions` over the **microscopy image** (`data.image`), edges color-coded TP / FP / FN / TN, with predicted probabilities, attention scores (`A1`, `A2`) and RoI boxes overlaid |
    | `Predictions/Graph_<id>_no_attention` | the same overlay **minus the attention text**. Once a graph has hundreds of edges the per-edge `A1 \| A2` annotations bury the image; this keeps probabilities and boxes legible |
    | `Attention/Graph_<id>` | [parallel-coordinates plot](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Interpretation.md#Attention%20parallel%20coordinates) of each edge's two attention weights |
    | `Probabilities/Graph_<id>` | [violin plot](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Interpretation.md#Predicted-probability%20violins) of predicted probability split by true label, with the fold's threshold drawn |
    | `Interpretation/Graph_<id>` | the full embedding + attribution figure |
    | `Interpretation/Graph_<id>_sampled` | the same, heatmap restricted to a balanced edge sample |
    | `Merge/Graph_<id>` | 2×2: source image, AIS fragments, GT cells, **predicted merge** ([Inference merge](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Cell%20Mask%20Graph%20Data%20Flow.md#Inference%20merge)) |

- **Text:** `Merge/Graph_<id>_summary` carries the merged-cell count and the topology tally (`singleton` / `path` / `branched` / `cyclic`) — `branched` and `cyclic` are predictions that violate the unbranched-acyclic biology, so this is the run's constraint-violation count.
- **Sidecar files** written next to the events file, for downstream analysis outside TensorBoard:
    - `attention_graph_<id>.csv` — one row per directed edge: `edge_idx, src, tgt, a1, a2, prob, true_label, edge_class`. Built from the same DataFrame that backs the parallel-coordinates figure, so plot and CSV cannot drift.
    - `prediction_graph_<id>.graphml` — the predicted-edge graph the merge was read from, nodes carrying `ais_label` / `cell` / centroid and edges carrying `prob` / `true_label` / `edge_class` / `a1` / `a2`. GraphML rather than a pickle so igraph / Cytoscape / Gephi can open it (and networkx dropped `write_gpickle` in 3.0).
  - **The background is a channel stack, not one fixed modality.** `data.image` may carry DIC alongside one or more fluorescence channels, so `plot_edge_predictions` dispatches on shape rather than assuming a source (`gnn_train.py:353-372`):

    | `data.image` shape | Rendered as |
    | --- | --- |
    | `(H, W)` or `(H, W, 1)` | grayscale |
    | `(H, W, 2)` | composite — channel 0 (DAPI) in blue, channel 1 (DIC) in grayscale, so both are visible at once |
    | `(H, W, 3)` | passed to `imshow` as-is (RGB) |

    The 2-channel composite percentile-stretches each channel independently (1st–99th) before compositing, so a dim fluorescence channel stays visible against DIC. Nuclei-era graphs are typically single-channel DAPI; fragment graphs typically carry DIC plus fluorescence.
- **Training-embedding overlay — CV only.** The `Interpretation/Graph_<id>` PCA / PLS-DA scatter plots can overlay the *training* fold's embeddings as open dashed circles colored by true label (see [GCN Model Interpretation](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Interpretation.md)). `n_fold_validation` passes the training fold in when `log_train_embeddings=True` (default, `gnn_train.py:753`); `train_overfit_test` passes `train_dataset=None` (`gnn_train.py:865`) **by design** — its training set *is* its eval set, so the overlay would just redraw the same points.