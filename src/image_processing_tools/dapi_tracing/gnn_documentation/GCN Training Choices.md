# GCN Training Choices

> How the model is trained to achieve the desired learning outcome. Every decision below links to its supporting evidence in [GCN Model Experiments](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md).

## Loss

The loss is a composite function targeting both individual edge accuracy and structural graph constraints. It combines a classification term with a biological plausibility term.

### 1. BCE (classification) loss with label smoothing

- **How:** Binary cross-entropy between the predicted edge probability and a *smoothed* ground-truth label. Hard labels 0 and 1 are replaced by `ε` and `1 − ε` (currently `ε = 0.1`), giving soft targets 0.1 and 0.9. The formula is `smoothed = target × (1 − ε) + (1 − target) × ε`. Smoothing is applied only during training on the sampled loss indices; `test_model` always evaluates against hard labels.
- **Why label smoothing:** When train BCE falls well below 0.1, the optimizer grows weight magnitudes to push edge logits toward ±∞ (because `sigmoid(z) → 1` requires `z → +∞`). Even after fixing the BatchNorm instability with GraphNorm, the logit scale continues to grow, pushing the F1-maximizing threshold to extreme values (near 0 or 1). Label smoothing creates a finite BCE floor of `H(1−ε, ε) = −ε log ε − (1−ε) log(1−ε) ≈ 0.325` (at `ε = 0.1`) that the loss cannot go below regardless of logit magnitude. This keeps probabilities within a useful range and produces more calibrated, interpretable outputs.
- **Why not weighted BCE:** A positive weight computed from the training fold does not transfer to a test fold with a different positive ratio — see [Class imbalance](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#8.%20Class%20imbalance).
- **Experimental basis:** [Saturated probabilities under leave-one-out CV](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#10.%20Saturated%20probabilities%20under%20leave-one-out%20CV%20(in%20progress)).

### 2. Sparsity-aware degree penalty

- **How:** MSE between the true node degree $k$ and a *constructed* predicted degree:
  - Sum the top-$k$ predicted probabilities incident to the node.
  - Subtract the mean of the remaining probabilities.
  - Compare the difference to $k$.
- **Why:** Forces wide separation between the $k$ edges the model believes in and the rest. Encourages predictions that respect the biological constraint of degree ≤ 2.
- **Known failure mode:** The model can still cheat by predicting near-zero probabilities everywhere, which minimizes the second (subtracted) term even though the first term is high. Partially mitigated but not eliminated.
- **Edge case:** When true node degree is 0, the degree loss is not evaluated, to avoid reinforcing the all-zero cheat.
- **Experimental basis:** See [Node degree loss](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#5.%20Node%20degree%20loss) for the evolution through three versions and why earlier formulations failed.

### 3. Acyclicity (implicit only)

- **How:** No explicit acyclicity term. Cycles are discouraged indirectly by BCE + degree loss.
- **Why:** Occasional cycles still appear but are rare, easy to spot, and pruned manually downstream. Explicit acyclicity objectives (NOTEARS, topological potentials, Sinkhorn permutations) are either infeasible for variable-size graphs or too data-hungry for a 5-graph dataset.
- **Experimental basis:** See [Acyclicity](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#7.%20Acyclicity).

## Negative edge sampling

- **How:** For every graph in every batch, sample negative (label-0) edges so the positive-to-negative ratio is fixed. Loss is computed only on the positive edges + the sampled negatives.
- **Why:** The graphs in the dataset have very different positive ratios (67% for 3-node graphs, 33% for 6-node graphs). Training on all edges means the loss landscape shifts depending on which graphs land in the training fold, making it difficult for the model to form a stable decision boundary. Fixed-ratio sampling gives the model a consistent class distribution regardless of graph mixture.
- **Why not weighted BCE:** A positive weight computed from the training fold does not transfer to a test fold with a different ratio — see the "What did not work" note in [Class imbalance](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#8.%20Class%20imbalance).
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

- **Criterion:** **Validation AUC** (not accuracy).
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

### Metrics and their interpretation

| Metric | Definition | Why it's tracked |
| --- | --- | --- |
| `Loss/Train_Total` | BCE + degree penalty, averaged over training batches | Primary optimization signal |
| `Loss/Train_BCE` | Sampled BCE component only (on the negative-sampled subset) | Diagnoses whether the classifier is learning the per-edge task |
| `Loss/Train_BCE_Unsampled` | BCE evaluated on **all** training edges (not just sampled) | Cross-checks whether the sampled loss generalizes; compare against constant-predictor floor `H(π_pos) ≈ 0.611` |
| `Loss/Train_DegreePenalty` | Sparsity-aware degree term | Diagnoses structural plausibility independently from BCE |
| `Diag/Pred_Mean`, `Diag/Pred_Std` | Mean and std of training predictions per epoch | Confirms training-side probability distribution is healthy (mean ≈ π_pos, std growing) |
| `Diag/Pred_Mean_Test`, `Diag/Pred_Std_Test` | Same statistics on the held-out graph | Key saturation diagnostic: `pred_std_test < 0.05` flags a collapse regardless of AUC |
| `EarlyStopping/Best_Epoch`, `EarlyStopping/Best_AUC` | Epoch and AUC of the saved snapshot | Cross-references the saved model against per-epoch curves; identifies fluke snapshots |
| `Accuracy/Train`, `Accuracy/Test` | Fraction of edges correctly classified at the chosen threshold | Sanity check; misleading in isolation under class imbalance |
| `AUC/Test` | Area under the ROC curve | Threshold-free ranking quality; **drives early stopping** |
| `PR_AUC/Test` | Area under the precision-recall curve | More informative than AUC when positives are rare; complements AUC |
| `F1/Test` | F1 at the threshold that maximizes validation F1 | Operating-point quality; used to pick the [decision threshold](#Decision%20threshold) |

### Console logging (`tqdm`)

- Training loops (`n_fold_validation`, `train_overfit_test`) wrap their epoch iterator in a `tqdm` bar.
- Each epoch updates the bar's postfix with: train loss, train accuracy, validation AUC, PR-AUC, F1.
- Purpose: catch obvious divergence or convergence without opening TensorBoard.

### TensorBoard logging

- **Directory layout:** `output/cv_experiment/<root_experiment_name>/<repeat_id>/fold_<k>` (or `output/overfit_experiment/...`).
- **Scalars:** all metrics above, logged per epoch.
- **Text summaries:** one `Fold Summary` entry per fold with train/test indices, chosen threshold, and final metrics — lets you recover exact fold configuration later.
- **Figures:** at final evaluation, `plot_edge_predictions` renders the test-set predictions over the original DAPI image, color-coded by TP / FP / FN / TN, with predicted probabilities and intermediate GCN attention scores (`A1`, `A2`) overlaid. Logged under `Predictions/Graph_<id>`. These are the main diagnostic for understanding *why* a fold succeeded or failed.