# GCN Training Choices

> How the model is trained to achieve the desired learning outcome. Every decision below links to its supporting evidence in [GCN Model Experiments](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md).
>
> **Scope — applies to both pipelines, with one exception.** The loss, optimizers, cross-validation and normalization are **shared and live**: identical for the historical **nuclei** pipeline and the current **cell-fragment merge** pipeline. The one exception is the [node-type cross-entropy](#4.%20Node-type%20cross-entropy%20(optional)) and its [balanced node sampling](#Balanced%20node%20sampling) — **fragment-only**, because only the fragment pipeline has node-type labels. Caveat: the supporting experiments were run on nuclei data and have not been re-measured on fragments (see [GCN Model Experiments](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md)). Full breakdown: [Nuclei vs. cell-fragment](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Cell%20Mask%20Graph%20Data%20Flow.md#Nuclei%20vs.%20cell-fragment%20—%20what%20carries%20over).

## Loss

$$\mathcal{L} = \underbrace{\text{BCE}_{\varepsilon=0.1}}_{\text{always}} \;+\; \underbrace{w_{\text{deg}} \cdot \mathcal{L}_{\text{degree}}}_{w_{\text{deg}}\,=\,0\ \text{— disabled}} \;+\; \underbrace{w_{\text{node}} \cdot \mathcal{L}_{\text{node}}}_{\text{optional; } w_{\text{node}}\,=\,1.0\ \text{when on}}$$

(`gnn_train.py:289`.) In the default configuration only the first term is live, making the loss **BCE with label smoothing on sampled edges** — a pure classification objective. The degree term is **disabled** (see [§2](#2.%20Sparsity-aware%20degree%20penalty%20(disabled))), so **no structural constraint enters the loss**. The node term (§4) is an **auxiliary classification** objective, not a structural one: it constrains what the representation must encode, never which edges may exist.

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

### 4. Node-type cross-entropy (optional)

- **How:** `CrossEntropyLoss` between the [Node Classifier Head](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Design%20Choices.md#Node%20Classifier%20Head%20(optional))'s raw logits and `data.node_type`, computed on a **class-balanced subsample** of nodes (see [Balanced node sampling](#Balanced%20node%20sampling)) and added at weight `node_loss_weight`. Both runs to date used `node_loss_weight = 1.0`. Off by default (`0.0`).
- **Why combine rather than train separately:** the point is a **joint representation**. The trunk is shared, so the node gradient shapes the embeddings the edge classifier reads. That is the whole mechanism by which "a true edge cannot span background↔cell or epithelial↔hyphal" gets learned — implicitly, from the fact that one representation must serve both tasks. Two separately-trained models would share nothing and learn none of it.
- **Why the weight is 1.0:** untuned. It was set to 1.0 as the neutral starting point and never swept; the improvement was measured against a matched baseline at that value. Whether the edge task would do better at a lower weight is **unknown and untested**.
- **Guards — both sides.** Asking for a node loss that cannot be computed **raises**, rather than training with a term stuck at zero. There are two ways to ask for one impossibly, and both fail the same invisible way: `Loss/Node_Train` at 0.0 every epoch reads exactly like a converged head, while the run quietly reproduces the edge-only baseline.

    | asked for a node loss, but… | |
    | --- | --- |
    | the model has no `predict_node_type=True` | **raises** (`gnn_train.py:173`) |
    | no graph in the dataset carries `node_type` | **raises** (`gnn_train.py:185`), checked before the first epoch |
    | *some* graphs carry it | **trains** on those — a partially typed dataset is legal |

    **Backward compatibility is the `node_loss_weight=0.0` default.** The nuclei pipeline and every dataset built before this work have no `node_type` and run unchanged at the default, never reaching the guard. What is refused is asking for the loss at weight 1.0 against data that cannot supply a target.
- **Experimental basis:** [GCN Model Experiments §11](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md).

## Negative edge sampling

- **How:** For every graph in every batch, sample negative (label-0) edges so the positive-to-negative ratio is fixed. Loss is computed only on the positive edges + the sampled negatives.
- **Why:** The graphs in the dataset have very different positive ratios (67% for 3-node graphs, 33% for 6-node graphs). Training on all edges means the loss landscape shifts depending on which graphs land in the training fold, making it difficult for the model to form a stable decision boundary. Fixed-ratio sampling gives the model a consistent class distribution regardless of graph mixture.
- **Why not weighted BCE — the approach this replaced:** Weighted BCE was tried *first*. It re-weights the loss by $w^{+} = N_{\text{neg}}/N_{\text{pos}}$ measured on the training fold, so the model's implied class prior remains a function of the fold's graph mixture — the mismatch changes sign rather than disappearing (train on 6-node graphs, over-predict on a balanced 3-node test graph). Sampling is different **in kind**: it does not correct a variable ratio with a coefficient, it makes the ratio a constant, leaving no fold-dependent prior to transfer. See [Weighted BCE](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#Weighted%20BCE%20—%20tried%20first,%20before%20negative%20sampling).
- **Experimental basis:** [Class imbalance](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#8.%20Class%20imbalance).

## Balanced node sampling

Applies only when the [node-type cross-entropy](#4.%20Node-type%20cross-entropy%20(optional)) is on. `sample_balanced_nodes` (`node_sampling.py`).

- **How:** per batch, take the **rarest class present** as the anchor and draw `ratio × n_min` nodes from every present class, capped at what each class actually has. `node_sample_ratio = 1.0` (both runs to date) gives exactly equal counts. **Resampled every epoch**, so no node is permanently discarded — the majority classes rotate through.
- **Why:** exactly the argument that governs [negative edge sampling](#Negative%20edge%20sampling), applied to a 3-class problem. **The model must not learn a prior on the class distribution.** Sampling makes the ratio a constant instead of correcting a variable one with a coefficient.
- **Why not class weights:** the per-image class ratios swing enormously — background is **2.5% in image 1 and 35.6% in image 2**; hyphal is 76.0% overall but 97.5% in image 1 (see [Node Type Label Construction](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Node%20Type%20Label%20Construction.md#4.%20Node%20labels)). A weight fitted on the training folds is a **fold statistic**, so under leave-one-out it actively mismatches the held-out image rather than correcting for it — the same failure that sank [weighted BCE](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#Weighted%20BCE%20—%20tried%20first,%20before%20negative%20sampling) on the edge task. **Do not reintroduce class weighting here.**
- **"Present" is per batch, not per dataset.** Images 0 and 1 contain **no epithelial nodes at all**, so a batch drawn from them anchors on two classes only. The head is never asked to hallucinate a class the batch cannot show it, and `node_type_metrics` mirrors this by reporting only the classes present in `y_true`.

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
- **Node-type scalars** (only when the [node head](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Design%20Choices.md#Node%20Classifier%20Head%20(optional)) is on):

    | Tag | When | What |
    | --- | --- | --- |
    | `Loss/Node_Train` | per epoch | the node cross-entropy alone, before `node_loss_weight`. **Flat at zero means the dataset has no `node_type`** — see [§4](#4.%20Node-type%20cross-entropy%20(optional)) |
    | `NodeType/Accuracy_Test`, `NodeType/F1_<class>_Test`, `NodeType/Support_<class>_Test` | CV, at `best_epoch` | per-class F1 and support on the held-out graph |
    | `NodeType/Accuracy_Eval`, `NodeType/F1_<class>_Eval` | overfit, at `best_epoch` | the same over the training graphs |
    | `NodeType/Summary` (text, on `aggregate/`) | end of CV | the across-fold node-type table |

    ⚠️ **The suffix differs by mode** — `_Test` under cross-validation, `_Eval` under the overfit test — because they mean different things: `_Test` is a held-out graph, `_Eval` is the graphs the model trained on. A tag glob of `NodeType/*_Test` silently returns nothing on an overfit run.

    **Only the classes present in the held-out graph are logged.** Images 0 and 1 have no epithelial nodes, so folds testing on them emit no `F1_epithelial_Test` at all — absence of a tag is a property of the fold, not a failure. `Support_<class>_Test` is logged alongside each F1 so a score can never be read without its denominator.
- **Text summaries:** one `Fold Summary` entry per fold with train/test indices, chosen threshold, and final metrics — lets you recover exact fold configuration later. The overfit test logs the equivalent under `Overfit Test Summary` (best epoch, final metrics, threshold; no indices, since it trains on everything).
- **Figures:** at final evaluation — after the best-AUC snapshot is restored — `_log_figures` renders the figures below, per graph. Both modes call it, so the overfit test produces them too, over its own training graphs. **Not all are unconditional:** `Merge/*` needs `data.ais_labels` / `data.gt_labels`, and `NodeType/*` needs both the node head and `data.node_type`. Every figure also needs `data.image` — omit it and the overlays are skipped (with a warning; it used to be silent).

    | Tag | What it shows |
    | --- | --- |
    | `Predictions/Graph_<id>` | `plot_edge_predictions` over the **microscopy image** (`data.image`), edges color-coded TP / FP / FN / TN, with predicted probabilities, attention scores (`A1`, `A2`) and RoI boxes overlaid |
    | `Predictions/Graph_<id>_no_attention` | the same overlay **minus the attention text**. Once a graph has hundreds of edges the per-edge `A1 \| A2` annotations bury the image; this keeps probabilities and boxes legible |
    | `Attention/Graph_<id>` | [parallel-coordinates plot](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Interpretation.md#Attention%20parallel%20coordinates) of each edge's two attention weights |
    | `Probabilities/Graph_<id>` | [violin plot](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Interpretation.md#Predicted-probability%20violins) of predicted probability split by true label, with the fold's threshold drawn |
    | `Interpretation/Graph_<id>` | the full embedding + attribution figure |
    | `Interpretation/Graph_<id>_sampled` | the same, heatmap restricted to a balanced edge sample |
    | `Merge/Graph_<id>` | 2×2: source image, AIS fragments, GT cells, **predicted merge** ([Inference merge](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Cell%20Mask%20Graph%20Data%20Flow.md#Inference%20merge)) |
    | `NodeType/Graph_<id>` | 2×2: source image, AIS fragments, **true** node types, **predicted** node types — node-head runs only ([Node-type figures](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Interpretation.md)) |
    | `NodeType/Graph_<id>_edge_outcomes` | each edge's TP / FN / FP / TN outcome broken down by the **types of its two endpoints** — node-head runs only |

- **Text:** `Merge/Graph_<id>_summary` carries the merged-cell count and the topology tally (`singleton` / `path` / `branched` / `cyclic`) — `branched` and `cyclic` are predictions that violate the unbranched-acyclic biology, so this is the run's constraint-violation count.
- **Sidecar files** written next to the events file, for downstream analysis outside TensorBoard:
    - `attention_graph_<id>.csv` — one row per directed edge: `edge_idx, src, tgt, a1, a2, prob, true_label, edge_class`. Built from the same DataFrame that backs the parallel-coordinates figure, so plot and CSV cannot drift.
    - `prediction_graph_<id>.graphml` — the predicted-edge graph the merge was read from, nodes carrying `ais_label` / `cell` / centroid and edges carrying `prob` / `true_label` / `edge_class` / `a1` / `a2`. GraphML rather than a pickle so igraph / Cytoscape / Gephi can open it (and networkx dropped `write_gpickle` in 3.0).
  - **The background is a channel stack, not one fixed modality.** `data.image` may carry DIC alongside one or more fluorescence channels, so the layout is read from the shape rather than assumed. The dispatch lives in `_imshow_microscopy` (`gnn_train.py:421-447`) and is shared by all three figure families — edge predictions, the merge 2×2, and the node-type 2×2 — so they cannot render the same graph differently:

    | `data.image` shape | Rendered as |
    | --- | --- |
    | `(H, W)` or `(H, W, 1)` | grayscale, via the colormap's own autoscaling |
    | `(H, W, 2)` | composite — channel 0 (DAPI) in blue over channel 1 (DIC) in grayscale, so both are visible at once |
    | `(H, W, 3+)` | RGB from the first three channels, **each percentile-stretched independently** |

    **Why multi-channel layouts must be stretched: they bypass the colormap.** Grayscale goes through a colormap, which autoscales to the data's own range, so a raw 16-bit channel renders correctly. RGB does not — `imshow` accepts RGB only as `uint8` or float in `[0, 1]` and **silently clips** anything else (it emits `Clipping input data to the valid range for imshow with RGB data`, which is easy to miss in a training log). A 16-bit stack handed over raw therefore rendered **almost entirely white**. `_stretch_channel` (`gnn_train.py:408`) maps each channel's 1st–99th percentile to `[0, 1]` first.

    **Per channel, not jointly.** Channels differ in absolute intensity by large factors, so a joint stretch lets the brightest channel dominate and washes the others out. Stretching each independently keeps a dim fluorescence channel visible next to a bright DIC one.

    ⚠️ **This was a display bug only.** It affected the logged figures and nothing else: the features are read from the separately-built summed intensity image, never from `data.image`. No metric, feature or trained model was affected, and no run needed re-running for it.

    Nuclei-era graphs are typically single-channel DAPI; fragment graphs typically carry DIC plus fluorescence.
- **Training-embedding overlay — CV only.** The `Interpretation/Graph_<id>` PCA / PLS-DA scatter plots can overlay the *training* fold's embeddings as open dashed circles colored by true label (see [GCN Model Interpretation](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Interpretation.md)). `n_fold_validation` passes the training fold in when `log_train_embeddings=True` (default, `gnn_train.py:753`); `train_overfit_test` passes `train_dataset=None` (`gnn_train.py:865`) **by design** — its training set *is* its eval set, so the overlay would just redraw the same points.