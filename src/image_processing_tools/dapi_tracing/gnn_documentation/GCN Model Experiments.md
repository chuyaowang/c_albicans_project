# GCN Model Experiments

> Experiments done to optimize the model and training protocol. Documents what worked, what did not, and the reasoning behind each outcome. Design and training choices documented in [GCN Design Choices](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Design%20Choices.md) and [GCN Training Choices](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Training%20Choices.md) trace back to the sections below.

## 1. Baseline and sanity checks

### Overfitting a single graph

- **Setup:** Train the model on one manually labeled graph to confirm the architecture has enough capacity to fit the task.
- **Result:** Model reaches 0 loss and 100% accuracy.
- **Decision:** Confirms the pipeline (features → message passing → classifier → loss) is wired correctly before moving to multi-graph experiments.

### Rotation augmentations

- **Setup:** Rotate one graph in 4 directions and perform four-fold leave-one-out cross validation. Used as a minimal generalization test.
- **Result:** The model fits all rotations successfully, but only after some [Node features](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Data%20Flow.md#Node%20features) and [Edge features](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Data%20Flow.md#Edge%20features) were redesigned to be rotation- and translation-invariant (absolute coordinates and absolute angles were replaced with relative quantities).
- **Decision:** Confirms invariances must be built into the features, not learned from data. Kept the rotation-invariant feature set going forward.

### Five-graph cross validation

- **Setup:** Five manually labeled graphs (3 easy straight-chain graphs, 2 harder graphs with turns). Five-fold leave-one-out CV, repeated 3 times.
- **Result:** Model can overfit when all 5 graphs are trained together, but under CV it predicts biologically implausible structures — cycles and nodes with degree > 2.
- **Decision:** Motivated the [degree loss](#5.%20Node%20degree%20loss) and the [class-imbalance handling](#8.%20Class%20imbalance). Remains the standard evaluation protocol for downstream experiments.

## 2. Architectural choices

### Non-linearities and MLP depth

- **Setup:** Added non-linearities by inserting [MLP Module](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Design%20Choices.md#MLP%20Module)s into the message function, [The Edge Updater](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Design%20Choices.md#The%20Edge%20Updater), and [Classifier Head](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Design%20Choices.md#Classifier%20Head).
- **Result:** Model became more capable of learning, but with more parameters it began to memorize the training set — training accuracy climbed significantly above test accuracy.
- **Decision:** Keep the MLP blocks but add dropout (`p = 0.5`) inside them. Exception: no dropout in the first GCN layer, because the full raw feature vector must reach the first message-passing step.

### Hidden layer size

- **Setup:** Swept hidden dimension from 32 → 64 → higher.
- **Result:** 32 → 64 gave a clear performance bump. Larger sizes slowed training and began to overfit without clear accuracy gain.
- **Decision:** Hidden size fixed at 64.

### Weight initialization

- **Setup:** Compared default PyTorch init with Kaiming Normal (for ReLU-activated hidden layers) and Xavier/Glorot Uniform (for the final linear projection).
- **Result:** Kaiming + Glorot converged faster and reached higher accuracy.
- **Decision:** Kaiming Normal baked into `CustomLazyLinear`; Glorot on the final linear layer.

### Residual connections

- **Setup:** After each GCN layer and each Edge Updater, concatenate the *original* raw node/edge features with the updated embeddings, followed by `LazyBatchNorm1d`.
- **Result:** Performance improved and training became more stable.
- **Why it works:** Repeated message passing causes oversmoothing (embeddings blur together). Reinjecting the raw features guarantees every depth has direct access to the unmodified biological evidence. The following BatchNorm re-centers the concatenation, which otherwise mixes tensors with very different scales.
- **Decision:** Residual-then-norm pattern is a standard component.

### Dynamic edge weight (attention)

- **Setup:** Replaced fixed-weight aggregation with the [Attention Mechanism (Attn-MLP)](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Design%20Choices.md#2.%20Attention%20Mechanism%20(Attn-MLP)), which outputs a softmax-normalized scalar per edge.
- **Result:** Performance improved.
- **Why it works:** The softmax forces candidate edges pointing at the same target node to *compete*, letting the model learn to suppress noisy background edges before they enter the node state. It also removes the need for the rigid $1/\sqrt{deg(i)deg(j)}$ normalization used in vanilla GCN, which is ill-suited to a fully connected candidate graph.
- **Decision:** Attn-MLP is the default aggregation weighting.

### Self loops

- **Setup:** Trained the model with and without self loops in `edge_index`.
- **Result:** Adding self loops degraded performance. The network likely could not cleanly separate "update from my own features" from "update from a real inter-cellular connection," blurring the role of the Edge Updater.
- **Caveat:** These experiments were run on a much simpler model and have not been repeated on the current architecture.
- **Decision:** Self loops omitted.

## 3. Feature engineering

### Adding angle-based edge features

- **Setup:** Added three new edge features: each node's relative angle to the edge, the minimum of the two relative angles to the edge, and the difference between the two node angles.
- **Result:** Performance improved.
- **Why it works:** Hyphal growth is directional — a true connection tends to emerge close to the major axis of both nuclei. Encoding this alignment explicitly gives the classifier a strong geometric prior that would otherwise have to be learned implicitly from cell shape and path intensity. When the cell is straight, the two nuclei would have about the same direction. When the hypha makes turns, at least one node will be aligned to path, and the minimum of the two node angles will catch that.
- **Decision:** Angle features are part of the standard edge feature set.

### Normalization of geometric features

- **Setup:** Normalize angles by $\pi/2$ (bounding them to $[0,1]$) and normalize edge length by the image's `avg_nucleus_length`.
- **Result:** Both improved performance.
- **Why it works:**
  - Angles are bounded and have physical meaning at the extremes (0 = aligned, 1 = perpendicular). Z-scoring would center the mean and stretch the variance, destroying those bounds.
  - Raw pixel distance does not generalize across microscope magnifications; dividing by the average nucleus length converts "pixels" into "cell lengths," a universal biological metric.
- **Decision:** Angles divided by $\pi/2$, length divided by `avg_nucleus_length`. Both are excluded from Z-score normalization (see [GCN Data Flow](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Data%20Flow.md#Edge%20features)).

### What did not work

- **Within-graph normalization** (dividing features by each graph's own mean instead of Z-scoring on the training fold): with only 3–6 nodes per graph, a single outlier (e.g., a merged artifact) drags the graph mean far from the true value. Identically sized nuclei end up with completely different normalized values depending on their neighbors. Performance dropped.
- **Node degree as a feature:** did not help. The graph is fully connected so all node degrees are nearly identical, and the structural information the feature is supposed to encode is not informative for this task anyway.

## 4. Optimizer choice

- **Setup:** Split parameters between two optimizers — Muon for 2D weight matrices (hidden layers), AdamW for 1D parameters (biases, batch norm affine params, classifier head).
- **Result:** Performance improved versus a single-optimizer setup.
- **Why it works:** Muon provides better-conditioned updates for 2D weight matrices but is not defined for 1D tensors; routing 1D params through AdamW gets the benefits of Muon where it applies without forcing a fallback.
- **Decision:** Dual optimizer is standard.

## 5. Node degree loss

- **Issue observed:** Nodes predicted with more than 2 true edges. Biologically impossible — chain endpoints have degree 1, interior nodes degree 2.
- **Current (v3) formulation:** sum the top-$k$ predicted probabilities, subtract the mean of the remaining probabilities, and compare the result to $k$ via MSE. This forces large separation between the top-$k$ edges and the rest.
- **Result:** False positives reduced. The model still occasionally cheats by predicting all near-zero probabilities (this minimizes the second term but makes the first term high). Partially mitigated but not eliminated.
- **Boundary case:** When the true node degree is 0, the degree loss is not evaluated. This matters because otherwise the only way to match a true degree of 0 is to predict all-zero edges, which reinforces the cheating failure mode.
- **Decision:** Keep v3 degree loss as part of the composite objective.

### What did not work

- **v1 — sum of all incoming probabilities vs. target degree:** for nodes with many candidate edges, many small probabilities can sum to the target value. No pressure on sparsity.
- **v2 — top-$k$ sum + mean of the rest, as two separate MSE terms summed together:** same cheating-by-zero failure as v3 but without the subtraction coupling. Also tried summing (instead of averaging) the non-top-$k$ probabilities: for nodes with many false candidates, a pile of small probabilities still sums to a high value and dominates the loss. Averaging is better.

## 6. Symmetric predictions: average vs. max

- **Issue observed:** Forward edge $A \rightarrow B$ and reverse edge $B \rightarrow A$ produce conflicting probabilities even though a physical connection has no direction.
- **Interpretations:**
  - *Edge as a connection*: connection holds regardless of viewing direction. Both predictions should agree.
  - *Edge as "growth from"*: directional; only one side is canonical.
- **Options and implications for gradient flow:**
  - **Average:** gradient is split equally between the two directed edges, pulling them toward the same value. A true edge requires *both* directions to predict high probability (an AND constraint).
  - **Max:** gradient flows only to the larger of the two predictions; the other is ignored. `torch.max` is differentiable ([see](https://discuss.pytorch.org/t/confused-about-torch-max-and-gradient/14283)) so backprop is intact. A true edge needs only *one* direction to predict high (an OR constraint).
- **Decision:** Average. Performance difference was small, but averaging aligns with the connection interpretation and forces the model to learn viewpoint-agnostic features.

### What did not work

- **Similarity loss** ($\text{MSE}(P_{A \to B}, P_{B \to A})$): the value is typically tiny, giving almost no gradient signal. Adding a scale weight fixes the magnitude but introduces another hyperparameter to tune, and the averaging approach already enforces the same property for free.

## 7. Acyclicity

- **Issue observed:** Model occasionally predicts cycles between nuclei, which is biologically impossible.
- **Current approach:** Rely on BCE + degree loss to discourage cycles implicitly. Residual cycles are rare, easy to spot, and can be pruned manually in the downstream analysis pipeline.
- **Known alternative not attempted:** A NOTEARS-style acyclicity loss. This requires an augmented Lagrangian training loop, which is likely too complex and data-hungry for our 5-graph dataset. Not pursued.

### What did not work

- **Sinkhorn permutation matrix** to rank nodes: requires a fixed `[N_nodes, N_ranks]` matrix, incompatible with our variable-size graphs.
- **Scalar "topological potential" per node** (connection flows only from high to low potential, e.g., `[1, 0.8, 0.7, …, 0]`): fails because the graph is bidirectional — every downhill edge has a reverse uphill edge. Combined with [max symmetry](#6.%20Symmetric%20predictions:%20average%20vs.%20max) (required for directional masking to have any bite), one of the two directed edges is always kept, and final classification reduces to the predicted probability alone. The potential becomes meaningless.
- **General takeaway:** forcing an undirected acyclic structure purely through directed topological potentials is not viable in this setup. Abandoned.

## 8. Class imbalance

- **Issue observed:** Model predicts almost everything as negative on larger graphs.
- **Root cause:** Positive-to-negative edge ratio varies dramatically with graph size (fully connected candidate graphs):
  - 3-node graph: 2 true / 3 total (67% positive)
  - 4-node graph: 3 true / 6 total (50% positive)
  - 6-node graph: 5 true / 15 total (33% positive)
- **Consequence for CV:** The positive ratio in the training fold does not match the test fold, so any statistic derived from the training distribution (e.g., class weights) is miscalibrated at test time.
- **Working fix — negative edge sampling:** For each graph in each batch, sample negative edges so the positive:negative ratio is fixed. Loss is computed only on sampled positives + sampled negatives.
- **Working fix — AUC-based early stopping:** Accuracy rewards "predict everything negative" under class imbalance. AUC does not, so early stopping tracks validation AUC.
- **Working fix — F1-maximizing threshold:** The 0/1 decision threshold is chosen per fold to maximize validation F1 rather than fixed at 0.5.

### What did not work

- **Weighted BCE:** the positive weight computed from the training fold does not transfer to the test fold because the class ratio varies across graphs. Gave inconsistent results across folds. Do not revisit.

## 9. Visual features

### Overfit test

- Can overfit on the training data.

### Adding visual feature

- Add the visual features in general improved performance compared to no visual features.
- The model can better identify turns.

### D_visual

- This parameter controls the vector length of the transformed visual feature, concatenated with the original node and edge dimensions.
- Tried: 16 and 32
- Small improvement.

### Hidden channel numbers

- Tried: 64 and 128
- More channels lead to better performance

### Channels to use for computing visual features

- Tested using DAPI+DIC and only DIC for microsam features
- The idea was that the direction and overlaps of the C. albicans cells are best seen in DIC images. Adding DAPI may obfuscate the information in the DIC channel.
- At dVisual=32, h=64: The result is almost a tie - sometimes using the DAPI+DIC features is better. Sometimes using just the DIC features is better.
- At dVisual=32, h=128: Using DAPI+DIC is (slightly) better most of the time.
  - For graph 1, using DIC only is (slightly) better. Predictions are the same but AUC and F1 scores are slightly higher (0.05).
  - For other graphs, using DAPI+DIC is (slightly) better (0.02).
  - Reason, graph 1 contains a 180 degree turn.
- Points of optimization for the future:
  - Incorporate fluorescence: always have 3 channels.
  - Use DAPI and DIC independently and let the model choose which one to use more - more computation upfront but it's a one-time operation.
  - Joint training of microsam encoder + GCNN so the features extracted are more relevant for our 2-channel data - requires more compute. microsam encoder takes 5G VRAM, GCNN takes 1.5G; I only have 6G VRAM.
  - Use DIC features for edges, DAPI+DIC for nodes.

### Correcting for DIC Shift

- The DIC channel has a shift of 15 px to the right and 15 px to the downside. Initially this shift was neglected.
- After correction the model performance increased.

## 10. Saturated probabilities under leave-one-out CV (in progress)

### Terminology

- **Saturation.** A sigmoid output is *saturated* when its input logit is large in magnitude (|logit| ≳ 5), so the output sits in a narrow band near 0 or near 1 with very little spread. A saturated prediction can still be *correct* in a rank sense (positives slightly above negatives within the band), which is why rank-based metrics (AUC, PR-AUC) can stay high even when the absolute probabilities are useless.
- **Saturation direction.** Which sigmoid tail the predictions get pushed into. *Low saturation* means all predictions cluster near 0 (the model is "predicting everything negative"); *high saturation* means all predictions cluster near 1 ("predicting everything positive"). The two are not symmetric in their downstream consequences — low saturation looks like missed connections, high saturation looks like spurious connections.
- **Collapse.** Used here interchangeably with saturation when describing the held-out graph. "Collapsed to 0" = low saturation, "collapsed to 1" = high saturation.
- **Constant-predictor floor.** The BCE achieved by always predicting the dataset's positive prevalence $\pi_{\text{pos}}$. Equals the binary entropy $H(\pi_{\text{pos}}) = -\pi \log \pi - (1-\pi) \log(1-\pi)$. Any BCE above this floor means the model is doing nothing better than reporting the prior.
- **Best-epoch snapshot.** The model state with the highest validation AUC seen so far; this is what early stopping saves and what the final fold metrics are computed on. A *fluke snapshot* is a best-epoch save where AUC is high but predictions are saturated — i.e., a high-AUC moment driven by tiny numerical noise inside a saturated band rather than genuine learning.

### Issue observed

Under 6-fold leave-one-out CV on the 6-graph visual-feature dataset (3 repeats), the F1-maximizing decision threshold collapses to extreme values — either `0.0000`, `1.0000`, or values like `0.9898`. Overlay plots confirm the per-graph predictions are bunched in a narrow band near 0 or near 1. AUC, PR-AUC, and F1 still report >0.8 on harder graphs and >0.9 on easier ones, because those metrics survive saturation as long as positives are ranked above negatives within the band.

The first reading was that this is the same "predict almost all 0s or all 1s" failure that happened in the [Node degree loss](#5.%20Node%20degree%20loss) and [Class imbalance](#8.%20Class%20imbalance) fixes. Setting `degree_penalty_weight=0` did not eliminate the symptom. Confirmed in code that this fully disables the degree term (`gnn_train.py` short-circuits the computation when the weight is 0 and multiplies by 0 in the total loss), so the degree loss is not the trigger.

### Diagnostic instrumentation added

To distinguish a real loss-driven collapse from eval-time saturation, the following scalars were added to `gnn_train.py`:

- `Loss/Train_BCE_Unsampled` — BCE evaluated on **all** training edges, not just the negative-sampled subset. Compared against the constant-predictor floor $H(\pi_{\text{pos}}) \approx 0.611$ for $\pi_{\text{pos}} \approx 0.3$.
- `Diag/Pred_Mean` and `Diag/Pred_Std` — mean and std of training predictions per epoch, weight-averaged by `data.num_graphs`.
- `Diag/Pred_Mean_Test` and `Diag/Pred_Std_Test` — same statistics on the held-out graph at evaluation time.
- `EarlyStopping/Best_Epoch` and `EarlyStopping/Best_AUC` — the epoch the snapshot was taken from, so the saved-model state can be cross-referenced against the per-epoch curves.

A helper script `summarize_cv_logs.py` reads these scalars plus the `Fold Summary` text out of each fold's events file and produces a one-row-per-(repeat, fold) table with `(repeat, fold, best_epoch, auc, f1, pr_auc, threshold, pred_mean_train, pred_std_train, pred_mean_test, pred_std_test)`. The script also exposes `pivot_by_fold` (and a `--by-fold` / `--from-csv` CLI mode, plus a `wrangle_csv(input_csv, output_csv)` helper) to flatten the table into one row per fold with each repeat's metrics side-by-side — that is the form used for the cross-repeat analysis below.

### Findings on the training side

- Both `Loss/Train_BCE` and `Loss/Train_BCE_Unsampled` decrease to <0.1 — well below the constant-predictor floor. The model genuinely fits per-edge structure on training, it is **not** collapsing to a constant on the training set.
- Training `Diag/Pred_Mean` settles near 0.3 (matching $\pi_{\text{pos}}$), and `Diag/Pred_Std` grows over training — consistent with class separation on the training fold.

### Findings on the held-out side

- `Diag/Pred_Std_Test` oscillates wildly within a single fold across epochs — typical range 0.2–0.4, with frequent excursions down to ~1e-4 and back up to ~0.45. The held-out forward pass is not stably saturated; it flips in and out epoch by epoch.
- Across 18 folds (3 repeats × 6 folds), saturation flags appear in 14/18 folds. Saturation flag used: `threshold < 0.01 or threshold > 0.99`, or `pred_std_test < 0.05`.
- **Saturation direction is not graph-determined.** The same held-out graph collapses *low* in one repeat and *high* in another — fold 5 across the three repeats: `pred_mean_test` of 0.27 / 0.03 / **0.997**, and threshold 0.265 / 0.0002 / **0.9994**. Fold 1: collapses to ~0 in repeat 1 (`pred_mean_test = 3e-19`) but to a high-mean bimodal distribution (mean ≈ 0.7) in repeats 0 and 2. If a single train-vs-test feature distribution shift caused the saturation, the direction would be reproducible across repeats — it is not.
- **High AUC + saturated predictions correlates with very early best epochs.** Repeat 1 fold 1 has `best_epoch = 13` with predictions on the order of 1e-19 and AUC = 1.0. Other suspicious early best epochs: repeat 0 fold 2 (36), repeat 1 fold 2 (21), repeat 2 fold 4 (27), repeat 2 fold 5 (36). All saturated, all extreme thresholds. Early stopping is preserving fluke snapshots — moments where a barely-trained model's tiny numerical noise happens to rank a small graph's edges in the right order.

### Findings on the train-vs.-held-out comparison

The `pred_mean_train` / `pred_std_train` columns added to the summary table allow a direct comparison between the trunk's behaviour on the training set and on the held-out graph at the *same saved epoch*. Across the 18 folds:

| | min | median | max |
|---|---|---|---|
| `pred_mean_train` | 0.32 | 0.37 | 0.58 |
| `pred_std_train` | 0.31 | 0.42 | 0.45 |
| `pred_mean_test`  | 1e-19 | 0.34 | 0.997 |
| `pred_std_test`   | 0 (~1e-18) | 0.36 | 0.49 |

Three concrete reads:

- **Train side is uniformly healthy.** `pred_mean_train` clusters tightly around the prior $\pi_{\text{pos}} \approx 0.3$–0.4 and `pred_std_train` is well above any saturation cutoff (always >0.30) on every single fold. The trunk fits the training graphs cleanly at the saved epoch, regardless of how the held-out graph behaves.
- **Eval side is wildly bimodal.** `pred_mean_test` and `pred_std_test` span almost their entire valid range across folds, and 14 of 18 folds trip the saturation flag (`pred_std_test < 0.05` OR `threshold < 0.01` OR `threshold > 0.99`). Repeat 0 saturates on 2/6 folds; repeats 1 and 2 saturate on all 6.
- **Same weights, divergent forward passes.** Examples like repeat-0 fold-2 (`pred_std_train = 0.39`, `pred_std_test = 0.05`) show the trunk producing healthy spread on training graphs and a saturated band on the held-out graph *at the same saved checkpoint*. The only mechanism in the architecture that can produce divergent forward passes from identical weights is a layer that uses different statistics at train vs. eval — i.e. **BatchNorm**. This is the strongest evidence so far that BN running stats are the root cause.

The single train-side outlier — repeat-1 fold-1 with `pred_mean_train = 0.58`, `pred_std_train = 0.31`, `best_epoch = 13` — is the same fluke snapshot already flagged in [Findings on the held-out side](#Findings%20on%20the%20held-out%20side). It supports the min-epoch floor recommendation: at epoch 13 the trunk hasn't fully shaped its training-side distribution either, so the snapshot is not a reflection of the trained model.

### Refined hypothesis

The held-out predictions on a saturated fold are dynamically unstable, not statically wrong. The plausible mechanism is now:

1. **`LazyBatchNorm1d` running stats drift each step.** Inside the [Residual connections](#Residual%20connections) and [Classifier Head](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Design%20Choices.md#Classifier%20Head), running mean/var update on every training step. With 5 small heterogeneous graphs per training fold (LOO holds out 1 of 6), those stats represent no single graph's distribution well, and they shift slightly each epoch.
2. **Saturated logits amplify any offset.** The detailed mechanism — what "trunk," "large logits," "uniform shift," and "linear stack" mean here — is unpacked in the next subsection.

#### How a small BN drift becomes a saturated prediction

- **Trunk.** Everything between the input features and the final sigmoid: the GCN convs, [The Edge Updater](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Design%20Choices.md#The%20Edge%20Updater)s, the residual+BatchNorm blocks, and the [Classifier Head](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Design%20Choices.md#Classifier%20Head)'s linear layers. The trunk's output is the *logit* — a single pre-sigmoid scalar per edge. The logit is then passed through `sigmoid(z) = 1 / (1 + e^{-z})` to produce the probability the loss is computed against.
- **Why logits become large when train BCE → 0.** Per-edge BCE is `-log(p)` for a positive and `-log(1 − p)` for a negative. To drive BCE near 0 you need `p → 1` for positives and `p → 0` for negatives, which because `p = sigmoid(z)` requires `z → +∞` for positives and `z → -∞` for negatives. Concretely, train BCE ≈ 0.1 corresponds to `|z| ≈ 2–3`; train BCE ≈ 0.01 to `|z| ≈ 5`; further reduction requires the trunk to keep growing the magnitude of its weights so it can produce ever-larger logits. So "train BCE < 0.1" is a direct statement about the trunk's weight scale.
- **What the small shifts are.** With 5 training graphs and `batch_size = 5`, every training step sees the same batch composition, so the source of running-stat drift is *not* batch-composition variance. Instead, BN's per-batch stats are functions of the current model parameters: each gradient step updates upstream weights, which changes the activations entering each BN layer, which changes `batch_mean` and `batch_var`. PyTorch BN updates the running stats as an exponential moving average of those per-batch values (`running_x ← (1 − m) · running_x + m · batch_x`, default `m = 0.1`), so the running stats are essentially a low-pass-filtered trajectory of the model's own internal feature distribution as it learns — they keep chasing a moving target throughout training. At evaluation, BN applies `(x − running_mean) / sqrt(running_var + eps) · γ + β` to the held-out graph. The held-out graph never contributed to those running stats, so its activations are normalized by parameters that were tuned to the training-batch's activation distribution at the current point in training. The mismatch produces a per-channel offset; if `running_mean` differs by `Δ` from the held-out graph's true channel mean, the BN output picks up an offset of approximately `−γ · Δ / sqrt(running_var + eps)`. The absolute size of this offset is small (a few hundredths of a unit per channel) but is the dominant epoch-to-epoch perturbation seen by the trunk on a held-out graph, and its sign is essentially random across epochs and repeats.
- **Why the shift is uniform across edges of the held-out graph.** BN normalizes per *channel* using a single (mean, var, γ, β) tuple per channel. Every node and every edge embedding in the held-out graph passes through the same BN with the same per-channel parameters, so the same per-channel offset is added to all of them. From the perspective of the rest of the trunk, the held-out graph receives a graph-wide shift in feature space — it is *not* a per-edge noise that averages out.
- **What "multiplied through the linear stack and slammed into a sigmoid tail" means.** After the BN block, the activations still pass through several linear layers (more GCN convs, more MLP layers, the final classifier projection). Each linear layer multiplies its input by a weight matrix; from the BCE-→-0 argument above, those weight matrices have grown to large magnitudes. A small uniform shift `δ` at the BN output is therefore multiplied by the spectral norm of every downstream linear layer in turn, so by the time it reaches the classifier head it has been amplified by the *product* of those norms — easily an order of magnitude or more. That amplified shift adds (positively or negatively) to the already-large logit, pushing it deeper into one of the two sigmoid tails. Because `sigmoid` is essentially flat for `|z| > 5`, the resulting probabilities collapse into a narrow band near 0 or near 1 — the saturated state observed in the overlay plots. The sign of the shift in any given epoch is essentially random (it follows whichever way the BN running stats happened to drift), so different epochs and different repeats land in the *low* tail or the *high* tail unpredictably, exactly matching the observed cross-repeat flips.

The earlier "feature distribution shift" framing was partially right (BN running stats *are* the source) but partially wrong: it predicted a *graph-determined* saturation direction, which the data falsifies.

### Findings after min_epoch floor experiment

`min_epoch=50` was added to `n_fold_validation` (and `train_overfit_test`) as a floor below which neither the best-epoch snapshot is updated nor the patience counter is incremented. The same 3×6 CV was re-run (`visual_dVisual32_h128_fixDICShift_trimNeg10_noDegLoss_minEpochFloor`).

**Floor is working.** All 18 best epochs now fall in the range 50–147. The fluke-snapshot category (previously 5 folds with `best_epoch ≤ 36`) is eliminated.

**Train side remains uniformly healthy.** `pred_std_train` stays in 0.397–0.464 across all 18 folds, minimum raised from 0.306 to 0.397. This confirms the training distribution is always well-shaped regardless of which graph is held out.

**Saturation by threshold reduced from 14/18 to 10/18.** However, the remaining 10 split into two qualitatively different categories:

- **Benign large-logit saturation (e.g. fold 1, all 3 repeats).** Threshold ≈ 1.0, but `pred_std_test ≈ 0.44–0.47` (spread is healthy) and AUC/F1 ≈ 1.0. The model is producing a bimodal held-out distribution whose positive-edge tail is clipped to exactly 1.0 by the sigmoid, which is why the F1-maximizing threshold lands at 1.0. This is large-logit correct classification, not a collapse — the rank ordering is preserved.
- **Genuine collapses (std-based).** `pred_std_test < 0.10` on: fold 3 R1 (0.071), fold 5 R1 (0.150), fold 5 R2 (0.080). These are true saturation events where all held-out edges pile into a single narrow band.

**Fold 4 remains the hardest and most unstable fold.** All three repeats saturate (R0/R1 collapse high, R2 collapses low), and AUC ranges from 0.667 to 0.917. The cross-repeat sign flip persists, consistent with the BN running-stat instability hypothesis — the direction of the per-channel offset is determined by the state of the running stats at the saved epoch, which is random across repeats.

**Folds 5 and 6 improved.** Fold 6 now has healthy thresholds in R0/R1 (0.57, 0.564) and fold 5 R0 produces a threshold of 0.188 — all cases that were saturated in the pre-floor run. The floor's removal of fluke snapshots is sufficient to stabilise the easy folds; the harder folds (fold 4 especially) still require the structural norm fix.

**Interpretation.** The floor successfully removes noise introduced by evaluating barely-trained models. The residual saturation on hard folds is not caused by fluke snapshots — it happens at epochs 50–105 with a fully trained trunk. This confirms the BN running-stat instability is the underlying mechanism, not just unlucky early stopping.

### Normalization alternatives: BatchNorm, LayerNorm, GraphNorm

Understanding the differences between normalization strategies is necessary before replacing `LazyBatchNorm1d` in the trunk.

#### BatchNorm (`LazyBatchNorm1d`)

BatchNorm normalizes activations across the **batch dimension**: for each channel (feature index), it computes the mean and variance over all samples in the current mini-batch and normalizes each sample's activation by those batch-wide statistics. After normalization, it applies learned per-channel scale `γ` and shift `β`.

At **training time** it uses live batch statistics. At **evaluation time** it switches to running estimates of the mean and variance accumulated throughout training via an EMA. This is the source of the saturation problem: the running estimates are calibrated to the training distribution, which does not match the held-out graph's activation distribution. The mismatch introduces a per-channel offset that gets amplified by downstream large-weight linear layers into a saturated sigmoid output. The direction of the offset changes as running stats drift, producing the observed cross-epoch and cross-repeat sign flips.

**BatchNorm is cheap** (one global mean/var per channel, computed once per step) and stabilizes early training, which is why it is the default. But it is fundamentally a **batch-statistic** norm: it requires the batch to be representative of the true data distribution. With mini-batches of 5 small heterogeneous graphs and a single held-out graph never seen during training, that condition does not hold.

#### LayerNorm

LayerNorm normalizes each **sample independently** across the feature dimension: given an embedding vector of size `d`, it computes the mean and variance over the `d` features *within that single vector* and normalizes. No running statistics are accumulated, and behavior is identical at train and eval time.

For a node or edge embedding of shape `[N_items, hidden_dim]` (as in the trunk), LayerNorm operates over the `hidden_dim` axis for each item row separately. Every node and every edge in both the training fold and the held-out graph is normalized by its own feature-wise statistics — there is no cross-item or cross-graph coupling. This eliminates the running-stat mismatch entirely.

The trade-off is that LayerNorm does nothing to align the scale of activations across different nodes or graphs, since it only looks within each individual embedding. In practice for GNNs this is usually acceptable, and in our setting it is preferable because the training fold and the held-out fold never interact during normalization.

#### GraphNorm (PyG)

GraphNorm (Cai et al. 2021, implemented in `torch_geometric.nn.norm.GraphNorm`) is a graph-specific variant designed to address a known failure mode of vanilla InstanceNorm (per-graph normalization) on variable-size graphs. It normalizes activations per graph, but includes a learned per-channel ratio `α ∈ [0, 1]` that controls how much of the graph-level mean is subtracted. The update is:

```
ĥ = (h − α · mean(h_graph)) / std(h_graph)
y = γ · ĥ + β
```

When `α = 1` this is standard InstanceNorm (subtract the full graph mean). When `α = 0` it only divides by std, leaving the mean untouched. The network learns `α` during training.

Like LayerNorm, GraphNorm accumulates no running statistics and has identical train/eval behavior. Unlike LayerNorm, it normalizes across all nodes (or all edges) within a single graph rather than within each individual embedding vector — it captures graph-level statistics. This may be more meaningful for message-passing architectures, where node representations are interdependent within a graph and their global mean carries structural information. The learnable `α` allows the model to decide how much to suppress that global mean.

**Summary table:**

| | Normalizes over | Running stats? | Train/eval asymmetry? | Graph-aware? |
|---|---|---|---|---|
| BatchNorm | Batch (all items × all graphs in one step) | Yes | Yes (root cause here) | No |
| LayerNorm | Feature dim of each item independently | No | No | No |
| GraphNorm | All items within one graph | No | No | Yes (`α`) |

For this project's use case — LOO CV with a single held-out graph, `batch_size = 5` training graphs — both LayerNorm and GraphNorm eliminate the train/eval mismatch. GraphNorm may capture more useful inductive structure (graph-level normalization), while LayerNorm is simpler and has no extra learned parameters. Both are viable candidates for the norm swap.

### Findings after GraphNorm + LayerNorm swap

`LazyBatchNorm1d` was replaced throughout `simple_gnn.py`:

- **Inside MLP Sequential blocks** (`GCNConv.mlp`, `GCNConv.update_mlp`, `EdgeUpdater.mlp`, `Classifier.mlp`, `FusionMLP.mlp`): replaced with `LayerNorm(out_channels)`. `GraphNorm` cannot be used here because `Sequential` has no mechanism to pass a `batch` vector; `LayerNorm` normalizes per-item across the feature dimension with identical train/eval behavior.
- **Residual skip-connection norms** (`norm_x1`, `norm_e1`, `norm_x2`, `norm_e2` in `Model.forward()`): replaced with `GraphNorm(hidden_channels + raw_feature_dim)`. These are called in `forward()` where `data.batch` (node batch vector) and `edge_batch = batch_vec[edge_index[0]]` (edge batch vector) are available. `GraphNorm` normalizes per graph: during training each of the 5 subgraphs in the batch is normalized within itself; during eval the single held-out graph is normalized as one unit. No running statistics in either case.

The same 3×6 CV was re-run (`visual_dVisual32_h128_fixDICShift_trimNeg10_noDegLoss_minEpochFloor_swapNorm`).

**True collapses eliminated: 2/18 → 0/18.** `pred_std_test` minimum went from 0.071 to 0.364. Every held-out graph across all folds and all repeats now produces a spread prediction distribution. The values are notably uniform: all 18 cells land in 0.364–0.462, compared to the 0.071–0.474 range under BN.

| | True collapses (`pred_std_test < 0.10`) | Saturated by threshold (`<0.01` or `>0.99`) | `pred_std_test` min | AUC mean |
|---|---|---|---|---|
| BN (no floor) | many | 14/18 | ~0 | — |
| BN + floor | 2/18 | 10/18 | 0.071 | 0.935 |
| GraphNorm + floor | **0/18** | 6/18 | **0.364** | 0.899 |

**Remaining "saturated" folds (6/18 by threshold) are all benign.** Unlike the BN collapses — where `pred_std_test` dropped to near zero — every remaining case with an extreme threshold still has `pred_std_test ≥ 0.364`. These are large-logit-but-correct predictions: the model is producing a bimodal distribution, but the positive-edge tail is pinned to exactly 1.0 (or the negative-edge tail to 0.0) because the logits are large in magnitude. The F1-optimal threshold lands at an extreme value as a consequence, but the underlying rank ordering is intact.

**AUC mean dropped from 0.935 → 0.899.** This is expected and honest. The drop is concentrated on fold 4 (AUC now 0.5–0.75, vs 0.667–0.917 under BN) and fold 6. Under BN, fold 4's eval-time saturation sometimes accidentally produced high AUC by numerical luck; GraphNorm removes that effect and exposes the true difficulty of the fold. Folds 1, 3, and 5 remain excellent (AUC 0.99–1.0 across all repeats).

**Fold 4 is genuinely the hardest graph.** Its AUC is now consistently in the 0.5–0.75 range regardless of repeat, with thresholds near 0 (all edges predicted positive). This is a real model limitation for this specific graph, not a normalization artifact. The cross-repeat sign flip observed under BN is gone — all three repeats now collapse in the same direction (low threshold) — which is consistent with the BN hypothesis: the sign flip was driven by random running-stat drift, not by anything graph-specific.

**Fold 5 best epochs are longer (95–242).** Under BN the model converged earlier; GraphNorm's per-graph normalization gives a different loss landscape that takes more epochs to traverse. This is fine given the `min_epoch=50` floor and the higher `max_epochs`.

**Conclusion.** The BN running-stat instability hypothesis is confirmed. Swapping to running-stat-free norms eliminates all genuine prediction collapses. Residual threshold extremes are benign large-logit behaviour. The remaining AUC gap on fold 4 is a genuine generalization problem for that graph, not a normalization artifact.

### Findings after label smoothing

Even with GraphNorm + floor, 6/18 folds had thresholds at the extremes of [0, 1] (criterion: <0.01 or >0.99). Two of these are benign: fold 5 R0 (threshold=0.9954, AUC=1.0) and fold 5 R2 (threshold=0.9997, AUC=0.993) — large-logit but correct, same category as the benign cases identified above. The remaining 4 — fold 4 R0/R1 (thresholds near 0, AUC 0.5–0.75) and fold 6 R0/R1 (thresholds near 0, AUC 0.81–0.86) — are genuine class-overlap failures. Two additional borderline cases sit just outside the strict criterion: fold 4 R2 (threshold=0.135, AUC=0.667) and fold 6 R2 (threshold=0.97, AUC=0.829). The cause of the 6 problematic/borderline cases is different from the BN collapse.

**Why healthy `pred_std` does not prevent extreme thresholds.** `pred_std` measures the *marginal* spread of all predictions taken together, regardless of class. What matters for threshold-setting is the *conditional* gap: are positive-edge predictions stochastically higher than negative-edge predictions? That is what AUC measures. A model can have a wide marginal spread (healthy `pred_std`) while assigning essentially the same probability range to both classes (low AUC). Fold 4 R1 (debug.csv) illustrates this precisely: `pred_std_test = 0.438` (healthy) but `AUC = 0.500` (random — no class signal at all).

**Why the F1-maximizing threshold then degenerates to an extreme.** When positive and negative predictions overlap completely, no selective threshold separates the classes cleanly, and the F1-optimal strategy collapses to a class-frequency decision. For fold 4 R1 (random predictions, balanced classes): working backwards from the recorded numbers — `threshold = 0.0009` (predict everything positive), `F1 = 0.667` — gives `2 × TP / (2 × TP + FP) = 2/3`, which means `TP = FP`, i.e., the test graph has equal numbers of true-positive and true-negative directed edges. With random predictions and balanced classes, no threshold can beat predict-everything-positive in F1, so the search collapses to `threshold ≈ 0`. For fold 6 R2, the mechanism is different: `AUC = 0.829` (decent discrimination) but `threshold = 0.97`, which is residual large-logit behaviour — a few positive edges are pushed to near-1.0 while the rest sit at moderate probabilities, so the optimal F1 boundary sits at the high end.

**Inferred mechanism for the class overlap.** The `debug.csv` for this run does not record training BCE or weight norms, so the following is a theoretical inference rather than a direct observation. Unconstrained BCE on hard labels drives predictions toward 0 and 1 (`BCE → 0` requires `logit → ±∞`), which in turn requires the trunk's weight matrices to grow in magnitude. Overfit weights that produce near-perfect discrimination on the training graph generalize poorly: on the held-out graph they produce varied predictions (healthy marginal std) but without the class-aligned structure needed for a clean decision boundary. The symptom — low AUC with healthy std — is directly observed; weight growth is the inferred cause.

**What label smoothing does.** Capping targets at `1 − ε` and `ε` (here `ε = 0.1`) creates a finite BCE floor of ≈ 0.325. The optimizer can no longer reduce the loss below this floor regardless of how large the logits grow, so there is no gradient incentive to keep growing weight magnitudes once predictions are comfortably inside the `[ε, 1−ε]` range. This acts as a mild regularizer that limits overfitting to the training graph and improves the conditional class separation on the held-out graph.

Two experiments were run:

1. `noDegLoss_minEpochFloor_swapNorm_labelSmooth` — GraphNorm + floor + label smooth 0.1, no degree loss.
2. `minEpochFloor_swapNorm_labelSmooth` — same plus degree penalty weight = 2.

| | Extreme thresh (`<0.01` or `>0.99`) | AUC mean | `pred_std_test` min |
|---|---|---|---|
| GraphNorm + floor | 6/18 | 0.899 | 0.364 |
| + label smooth 0.1 | **0/18** | **0.908** | 0.284 |
| + label smooth 0.1 + deg=2 | 0/18 | 0.896 | 0.342 |

**Label smoothing alone is the best configuration.** Thresholds are now in [0.075, 0.876] for all 18 cells. Under the tighter criterion, only fold 4 R0 (0.097), R1 (0.080), R2 (0.075) remain borderline — these are not BN-instability or logit-explosion artifacts but a genuine model failure on a structurally difficult graph (see below). AUC mean improves slightly from 0.899 → 0.908 because label smoothing also acts as a mild regularizer.

**pred_std_test is slightly lower** (median 0.344 vs 0.412 before). This is expected: targets capped at 0.9/0.1 constrain the model's output range, so predictions stay closer to 0.5. This is not a collapse; minimum is 0.284.

**Degree loss at weight = 2 hurts.** AUC drops from 0.908 → 0.896 and fold 4 R1 returns to AUC = 0.5. In the GraphNorm + label smoothing setting, BCE is already well-shaped and provides clear per-edge supervision; adding a competing degree penalty at weight 2 interferes rather than helps. **Degree loss is disabled going forward.**

**Fold 4 (5-node graph, two overlapping hypha chains) is a structural generalization failure.** All three repeats converge to threshold ≈ 0.08 and AUC 0.583–0.833 regardless of normalization strategy or label smoothing. Predictions ARE spread (pred_std_test 0.36–0.39) but positives and negatives land in overlapping probability ranges — the model has some rank-discrimination ability (AUC > 0.5) but cannot find a clean decision boundary. With two interleaved chains, local message passing sees topologically indistinguishable neighborhoods for both chains and cannot determine which edges cross chains vs. which are true connections without long-range structural context. This is a data-level limitation, not a model implementation problem.

### Confirmed best configuration

| Setting | Value |
|---|---|
| Normalization (MLP blocks) | `LayerNorm(out_channels)` |
| Normalization (residual skips) | `GraphNorm(hidden_channels + raw_dim)` |
| Early stopping floor | `min_epoch = 50` |
| Label smoothing | `ε = 0.1` |
| Degree loss | disabled (`weight = 0`) |

### Next direction

1. **Understand fold 4's failure more deeply.** It is a 5-node graph with two overlapping hypha cells. The model sees topologically indistinguishable neighborhoods for both chains. Visualize the TensorBoard prediction overlays for fold 4 to confirm whether the failure mode is random inter-chain edge mis-scoring or a systematic bias.
2. **Consider data augmentation or additional labelled graphs** of the two-chain type to give the model more training signal for this topology.
3. **Optional preflight:** log the held-out graph's post-z-score feature mean/std per fold to confirm the feature-distribution contribution is negligible.

> **Status:** all normalization and calibration fixes in place and confirmed. Genuine prediction collapses eliminated. Remaining difficulty on fold 4 is a data-level generalization problem, not a model implementation artifact.
