# GCN Model Experiments

> Experiments done to optimize the model and training protocol. Documents what worked, what did not, and the reasoning behind each outcome. Design and training choices documented in [GCN Design Choices](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Design%20Choices.md) and [GCN Training Choices](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Training%20Choices.md) trace back to the sections below.

> **Scope — historical.** Every experiment on this page was run on the **nuclei** pipeline (node = one DAPI nucleus, fully-connected candidate edges, manual labels, 6 node / 6 edge features), which is no longer run. The *conclusions* carry over to the current **cell-fragment merge** pipeline — the model and trainer are identical (see [Nuclei vs. cell-fragment](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Cell%20Mask%20Graph%20Data%20Flow.md#Nuclei%20vs.%20cell-fragment%20—%20what%20carries%20over)) — but the **reported numbers, graph sizes and feature schema are nuclei-era and have not been re-measured on fragment data**. Treat the metrics as historical evidence for the design decisions, not as expected fragment performance.

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
- **Decision — disabled.** ⚠️ Superseded. v3 was initially kept in the composite objective, but the degree loss is now **off**: `degree_penalty_weight` defaults to `0.0` in `train_model`, `n_fold_validation` and `train_overfit_test`, and the penalty is only computed when the weight is `> 0`. Two independent reasons, neither of which depends on the node type (nuclei or cell-mask fragment):
  - **The cheat was never eliminated.** Predicting all-near-zero probabilities minimises the second term; the subtraction coupling in v3 only *partially* mitigates it. This is a property of the formulation.
  - **It measurably hurt.** At weight 2, AUC dropped 0.908 → 0.896 and fold 4 R1 returned to 0.5 — with GraphNorm + label smoothing the BCE is already well-shaped and a competing degree penalty interferes. See [Findings after label smoothing](#Findings%20after%20label%20smoothing).

  Do not reintroduce it. Structural constraints, if ever needed, belong in a decode step over the predicted probabilities, not in the loss.

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

- **Issue observed:** Model occasionally predicts cycles, which is biologically impossible.
- **Superseded approach (nuclei-era):** Rely on BCE + degree loss to discourage cycles implicitly. Residual cycles are rare and easy to spot, and were pruned manually in the downstream analysis.
  - **Caveat — the degree loss could not forbid a ring anyway.** The degree penalty is a *local* constraint while acyclicity is a *global* property. A hexagon gives every node degree exactly 2, perfectly satisfying the penalty, so there is no gradient against it. The fallback was always weaker than "BCE + degree loss" suggests: it discourages the degree violations that usually *accompany* a cycle, not the cycle itself.
- **Current approach — BCE + visual features; the degree loss is off.** `degree_penalty_weight` now defaults to `0.0` in `train_model`, `n_fold_validation` and `train_overfit_test`, and the penalty is only computed when the weight is `> 0` — so the degree term contributes nothing unless explicitly switched on. It was disabled on evidence, not by neglect: at weight 2 it *hurt* (AUC 0.908 → 0.896, fold 4 R1 back to 0.5), because with GraphNorm + label smoothing the BCE is already well-shaped and a competing degree penalty interferes — see [Findings after label smoothing](#Findings%20after%20label%20smoothing). What discourages cycles today is the [visual features](#9.%20Visual%20features): they give each edge much stronger *independent* evidence of being true or false, and cycles arise precisely when the model must guess between comparably-scored candidates. Explicit DAG machinery was judged unnecessary once the visual branch was in.
- **⚠️ Cycles still matter for cell fragments — topology is part of the answer.** Fragments are pieces of one hyphal cell, and their **connection order encodes the direction of growth**. This is exactly why the training labels are a per-cell **minimum spanning tree over *adjacent* fragments rather than a clique** ([Training labels](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Cell%20Mask%20Graph%20Data%20Flow.md#Training%20labels%20(per-cell%20MST))) — a chain, not a blob. A cycle corrupts that chain just as it did for nuclei, so acyclicity is **not** obsolete in the fragment pipeline; it is inherited.
  - Connected components would still recover the correct *set* of fragments to merge even in the presence of a cycle, so a cycle is harmless **if** only the partition is wanted. But the chain order is also wanted, so that leniency does not apply.
- **Cycles are now measured, not just feared.** `cell_merge_inference.merge_fragments` ([Inference merge](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Cell%20Mask%20Graph%20Data%20Flow.md#Inference%20merge)) builds each cell's subnetwork as an `nx.Graph` and classifies its topology as `singleton` / `path` / `branched` / `cyclic`. Only the first two are biologically well-formed, and the tally is logged to `Merge/Graph_<id>_summary` at the end of every run. This turns "cycles are biologically impossible" from an assumption into a **per-run count** — the first concrete evidence of how often the model actually violates the constraint, which is what any decision about fixing it should rest on.
  - First observation, from a deliberately under-trained 12-epoch overfit run on the 157-fragment graph: `54 cells: 27 singleton, 17 path, 6 branched, 4 cyclic`. So ~19% of merged cells had a forbidden topology at that (poor) operating point. Not a trained-model number — recorded only as a baseline for comparison.
- **Still open — cycles are reported, not prevented or repaired.** The merge groups correctly regardless (a cycle is harmless to the fragment *set*), but a cyclic cell yields **no** chain order, and a branched cell falls back to its diameter path. Whether to add a structural decode (maximum spanning tree per component, or a degree-capped path cover — both survive the symmetric read-out, unlike the [directional mask](#7.%20Acyclicity) that failed) or a repair pass is undecided. The topology tally now supplies the evidence for that call.
- **Known alternative not attempted:** A NOTEARS-style acyclicity loss. This requires an augmented Lagrangian training loop, which is likely too complex and data-hungry for our 5-graph dataset. Not pursued.

### What did not work

> Long-form account, with the model code and the full argument: [Topological DAG Constraint (Abandoned)](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Topological%20DAG%20Constraint%20(Abandoned).md). The implementations survive under `dapi_tracing/deprecated/` (`sinkhorn_gnn.py`, `train_sinkhorn.py`, `sinkhorn_architecture.md`). **The naming is a fossil:** no Sinkhorn operator was ever implemented — it was rejected at the design stage — and what those files actually contain is the topological-potential model (`AcyclicModel`).

- **Sinkhorn permutation matrix** to rank nodes: requires a fixed `[N_nodes, N_ranks]` matrix, incompatible with our variable-size graphs (3–6 nodes here, and far more variable for cell fragments). Rejected at the design stage; never implemented.
- **Scalar "topological potential" per node** (connection flows only from high to low potential, e.g., `[1, 0.8, 0.7, …, 0]`): sidesteps the fixed-size problem — a per-node scalar works at any graph size — but **conflicts with the symmetric edge-prediction setting** ([§6](#6.%20Symmetric%20predictions:%20average%20vs.%20max)) that defines our undirected task. Because symmetrization means *one direction true ⇒ both true*, and any two potentials always leave exactly one direction running downhill, the mask can never suppress a pair — it keeps the downhill edge, and symmetrization hands that verdict straight back to the uphill reverse. Both symmetrizations fail, in opposite ways:
  - **Max:** `max(≈0, ≈1) = ≈1`. The kept direction restores the suppressed one; final classification reduces to the predicted probability alone and the potential does nothing.
  - **Average:** `(0 + 1) / 2 = 0.5` for *every* true edge, regardless of what the classifier actually believes. No better than randomly guessing the edge class.

  Max was used for this experiment (the baseline uses average, §6) precisely because average is unusable here — but max is exactly what makes the mask vacuous. No setting of `temperature` escapes the pincer: it follows from combining a hard directional mask with a symmetric read-out, not from tuning.
- **General takeaway:** forcing an undirected acyclic structure purely through directed topological potentials is not viable in this setup. Abandoned, and the ranking branch was removed from the live model. If cycles ever become a practical problem again, the lesson is that any fix must **survive symmetrization** — a post-hoc decoding step over the undirected probabilities (maximum spanning forest, or a degree-capped path cover) is compatible with a symmetric read-out in a way that a directional mask is not.

## 8. Class imbalance

- **Issue observed:** Model predicts almost everything as negative on larger graphs.
- **Root cause:** Positive-to-negative edge ratio varies dramatically with graph size (fully connected candidate graphs):
  - 3-node graph: 2 true / 3 total (67% positive)
  - 4-node graph: 3 true / 6 total (50% positive)
  - 6-node graph: 5 true / 15 total (33% positive)
- **Consequence for CV:** The positive ratio in the training fold does not match the test fold, so any statistic derived from the training distribution (e.g., class weights) is miscalibrated at test time. The model's *implied prior over the classes* becomes a function of which graphs the fold split happened to deal.
- **Working fix — negative edge sampling:** For each graph in each batch, sample negative edges so the positive:negative ratio is fixed. Loss is computed only on sampled positives + sampled negatives.
- **Working fix — AUC-based early stopping:** Accuracy rewards "predict everything negative" under class imbalance. AUC does not, so early stopping tracks validation AUC.
- **Working fix — F1-maximizing threshold:** The 0/1 decision threshold is chosen per fold to maximize validation F1 rather than fixed at 0.5.

### What did not work

#### Weighted BCE — tried first, before negative sampling

> ⚠️ **Do not revisit.** It does not remove the train/test distribution mismatch — it inverts the direction of the mismatch. Recorded here so the reasoning is not re-derived from a vague memory.

**Definition.** Standard positive-class reweighting, with the weight computed from the true/false edge ratio of the **training fold**:

$$L = -\frac{1}{N}\sum_i \left[\; w^{+} \cdot y_i \log p_i \;+\; (1 - y_i)\log(1 - p_i) \;\right], \qquad w^{+} = \frac{N_{\text{neg}}}{N_{\text{pos}}}$$

where $y_i \in \{0, 1\}$ is the label of candidate edge $i$, $p_i$ the predicted probability, and $N_{\text{pos}}$ / $N_{\text{neg}}$ are counted over the **training fold's** candidate edges. $w^{+}$ is computed once and held fixed for the run.

**Intent.** With large graphs dominating the training fold, false edges vastly outnumber true ones, so the model learns that *false is the common class* and its predictions collapse toward small values. Setting $w^{+} > 1$ was meant to counteract this by making a missed true edge (false negative) cost more than a spurious one.

**Why it failed.** $w^{+}$ is itself a training-fold statistic, so it re-introduces the very mismatch it was meant to fix — only with the sign flipped. Using the ratios above, $w^{+}$ is fully determined by the fold's graph mixture:

| Training fold | $N_{\text{pos}} : N_{\text{neg}}$ | $w^{+}$ | What the model infers |
| --- | --- | --- | --- |
| all 3-node | 2 : 1 | **0.5** | false negatives matter *less* |
| all 4-node | 3 : 3 | **1.0** | unweighted |
| all 6-node | 5 : 10 | **2.0** | false negatives matter 2× more |

Train on 6-node graphs ($w^{+} = 2$, false negatives penalized heavily) and test on a 3-node graph (67% positive — already balanced-to-positive-heavy), and the model now carries the assumption that **true edges are the important class**. The predictions shift *high* rather than low. The bias changes direction; it does not disappear. Both the unweighted and the weighted objective make the implied prior a function of the training fold's graph mixture, and the test fold's mixture differs by construction — so results were inconsistent across folds either way.

**Why negative sampling is different in kind.** It does not try to *correct* a skewed ratio with a coefficient; it **fixes the ratio to a constant** for every graph in every batch. No fold-dependent prior is left to transfer. That is the distinction: $w^{+}$ relocated the problem, sampling removed it.

📄 **No implementation survives.** Every loss construction in the project is a bare `torch.nn.BCELoss()` (`gnn_train.py:692`, `:802`). The weighted variant was removed without leaving code, and this directory is not under version control, so the definition above is reconstructed from the prose records in this section and in `dapi_tracing/CLAUDE.md` §5.3 — not read off an implementation. The mechanism is also unrecoverable: the model emits probabilities via `torch.sigmoid` (`simple_gnn.py:129`) into `BCELoss`, which has no `pos_weight` argument, so this was either a per-element `BCELoss(weight=…)` mask tensor or a swap to `BCEWithLogitsLoss` with the sigmoid removed.

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

---

## 11. Node type classification

> **Verdict up front.** The node head **improves edge ranking on every fold** (AUC 0.8445 → 0.8807, better on 6/6) while **failing at its own task** (node accuracy 0.7503, below the per-image majority baseline on 4/6 folds) and **not fixing the failure that motivated it** (background F1 0.41). It earns its place as an **auxiliary task that regularises the shared representation**, not as a working node classifier. The gain is in *ranking*, not in *decisions*: PR-AUC and edge accuracy are flat.

### Why it was tried

Two failure modes in the merge predictions, both statements about nodes rather than edges:

1. **AIS calls background regions cells.** Those fragments become nodes, get candidate edges, and get merged into cells that do not exist.
2. **Candidate edges span epithelial and hyphal masks.** Those can never be a true merge — they are different cells, and different kinds of cell.

Both imply a constraint on edges: **a true edge cannot span background↔cell or epithelial↔hyphal**. The graph already contains the counterexamples — 1134 cell↔background and 1078 epithelial↔hyphal directed negatives (see [Node Type Label Construction](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Node%20Type%20Label%20Construction.md#5.%20Edge%20labels%20from%20the%20same%20split)) — so the evidence to learn from was already present and unused.

**The hypothesis:** predicting node type from the node embeddings, with a combined loss over a shared trunk, forces the representation to encode *what a node is* in order to score its edges — so the constraints are learned **implicitly**. Nothing is fed in explicitly; the type is what is being predicted, so it cannot be an input. Design: [Node Classifier Head](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Design%20Choices.md#Node%20Classifier%20Head%20(optional)). Labels: [Node Type Label Construction](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Node%20Type%20Label%20Construction.md).

### Protocol — a matched pair

The two runs differ **only** by the node head. Same `k=10`, same `min_overlap_frac=0.1`, same visual branch, same hyperparameters, same folds, same seed.

| | Baseline (edge-only) | Node head |
| --- | --- | --- |
| Notebook | `10_Merge Oversegmentation GNN.ipynb` | `12_Node Type GNN.ipynb` |
| `predict_node_type` | `False` | **`True`** |
| `node_loss_weight` | — | **1.0** (untuned — never swept) |
| `node_sample_ratio` | — | 1.0 (equal counts per present class) |
| `use_visual_features` | `True` | `True` |
| Run | `cv_experiment/merge/merge_cv_k10_minFrac0_1/` | `cv_experiment/nodetype/nodetype_cv_k10_minFrac0_1_visual/` |

Notebook 10 was **re-run** at `k=10` / `min_frac=0.1` for this comparison — its earlier `merge_cv` results used different values and are not a valid baseline.

**Fold → test image.** Folds are `KFold(shuffle=True, random_state=42)`, so fold *k* is **not** image *k*. Recovered from each fold's node-type support counts against the per-image label distribution — the match is unique:

| fold | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| **test image** | 0 | 1 | **5** | 2 | 4 | 3 |

*Source: `NodeType/Support_<class>_Test` in each `fold_*/events*`, matched against notebook 11 cell 15.*

### Overfit sanity check — the head works

Both configurations overfit a single graph (image 3) to near-perfection:

| | Edge AUC/Eval | Node accuracy | best epoch |
| --- | --- | --- | --- |
| Baseline | 1.0000 | — | 316 |
| Node head | 0.9999 | **0.9952** (F1 bg 0.976 / epi 1.000 / hyph 0.997) | 106 |

*Source: `overfit_experiment/{merge/merge_overfit_one_graph_k10_minFrac0_1, nodetype/nodetype_overfit_one_graph_k10_minFrac0_1_visual}/**/events*`.*

**This is the most important number in the section.** The head reaches **0.9952** node accuracy on a graph it trained on, and **0.7503** on graphs it did not. The task is learnable and the capacity is there; what fails is **generalisation across images**.

> The node-head run also reached its best epoch sooner here (106 vs 316), but **this does not replicate** — across the six CV folds mean `best_epoch` is 93.3 (baseline) vs 92.5 (node head), i.e. identical. The overfit difference is one run against one run and should not be read as faster convergence.

### Cross-validation results

Leave-one-out over 6 images, **one repeat**.

**Edge AUC — improves on every fold:**

| fold (img) | baseline | node head | Δ |
| --- | --- | --- | --- |
| 1 (img 0) | 0.8561 | 0.8793 | **+0.0232** |
| 2 (img 1) | 0.8328 | 0.8612 | **+0.0283** |
| 3 (img 5) | 0.8646 | 0.8995 | **+0.0349** |
| 4 (img 2) | 0.9006 | 0.9428 | **+0.0422** |
| 5 (img 4) | 0.7966 | 0.8437 | **+0.0471** |
| 6 (img 3) | 0.8162 | 0.8576 | **+0.0414** |
| **mean** | **0.8445** | **0.8807** | **+0.0362 — better on 6/6** |

**Every other headline metric — flat:**

| metric | baseline | node head | Δ | folds improved |
| --- | --- | --- | --- | --- |
| AUC | 0.8445 | 0.8807 | **+0.0362** | **6/6** |
| F1 | 0.4862 | 0.5160 | +0.0298 | 6/6 — *but see below* |
| PR-AUC | 0.4030 | 0.4070 | **+0.0040** | **3/6** |
| Edge accuracy | 0.8792 | 0.8767 | **−0.0025** | **3/6** |

*Source: `aggregate/cv_summary.csv` of both runs for AUC / F1 / PR-AUC; `Accuracy/Test` at each fold's `EarlyStopping/Best_Epoch` for accuracy.*

#### Reading this honestly

**The gain is in ranking, not in decisions.** AUC asks "are positives ranked above negatives?" and improves uniformly. PR-AUC asks the same question in a way that is sensitive to the 10% positive rate — and does **not move** (+0.004, worse on folds 1 and 4 by −0.060 and −0.056). Edge accuracy at the chosen threshold is likewise flat. A better ranking that does not produce better decisions is a real but limited result.

**Do not read F1's 6/6 as confirmation.** The decision threshold is chosen per fold to **maximise F1 on the best-AUC epoch** ([Decision threshold](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Training%20Choices.md#Decision%20threshold)), and under leave-one-out the validation graph **is** the held-out graph. So F1 is measured at a threshold fitted on the very graph it scores — it reports the *best achievable* operating point, not an achievable one. PR-AUC is the threshold-free version of the same precision/recall question, and it is flat. **The gap between F1 (+0.030, 6/6) and PR-AUC (+0.004, 3/6) is the fitted threshold, not the model.** Two folds' F1 gains are noise anyway (+0.0007, +0.0001).

**6/6 on AUC is still meaningful.** If the head had no effect and each fold were a coin flip, 6/6 in one direction has probability `2⁻⁶ ≈ 0.016`. But the folds share five-sixths of their training data, so they are **not independent** and that figure is optimistic. With **one repeat**, seed variance is unmeasured and cannot be separated from the effect.

### Node-type results — the head does not do its own job

| fold (img) | accuracy | majority baseline | Δ | F1 bg | F1 epi | F1 hyph | support (bg/epi/hyph) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (img 0) | 0.9149 | **0.9291** | −0.014 | 0.6667 | — | 0.9572 | 10 / 0 / 131 |
| 2 (img 1) | 0.9108 | **0.9745** | −0.064 | 0.3077 | — | 0.9527 | 4 / 0 / 153 |
| 3 (img 5) | **0.5846** | 0.3538 | **+0.231** | 0.3333 | 0.5641 | 0.7213 | 21 / 21 / 23 |
| 4 (img 2) | **0.6949** | 0.4068 | **+0.288** | 0.6111 | 0.8511 | 0.5714 | 21 / 24 / 14 |
| 5 (img 4) | 0.6620 | **0.7254** | −0.063 | 0.1143 | 0.5556 | 0.7897 | 8 / 31 / 103 |
| 6 (img 3) | 0.7343 | **0.7826** | −0.048 | 0.4444 | 0.5135 | 0.8339 | 20 / 25 / 162 |
| **mean** | **0.7503** | | | **0.4129** (6/6) | **0.6211** (4/6) | **0.8044** (6/6) | |

*Source: `NodeType/*_Test` at `best_epoch` in each `fold_*/events*`; majority baseline computed from notebook 11 cell 15.*

**Accuracy is the wrong lens, and it is worth saying why.** The majority baseline — always predict the image's commonest class — beats the head on 4/6 folds. But that baseline scores 0.9291 on image 0 while having **F1 = 0 for background**: it never finds a single background fragment, which is precisely the failure this head was built to fix. The head trades majority-class accuracy for the ability to find minority classes at all. **That is the right trade; the problem is how little it buys.**

- **Background F1 = 0.41.** The originally-described failure — AIS calling background cells — is **not fixed**. Per-fold it ranges 0.11–0.67 with supports of only 4–21 nodes.
- **The head only beats majority where the classes are balanced** — images 5 (21/21/23) and 2 (21/24/14). On the hyphal-dominated images it does not.
- **Epithelial is reported on 4/6 folds only.** Images 0 and 1 have no epithelial nodes, so those folds have no epithelial support. Absence of the metric is a property of the fold.

### What did not work

- **Fixing the background failure.** F1 0.41 at 10.9% prevalence. The head flags *some* background but not reliably enough to act on.
- **Improving decisions.** PR-AUC +0.004 and edge accuracy −0.003. Whatever the head adds to the representation shows up in ordering, not in the operating point.
- **Generalising the node task.** 0.9952 overfitting one graph versus 0.7503 across graphs is the whole story: **not capacity, not the loss, not the labels — data.**

### Caveats

1. **n = 6 images, one repeat.** No seed-variance estimate; the folds are not independent.
2. **`node_loss_weight = 1.0` is untuned.** Set as the neutral default, never swept. The result is the gain at an arbitrary weight.
3. **The node labels use a per-image threshold.** The epithelial/hyphal boundary is set per image because magnification differs (`mean_width` medians 17.5 px to 179.9 px across images). The head therefore learns from a target whose definition is not globally consistent — deliberate, but it caps what "generalise" can mean here.
4. **`mean_width` is scale-dependent**, so the labels themselves would not transfer to a new magnification without a new threshold.

### Where this leaves the approach

The head is **kept**: it costs little and improves ranking on every fold. But it did not deliver the mechanism it was built for, and the overfit-vs-CV gap says why — with six images of three distinct modalities, there is not enough evidence to learn what a cell type *is* in general. See [Future Directions](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Future%20Directions.md).

> **Status:** complete and not under active development. The result is real but narrow; the binding constraint is the dataset, not the architecture.
