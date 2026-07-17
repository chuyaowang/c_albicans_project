# Approach History

Why the pipeline looks the way it does. Three generations of approach, each motivated by the previous one's failure:

**deterministic greedy tracing → nuclei-node GCN → cell-fragment-node GCN.**

```mermaid
flowchart LR
    A["<b>0. Deterministic</b><br/>greedy tracing<br/><i>hand-tuned scores</i>"]
    B["<b>1. Nuclei-node GCN</b><br/>node = one nucleus<br/><i>learned scores</i>"]
    C["<b>2. Fragment-node GCN</b><br/>node = one AIS cell mask<br/><i>merge oversegmentation</i>"]
    D["<i>side branch:</i><br/><b>Topological DAG</b><br/>constraint"]
    E["<b>2b. + node-type head</b><br/><i>learn what a node is,<br/>to score its edges</i>"]

    A -->|"not differentiable;<br/>fails on >1 hypha"| B
    B -->|"nuclei ≠ cells;<br/>DAPI-only, no cell outline"| C
    B -.->|"cycles are<br/>biologically impossible"| D
    D -.->|"abandoned:<br/>mask bypassed by max<br/>+ visual features made<br/>it unnecessary"| B
    C -->|"background scored as cells;<br/>edges span cell types"| E
    E -.->|"kept: ranking improved.<br/>but data-limited, not<br/>architecture-limited"| C
```

| Stage | Node | Segmentation needed | Status | Doc |
| --- | --- | --- | --- | --- |
| 0 | *(none — pixels + nuclei)* | Otsu / RF nuclei | **Deprecated** | [Deterministic Hyphal Tracing](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Deterministic%20Hyphal%20Tracing%20(Deprecated).md) |
| 1 | one DAPI nucleus | Otsu + watershed / RF | **Historical** | [GCN Data Flow](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Data%20Flow.md) |
| 1b | *(side branch)* | — | **Abandoned** | [Topological DAG Constraint](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Topological%20DAG%20Constraint%20(Abandoned).md) |
| 2 | one AIS cell-mask fragment | fine-tuned micro-SAM AIS | **Live** | [Cell Mask Graph Data Flow](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Cell%20Mask%20Graph%20Data%20Flow.md) |
| 2b | *(extension of 2)* — same node, **plus a node-type head** | same | **Live**, off by default | [Node Type Label Construction](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Node%20Type%20Label%20Construction.md) · [§11](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#11.%20Node%20type%20classification) |

---

## Stage 0 → 1: why learn the scoring function?

The deterministic tracer scored every nucleus pair with a hand-written formula (path intensity × orientation alignment × a distance prior), then selected edges greedily under degree and acyclicity constraints. It worked on single, well-separated hyphae and **failed as soon as an image contained more than one hypha**.

The failure was structural, not a tuning problem:

- **Nothing was learned.** Every constant — the "expected" 5-nucleus-length spacing, the 90° angle cut-off, the 0.85 linearity floor — was hand-set and image-specific.
- **Decisions could not be revised.** Greedy acceptance plus union-find is irrevocable: one wrong high-scoring edge permanently consumes a node's degree budget and blocks the correct one.
- **Pairs were scored in isolation.** The score of `(i, j)` knew nothing about `(i, k)` competing for the same nucleus. Structure was imposed *afterwards* as hard constraints rather than reasoned about jointly.

The GCN answers each of these directly: the scoring function is **learned** from the same evidence, attention makes candidate edges **compete** during message passing, and the degree penalty turns the hard degree cap into a **soft, differentiable** objective. The lineage is literal — the tracer's three ingredients (path intensity, normalized distance, orientation alignment) became the nuclei pipeline's edge features almost one-for-one.

## Stage 1: why nuclei first?

**Because nuclei were free to segment.** DAPI nuclei are small, round, high-contrast and well-separated — `threshold_otsu` plus an optional watershed (or a small random forest) segments them adequately with no deep model at all, and no fine-tuning. That made it possible to build and validate the *entire* graph pipeline — feature extraction, PyG data flow, the GCN, cross-validation, the visual branch, interpretation — while the segmentation problem stayed a solved, cheap prerequisite.

The trade-off is that **a nucleus is not a cell**. Nuclei are proxies: a hypha is inferred from the *chain* of nuclei running through it, and the connection evidence is whatever DAPI signal happens to lie between two nuclei. The cell body itself is never segmented, so the model reasons about cells it cannot see.

## Stage 1 → 2: why move to cell-mask fragments?

Fine-tuning micro-SAM's AIS decoder made whole-cell masks available from the DIC channel (see the micro-SAM fine-tuning reference), which changes the question from "which nuclei belong to one cell?" to "which mask fragments belong to one cell?" — the cell body is now directly observed rather than inferred.

That move brought its own problem, which is the reason the current pipeline exists: **AIS oversegments long hyphae**, splitting one cell into several fragments (partly a tile-seam artifact in the reassembled decoder distance maps, partly genuine hyphal length). Rather than fight the segmentation, the graph network was re-pointed at the fragments: node = fragment, edge = "same biological cell?", inference = group the predicted-positive edges into cells and read back each cell's fragment **chain order** (which encodes the direction of growth) — see [Inference merge](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Cell%20Mask%20Graph%20Data%20Flow.md#Inference%20merge).

Crucially this was an **extension, not a rewrite**. The model, trainer, loss, cross-validation and visual branch carried over unchanged; only graph construction and the RoI box source are new. See [Nuclei vs. cell-fragment — what carries over](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Cell%20Mask%20Graph%20Data%20Flow.md#Nuclei%20vs.%20cell-fragment%20—%20what%20carries%20over).

## Stage 2b: why give the model a second head?

Stage 2 works, but two of its mistakes are not really about edges. **AIS calls some background regions cells**, and those phantom nodes get merged into cells that do not exist; and **candidate edges get drawn between epithelial and hyphal masks**, which can never be one cell. Both are statements about *nodes*, and both imply the same constraint: a true edge cannot span background↔cell or epithelial↔hyphal.

The tempting fix is to filter — classify the fragments, then delete the impossible edges. That is a hand-written rule, and it needs the node type *at inference*, which is the very thing you do not have. So instead the model got a **second head** predicting node type from the same embeddings, trained with a combined loss over a shared trunk. Nothing is fed in explicitly: because one representation must serve both tasks, it has to encode what a node *is* in order to score its edges, and the constraint is learned implicitly.

**It half-worked, and the half that failed is the informative one.** Edge ranking improved on every fold (AUC 0.8445 → 0.8807), so the shared representation genuinely got better. But the head does not do its own job — background F1 is **0.41**, the failure that motivated it is not fixed, and PR-AUC did not move. The diagnosis is in the gap between two numbers: **0.9952** node accuracy overfitting one graph, **0.7503** across six. Not capacity, not the loss, not the labels — **data**. See [§11](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#11.%20Node%20type%20classification).

## The side branch: enforcing a DAG

Along the way, stage 1 exposed a real weakness: **edge predictions are independent of one another, so nothing stops the model from predicting a cycle** — biologically impossible for a hyphal chain. A branch of work tried to make acyclicity *structural* by having the network learn a node ranking. It was abandoned: the mechanism was mathematically self-defeating, and the arrival of visual features made the problem largely moot by giving each edge much stronger independent evidence. The decision was to let the model learn acyclicity **implicitly** from the (always-acyclic) labels. See [Topological DAG Constraint](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Topological%20DAG%20Constraint%20(Abandoned).md).

---

## The through-line

Each generation kept the previous one's *evidence* and replaced its *decision rule*:

| | Evidence | Decision rule |
| --- | --- | --- |
| **Deterministic** | path intensity, distance, orientation | hand-tuned formula + greedy constrained selection |
| **Nuclei GCN** | the same three, as learned edge features (+ SAM visual features) | learned message passing; attention as soft competition; degree penalty as soft constraint |
| **Fragment GCN** | the same, re-anchored on mask boundaries, + contact/complementarity/continuity | unchanged from the nuclei GCN |
| **+ node-type head** | unchanged — **no new evidence at all**; the same inputs, asked a second question | unchanged; the constraint on which edges may exist is left implicit in a shared representation |

The direction of travel is consistent: **push structure out of hand-written rules and into learned representations**, and give the model better evidence rather than stronger constraints. The abandoned DAG branch is the one time that direction was reversed — and it is the one that failed.

The node-type head is the purest expression of that direction — it adds **no evidence whatsoever**, only a second question asked of the same inputs, and lets the constraint fall out of the representation. That it helped at all (better ranking on 6/6 folds) is a point in the direction's favour. That it helped so little is the point at which the direction runs out of road.

## Where this stops

**Every remaining failure is now a data problem, not a design problem.** The evidence converged on this from three directions:

- the node head: **0.9952** overfitting one graph vs **0.7503** across six ([§11](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#11.%20Node%20type%20classification))
- five-graph CV, back at stage 1: fits all five together, predicts implausible structure held out ([§1](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#Five-graph%20cross%20validation))
- fold 4's collapse: "a data-level limitation, not a model implementation problem" ([§10](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Model%20Experiments.md#10.%20Saturated%20probabilities%20under%20leave-one-out%20CV%20(in%20progress)))

Six images, no two of them quite the same kind of picture, is not enough to learn what a cell *is* in general — and no fourth generation of architecture changes that. The next move is not a better model; it is **more and better-labelled data**, and a segmenter good enough that the failures the GCN cannot express (under-segmentation above all) stop arriving in the first place.

**The project was wrapped here, on 2026-07-17, with the pipeline working end to end.** What would restart it — and what the evidence already rules in and out — is in [Future Directions](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Future%20Directions.md).
