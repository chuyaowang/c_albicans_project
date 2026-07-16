# GCN Model Mermaid Diagram

The following diagrams visualize the overall data flow and the internal sub-structures of the `Model` defined in `simple_gnn.py`.

> **Scope — applies to both pipelines.** The architecture drawn here is **shared and live**: identical for the historical **nuclei** pipeline and the current **cell-fragment merge** pipeline. Only the input dimensions differ — `node_feature_dim=8`, `edge_feature_dim=10` for fragments (historically 6 / 6) — along with the RoI box source feeding the visual branch. Full breakdown: [Nuclei vs. cell-fragment](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/Cell%20Mask%20Graph%20Data%20Flow.md#Nuclei%20vs.%20cell-fragment%20—%20what%20carries%20over).

## 1. Overall Model Architecture

The diagram shows the full flow with the optional [Visual branch](C_Albicans%20Thesis%20Project/5.%20Results/4.%20GCN%20Design%20and%20Training/GCN%20Design%20Choices.md#Visual%20branch) enabled. When `use_visual_features=False`, the two `FusionMLP` nodes and everything feeding into them are skipped and `x` / `edge_attr` flow directly into the first GCN layer.

```mermaid

flowchart TD

%%{init: {"flowchart": {"defaultRenderer": "elk"}} }%%

%% Input Data

X[(Node Features: x)]

EI[(Edge Index: edge_index)]

EA[(Edge Attributes: edge_attr)]

  

X_Orig[("Original x")]

EA_Orig[("Original edge_attr")]

  

X --> X_Orig

EA --> EA_Orig

  

%% Visual branch (optional)

FM[("microsam_embedding<br>(256, H_f, W_f)")]

CN[("centroids (y, x)")]

PPF[("pixels_per_feature")]

  

subgraph Visual [Visual branch]

direction TB

NBox["_node_boxes<br>(150x150 around centroid)"]

EBox["_edge_boxes<br>(endpoints + margin)"]

RoI["roi_align<br>spatial_scale=1/pixels_per_feature"]

NCNN["NodeVisualCNN"]

ECNN["EdgeVisualCNN"]

NFuse["NodeFusionMLP"]

EFuse["EdgeFusionMLP"]

CN --> NBox

CN --> EBox

EI --> EBox

NBox --> RoI

EBox --> RoI

PPF --> RoI

FM --> RoI

RoI -- "node patches" --> NCNN

RoI -- "edge patches" --> ECNN

NCNN -- "node_visual" --> NFuse

ECNN -- "edge_visual" --> EFuse

end

  

X --> NFuse

EA --> EFuse

  

%% High-level Model Flow

NFuse -- "x_fused" --> Conv1[GCNConv Layer 1]

EI --> Conv1

EFuse -- "edge_fused" --> Conv1

  

Conv1 -- "x" --> ConcatX1[Concat: x, x_orig]

X_Orig -.-> ConcatX1

ConcatX1 --> NormX1[GraphNorm]

  

NormX1 -- "x" --> EU1[EdgeUpdater 1]

EI --> EU1

EA --> EU1

  

EU1 -- "edge_attr" --> ConcatE1[Concat: edge_attr, edge_attr_orig]

EA_Orig -.-> ConcatE1

ConcatE1 --> NormE1[GraphNorm]

  

NormX1 -- "x" --> Conv2[GCNConv Layer 2]

EI --> Conv2

NormE1 -- "edge_attr" --> Conv2

  

Conv2 -- "x" --> ConcatX2[Concat: x, x_orig]

X_Orig -.-> ConcatX2

ConcatX2 --> NormX2[GraphNorm]

  

NormX2 -- "x" --> EU2[EdgeUpdater 2]

EI --> EU2

NormE1 -- "edge_attr" --> EU2

  

EU2 -- "edge_attr" --> ConcatE2[Concat: edge_attr, edge_attr_orig]

EA_Orig -.-> ConcatE2

ConcatE2 --> NormE2[GraphNorm]

  

NormX2 -- "x" --> Classify[Classifier]

EI --> Classify

NormE2 -- "edge_attr" --> Classify

  

Classify -- "Edge Probabilities" --> Out([Predictions])

```

  

## 2. GCN Layer Architecture

  

```mermaid

flowchart TD

subgraph GCNConv_Internal [GCNConv Sub-structure]

direction TB

M_Concat["Concat: [x_j - x_i, edge_attr]"]

  

subgraph Message_Function [Message Function MLP]

direction LR

M_LL("CustomLazyLinear (bias=False)")

M_BN("LayerNorm")

M_Act("ReLU")

M_Drop("Dropout")

M_L("Linear (bias=True)")

M_LL --> M_BN --> M_Act --> M_Drop --> M_L

end

subgraph Attention_Function [Attention Mechanism]

direction LR

A_LL("attn_mlp: CustomLazyLinear (out=1)")

A_Softmax("softmax(..., index)")

A_LL --> A_Softmax

end

M_Concat --> Message_Function

M_Concat --> Attention_Function

  

Message_Function -- "msg" --> MsgScale{"Multiply<br>(msg * alpha)"}

Attention_Function -- "alpha" --> MsgScale

  

MsgScale --> Aggr{"Message Passing<br>(aggr='sum')"}

subgraph Update_Function [Update Function MLP]

direction LR

U_Concat["Concat: [x, aggregated_messages]"]

U_MLP["MLP (Linear -> LN -> ReLU -> ...)"]

U_Concat --> U_MLP

end

  

Aggr -- "aggregated_messages" --> U_Concat

end

```

  

## 3. Classifier Architecture

  

```mermaid

flowchart LR

subgraph Classifier_Internal [Classifier Sub-structure MLP]

direction LR

C_Extract["Extract Source (x[edge_index[0]]) & Target (x[edge_index[1]])"]

C_Concat["Concat: [source_feature, target_feature, edge_attr]"]

C_LL("CustomLazyLinear (bias=False)")

C_BN("LayerNorm")

C_Act("ReLU")

C_Drop("Dropout")

C_L("Linear (out_features=1, bias=True)")

C_Squeeze["Squeeze(-1) & Sigmoid()"]

C_Extract --> C_Concat --> C_LL --> C_BN --> C_Act --> C_Drop --> C_L --> C_Squeeze

end

```

  

## 4. EdgeUpdater Architecture

  

```mermaid

flowchart LR

subgraph EdgeUpdater_Internal [EdgeUpdater Sub-structure MLP]

direction LR

E_Extract["Extract Source (x[edge_index[0]]) & Target (x[edge_index[1]])"]

E_Concat["Concat: [source_feature, target_feature, edge_attr]"]

E_LL("CustomLazyLinear (bias=False)")

E_BN("LayerNorm")

E_Act("ReLU")

E_Drop("Dropout")

E_L("Linear (bias=True)")

E_Extract --> E_Concat --> E_LL --> E_BN --> E_Act --> E_Drop --> E_L

end

```