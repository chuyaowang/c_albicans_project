import torch
from torch.nn import Sequential, ReLU, Dropout, Linear
from torch.nn.modules.batchnorm import LazyBatchNorm1d

# Import existing components from your baseline GNN
from image_processing_tools.dapi_tracing.simple_gnn import (
    GCNConv, EdgeUpdater, Classifier, CustomLazyLinear
)


class AcyclicModel(torch.nn.Module):
    """
    A Graph Neural Network that incorporates a Topological Potential ranking branch
    to enforce acyclic, directed predictions on inherently undirected graphs.
    """
    def __init__(self, hidden_channels, dropout_p=0.2, temperature=1.0):
        """
        Initializes the AcyclicModel.
        
        Args:
            hidden_channels (int): Dimension of the hidden node and edge embeddings.
            dropout_p (float): Dropout probability for regularization. Defaults to 0.2.
            temperature (float): Temperature for the directional mask sigmoid. Defaults to 1.0.
        """
        super().__init__()
        # --- Standard GNN Layers ---
        self.conv1 = GCNConv(hidden_channels, dropout_p=0)
        self.edge_updater = EdgeUpdater(hidden_channels, dropout_p)
        self.conv2 = GCNConv(hidden_channels, dropout_p)
        self.edge_updater_1 = EdgeUpdater(hidden_channels, dropout_p)
        self.classifier = Classifier(hidden_channels, dropout_p)
        
        self.norm_x1 = LazyBatchNorm1d()
        self.norm_e1 = LazyBatchNorm1d()
        self.norm_x2 = LazyBatchNorm1d()
        self.norm_e2 = LazyBatchNorm1d()

        # --- Topological Potential Ranking Branch ---
        self.rank_mlp = Sequential(
            CustomLazyLinear(hidden_channels, bias=False),
            LazyBatchNorm1d(),
            ReLU(),
            Dropout(p=dropout_p),
            Linear(hidden_channels, 1, bias=False) # Predicts a single continuous scalar per node
        )
        
        self.temperature = temperature
        
    def forward(self, x, edge_index, edge_attr, batch=None, return_attention=False):
        """
        Forward pass of the Acyclic model.
        
        Args:
            x (torch.Tensor): Node feature matrix of shape [Total_Nodes, Node_Features].
            edge_index (torch.Tensor): Graph connectivity matrix of shape [2, Total_Edges].
            edge_attr (torch.Tensor): Edge feature matrix of shape [Total_Edges, Edge_Features].
            batch (torch.Tensor, optional): Batch vector assigning each node to its graph. Defaults to None.
            return_attention (bool): Whether to return the internal attention weights. Defaults to False.
            
        Returns:
            torch.Tensor | tuple: The masked directed predictions of shape [Total_Edges].
                                  If return_attention is True, returns a tuple: (masked_pred, (alpha1, alpha2), node_potentials).
        """
        # --- 1. Standard Message Passing ---
        x_orig = x
        edge_attr_orig = edge_attr

        x, alpha1 = self.conv1(x, edge_index, edge_attr)
        x = torch.cat([x, x_orig], dim=-1)
        x = self.norm_x1(x)
        
        edge_attr = self.edge_updater(x, edge_index, edge_attr)
        edge_attr = torch.cat([edge_attr, edge_attr_orig], dim=-1)
        edge_attr = self.norm_e1(edge_attr)
        
        x, alpha2 = self.conv2(x, edge_index, edge_attr)
        x = torch.cat([x, x_orig], dim=-1)
        x = self.norm_x2(x)
        
        edge_attr = self.edge_updater_1(x, edge_index, edge_attr)
        edge_attr = torch.cat([edge_attr, edge_attr_orig], dim=-1)
        edge_attr = self.norm_e2(edge_attr)
        
        # The raw symmetric base predictions
        raw_pred = self.classifier(x, edge_attr, edge_index)

        # --- 2. Topological Potential Ranking & Masking ---
        # Predict a 1D continuous "Potential Score" (rank) for every single node
        node_potentials = self.rank_mlp(x).squeeze(-1) # Shape: [Total_Nodes]
        
        # Create the directional mask: ~1.0 if Source Potential < Target Potential, else ~0.0
        # This mathematically guarantees an acyclic graph (DAG)
        potential_diff = node_potentials[edge_index[1]] - node_potentials[edge_index[0]]
        directional_mask = torch.sigmoid(potential_diff / self.temperature)
        
        masked_pred = raw_pred * directional_mask
        
        if return_attention:
            return masked_pred, (alpha1, alpha2), node_potentials
        return masked_pred
