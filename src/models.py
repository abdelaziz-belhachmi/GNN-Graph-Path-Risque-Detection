"""
Graph Neural Network models for edge risk prediction.
Implements GNN architectures that learn to predict link risk scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, BatchNorm
from torch_geometric.data import Data
from typing import Optional

import config


class EdgeRiskGNN(nn.Module):
    """
    GNN model for predicting edge risk scores.
    
    Uses node embeddings from GNN layers combined with edge features
    to predict risk scores for each edge.
    """
    
    def __init__(
        self,
        node_feature_dim: int = config.NODE_FEATURE_DIM,
        edge_feature_dim: int = config.EDGE_FEATURE_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = config.NUM_GNN_LAYERS,
        dropout: float = config.DROPOUT,
        gnn_type: str = 'sage'
    ):
        super(EdgeRiskGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Node feature encoder
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if gnn_type == 'gcn':
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gat':
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            elif gnn_type == 'sage':
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
            
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge risk predictor
        # Input: source node embedding + target node embedding + edge features
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output risk score between 0 and 1
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Risk scores for each edge
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Encode node features
        x = self.node_encoder(x)
        x = F.relu(x)
        
        # Apply GNN layers with residual connections
        for i in range(self.num_layers):
            x_new = self.gnn_layers[i](x, edge_index)
            x_new = self.batch_norms[i](x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # Residual connection
        
        # Get node embeddings for source and target of each edge
        src_nodes = edge_index[0]
        tgt_nodes = edge_index[1]
        
        src_embeddings = x[src_nodes]  # [num_edges, hidden_dim]
        tgt_embeddings = x[tgt_nodes]  # [num_edges, hidden_dim]
        
        # Encode edge features
        edge_embeddings = self.edge_encoder(edge_attr)  # [num_edges, hidden_dim]
        
        # Concatenate all features for edge prediction
        edge_repr = torch.cat([src_embeddings, tgt_embeddings, edge_embeddings], dim=1)
        
        # Predict risk scores
        risk_scores = self.edge_predictor(edge_repr).squeeze(-1)
        
        return risk_scores
    
    def get_node_embeddings(self, data: Data) -> torch.Tensor:
        """
        Get learned node embeddings.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Node embeddings tensor
        """
        x, edge_index = data.x, data.edge_index
        
        x = self.node_encoder(x)
        x = F.relu(x)
        
        for i in range(self.num_layers):
            x_new = self.gnn_layers[i](x, edge_index)
            x_new = self.batch_norms[i](x_new)
            x_new = F.relu(x_new)
            x = x + x_new
        
        return x


class SimpleEdgeMLP(nn.Module):
    """
    Simple MLP baseline for edge risk prediction.
    Does not use graph structure (for comparison).
    """
    
    def __init__(
        self,
        node_feature_dim: int = config.NODE_FEATURE_DIM,
        edge_feature_dim: int = config.EDGE_FEATURE_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        dropout: float = config.DROPOUT
    ):
        super(SimpleEdgeMLP, self).__init__()
        
        # Input: source node features + target node features + edge features
        input_dim = node_feature_dim * 2 + edge_feature_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        src_nodes = edge_index[0]
        tgt_nodes = edge_index[1]
        
        src_features = x[src_nodes]
        tgt_features = x[tgt_nodes]
        
        edge_repr = torch.cat([src_features, tgt_features, edge_attr], dim=1)
        risk_scores = self.mlp(edge_repr).squeeze(-1)
        
        return risk_scores


if __name__ == "__main__":
    # Test model architecture
    from data_loader import load_gml_graph
    from feature_generator import generate_all_features
    from dataset import create_pyg_data, create_edge_splits
    
    G = load_gml_graph(config.DEFAULT_NETWORK)
    node_features, edge_features = generate_all_features(G)
    data = create_pyg_data(G, node_features, edge_features)
    data = create_edge_splits(data)
    
    # Test GNN model
    model = EdgeRiskGNN()
    print(f"\nModel architecture:\n{model}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(data)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
