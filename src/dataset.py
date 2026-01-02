"""
PyTorch Geometric dataset preparation.
Converts NetworkX graphs to PyTorch Geometric Data objects.
"""

import torch
import numpy as np
import networkx as nx
import pandas as pd
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from typing import Tuple

import config


def create_pyg_data(G: nx.Graph, 
                    node_features: pd.DataFrame, 
                    edge_features: pd.DataFrame) -> Data:
    """
    Convert NetworkX graph with features to PyTorch Geometric Data object.
    
    Args:
        G: NetworkX graph
        node_features: DataFrame with node features
        edge_features: DataFrame with edge features
        
    Returns:
        PyTorch Geometric Data object
    """
    # Prepare node features tensor
    node_feat_cols = ['load', 'degree_centrality', 'betweenness_centrality']
    x = torch.tensor(node_features[node_feat_cols].values, dtype=torch.float)
    
    # Prepare edge index (COO format) - need both directions for undirected graph
    sources = edge_features['source'].values
    targets = edge_features['target'].values
    
    # Create bidirectional edges
    edge_index = torch.tensor(
        np.array([
            np.concatenate([sources, targets]),
            np.concatenate([targets, sources])
        ]), dtype=torch.long
    )
    
    # Prepare edge features tensor (duplicate for both directions)
    edge_feat_cols = ['latency', 'bandwidth', 'utilization', 'loss_rate']
    edge_attr_half = edge_features[edge_feat_cols].values
    edge_attr = torch.tensor(
        np.concatenate([edge_attr_half, edge_attr_half], axis=0), 
        dtype=torch.float
    )
    
    # Normalize edge features
    edge_attr = normalize_features(edge_attr)
    
    # Prepare edge labels (risk scores) - duplicate for both directions
    risk_scores = edge_features['risk_score'].values
    edge_labels = torch.tensor(
        np.concatenate([risk_scores, risk_scores]), 
        dtype=torch.float
    )
    
    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_labels=edge_labels,
        num_nodes=len(G.nodes()),
    )
    
    return data


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """
    Normalize features to zero mean and unit variance.
    
    Args:
        features: Feature tensor
        
    Returns:
        Normalized feature tensor
    """
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    std[std == 0] = 1  # Avoid division by zero
    return (features - mean) / std


def create_edge_splits(data: Data, 
                       train_ratio: float = config.TRAIN_RATIO,
                       val_ratio: float = config.VAL_RATIO) -> Data:
    """
    Create train/val/test splits for edges.
    
    Args:
        data: PyTorch Geometric Data object
        train_ratio: Ratio of training edges
        val_ratio: Ratio of validation edges
        
    Returns:
        Data object with train/val/test masks
    """
    num_edges = data.edge_index.size(1)
    half_edges = num_edges // 2  # Since we have bidirectional edges
    
    # Create indices for original edges only (not duplicated)
    indices = np.arange(half_edges)
    
    # Split indices
    train_idx, temp_idx = train_test_split(
        indices, train_size=train_ratio, random_state=config.RANDOM_SEED
    )
    val_size = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_size, random_state=config.RANDOM_SEED
    )
    
    # Create masks for both directions
    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    val_mask = torch.zeros(num_edges, dtype=torch.bool)
    test_mask = torch.zeros(num_edges, dtype=torch.bool)
    
    # Set masks for both directions
    train_mask[train_idx] = True
    train_mask[train_idx + half_edges] = True
    val_mask[val_idx] = True
    val_mask[val_idx + half_edges] = True
    test_mask[test_idx] = True
    test_mask[test_idx + half_edges] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data


if __name__ == "__main__":
    from data_loader import load_gml_graph
    from feature_generator import generate_all_features
    
    # Test dataset creation
    G = load_gml_graph(config.DEFAULT_NETWORK)
    node_features, edge_features = generate_all_features(G)
    
    data = create_pyg_data(G, node_features, edge_features)
    data = create_edge_splits(data)
    
    print(f"\nPyG Data object:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Node features shape: {data.x.shape}")
    print(f"  Edge features shape: {data.edge_attr.shape}")
    print(f"  Edge labels shape: {data.edge_labels.shape}")
    print(f"  Train edges: {data.train_mask.sum().item()}")
    print(f"  Val edges: {data.val_mask.sum().item()}")
    print(f"  Test edges: {data.test_mask.sum().item()}")
