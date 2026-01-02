"""
Network Risk Path Detection using Graph Neural Networks.

This package provides tools to:
- Load network topologies from Internet Topology Zoo
- Generate synthetic node and edge features
- Train GNN models to predict link risk scores
- Analyze and identify critical network paths
- Visualize network risk maps
"""

from .data_loader import load_gml_graph, list_available_networks, get_network_info
from .feature_generator import generate_all_features, generate_node_features, generate_edge_features
from .dataset import create_pyg_data, create_edge_splits
from .models import EdgeRiskGNN, SimpleEdgeMLP
from .train import Trainer, train_model
from .path_analysis import PathAnalyzer, analyze_paths
from .visualization import (
    plot_network_risk_map,
    plot_training_history,
    plot_risk_distribution,
    plot_critical_paths,
    plot_comparison_metrics,
    create_all_visualizations
)

__all__ = [
    'load_gml_graph',
    'list_available_networks',
    'get_network_info',
    'generate_all_features',
    'generate_node_features',
    'generate_edge_features',
    'create_pyg_data',
    'create_edge_splits',
    'EdgeRiskGNN',
    'SimpleEdgeMLP',
    'Trainer',
    'train_model',
    'PathAnalyzer',
    'analyze_paths',
    'plot_network_risk_map',
    'plot_training_history',
    'plot_risk_distribution',
    'plot_critical_paths',
    'plot_comparison_metrics',
    'create_all_visualizations',
]
