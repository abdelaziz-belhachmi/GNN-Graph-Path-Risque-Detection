"""
Main entry point for the Network Risk Path Detection project.
Orchestrates the full pipeline: data loading, training, analysis, and visualization.
"""

import os
import sys
import argparse
import json
import numpy as np

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_loader import load_gml_graph, list_available_networks, get_network_info
from src.feature_generator import generate_all_features
from src.dataset import create_pyg_data, create_edge_splits
from src.models import EdgeRiskGNN
from src.train import train_model, Trainer
from src.path_analysis import analyze_paths, PathAnalyzer
from src.visualization import create_all_visualizations


def run_pipeline(
    network_name: str = config.DEFAULT_NETWORK,
    gnn_type: str = 'sage',
    epochs: int = config.EPOCHS,
    skip_training: bool = False,
    visualize: bool = True
):
    """
    Run the complete pipeline for network risk path detection.
    
    Args:
        network_name: Name of the network to analyze
        gnn_type: Type of GNN architecture ('gcn', 'gat', 'sage')
        epochs: Number of training epochs
        skip_training: Whether to skip training and load existing model
        visualize: Whether to generate visualizations
    """
    print("=" * 60)
    print("NETWORK RISK PATH DETECTION - GNN Pipeline")
    print("=" * 60)
    
    # Step 1: Load network topology
    print(f"\n[1/5] Loading network: {network_name}")
    print("-" * 40)
    G = load_gml_graph(network_name)
    info = get_network_info(G)
    print(f"Network statistics:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Step 2: Generate synthetic features
    print(f"\n[2/5] Generating synthetic features")
    print("-" * 40)
    node_features, edge_features = generate_all_features(G)
    print(f"Node features: {list(node_features.columns)}")
    print(f"Edge features: {list(edge_features.columns)}")
    print(f"\nNode feature statistics:")
    print(node_features.describe().round(3))
    print(f"\nEdge feature statistics:")
    print(edge_features.describe().round(3))
    
    # Step 3: Prepare PyTorch Geometric data
    print(f"\n[3/5] Preparing PyTorch Geometric data")
    print("-" * 40)
    data = create_pyg_data(G, node_features, edge_features)
    data = create_edge_splits(data)
    print(f"Data object created:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Node features: {data.x.shape}")
    print(f"  Edge features: {data.edge_attr.shape}")
    print(f"  Train/Val/Test edges: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")
    
    # Step 4: Train GNN model
    print(f"\n[4/5] Training GNN model ({gnn_type.upper()})")
    print("-" * 40)
    
    model_path = os.path.join(config.MODELS_DIR, 'best_model.pt')
    
    if skip_training and os.path.exists(model_path):
        print("Loading existing model...")
        model = EdgeRiskGNN(gnn_type=gnn_type)
        trainer = Trainer(model)
        trainer.load_model(model_path)
        history = None
    else:
        model, trainer, history = train_model(
            data, 
            model_type='gnn', 
            gnn_type=gnn_type
        )
    
    # Get predictions
    predicted_risks = trainer.predict(data)
    print(f"\nPredicted risk scores range: [{predicted_risks.min():.4f}, {predicted_risks.max():.4f}]")
    
    # Step 5: Path analysis
    print(f"\n[5/5] Analyzing critical paths")
    print("-" * 40)
    analysis_results = analyze_paths(G, edge_features, predicted_risks)
    
    # Save results
    print(f"\nSaving results to {config.OUTPUT_DIR}")
    
    # Save edge features with predictions
    edge_features_output = edge_features.copy()
    edge_features_output['predicted_risk'] = predicted_risks[:len(edge_features)]
    edge_features_output.to_csv(
        os.path.join(config.OUTPUT_DIR, 'edge_predictions.csv'),
        index=False
    )
    
    # Save critical paths
    critical_paths_data = []
    for path_info in analysis_results['critical_paths']:
        critical_paths_data.append({
            'source': int(path_info['source']),
            'target': int(path_info['target']),
            'path': ' -> '.join(map(str, path_info['path'])),
            'total_risk': float(path_info['total_risk']),
            'max_risk': float(path_info['max_risk']),
            'avg_risk': float(path_info['avg_risk']),
            'path_length': int(path_info['path_length'])
        })
    
    with open(os.path.join(config.OUTPUT_DIR, 'critical_paths.json'), 'w') as f:
        json.dump(critical_paths_data, f, indent=2, cls=NumpyEncoder)
    
    # Save critical edges
    analysis_results['critical_edges'].to_csv(
        os.path.join(config.OUTPUT_DIR, 'critical_edges.csv'),
        index=False
    )
    
    # Generate visualizations
    if visualize:
        print(f"\nGenerating visualizations...")
        create_all_visualizations(
            G=G,
            node_features=node_features,
            edge_features=edge_features,
            predicted_risks=predicted_risks,
            history=history,
            critical_paths=analysis_results['critical_paths'],
            comparison_df=analysis_results['comparison']
        )
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nOutput files saved to: {config.OUTPUT_DIR}")
    print("  - edge_predictions.csv: Edge features with predicted risks")
    print("  - critical_paths.json: Top critical paths")
    print("  - critical_edges.csv: Most critical edges")
    print("  - test_metrics.json: Model performance metrics")
    print("  - risk_map.png: Network risk visualization")
    print("  - training_history.png: Training curves")
    print("  - risk_distribution.png: Risk score distribution")
    print("  - critical_paths.png: Critical path visualizations")
    print("  - comparison_metrics.png: GNN vs static metrics comparison")
    
    return {
        'graph': G,
        'node_features': node_features,
        'edge_features': edge_features,
        'predictions': predicted_risks,
        'model': model,
        'analysis': analysis_results
    }


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Network Risk Path Detection using Graph Neural Networks'
    )
    parser.add_argument(
        '--network', '-n',
        type=str,
        default=config.DEFAULT_NETWORK,
        help=f'Network name to analyze (default: {config.DEFAULT_NETWORK})'
    )
    parser.add_argument(
        '--gnn-type', '-g',
        type=str,
        choices=['gcn', 'gat', 'sage'],
        default='sage',
        help='GNN architecture type (default: sage)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=config.EPOCHS,
        help=f'Number of training epochs (default: {config.EPOCHS})'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training and load existing model'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip visualization generation'
    )
    parser.add_argument(
        '--list-networks',
        action='store_true',
        help='List available networks and exit'
    )
    
    args = parser.parse_args()
    
    if args.list_networks:
        networks = list_available_networks()
        print(f"Available networks ({len(networks)} total):")
        for i, network in enumerate(networks, 1):
            print(f"  {i:3d}. {network}")
        return
    
    run_pipeline(
        network_name=args.network,
        gnn_type=args.gnn_type,
        epochs=args.epochs,
        skip_training=args.skip_training,
        visualize=not args.no_visualize
    )


if __name__ == "__main__":
    main()
