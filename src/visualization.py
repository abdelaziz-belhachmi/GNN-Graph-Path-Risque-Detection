"""
Visualization module.
Creates visualizations for network risk analysis.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
from typing import Optional, List, Dict

import config


def setup_style():
    """Setup matplotlib style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")


def plot_network_risk_map(
    G: nx.Graph,
    edge_features: pd.DataFrame,
    predicted_risks: Optional[np.ndarray] = None,
    node_features: Optional[pd.DataFrame] = None,
    title: str = "Network Risk Map",
    save_path: Optional[str] = None,
    highlight_path: Optional[List[int]] = None
):
    """
    Plot network graph with edges colored by risk score.
    
    Args:
        G: NetworkX graph
        edge_features: DataFrame with edge features
        predicted_risks: Optional predicted risk scores
        node_features: Optional node features DataFrame
        title: Plot title
        save_path: Path to save figure
        highlight_path: Optional path to highlight
    """
    setup_style()
    
    fig, ax = plt.subplots(1, 1, figsize=config.FIGURE_SIZE)
    
    # Get node positions
    positions = {}
    has_geo = True
    for node in G.nodes():
        node_data = G.nodes[node]
        if 'Longitude' in node_data and 'Latitude' in node_data:
            positions[node] = (node_data['Longitude'], node_data['Latitude'])
        else:
            has_geo = False
            break
    
    if not has_geo:
        positions = nx.spring_layout(G, seed=config.RANDOM_SEED, k=2)
    
    # Create edge to risk mapping
    edge_risks = {}
    for idx, row in edge_features.iterrows():
        src, tgt = int(row['source']), int(row['target'])
        if predicted_risks is not None and idx < len(predicted_risks) // 2:
            risk = predicted_risks[idx]
        else:
            risk = row['risk_score']
        edge_risks[(src, tgt)] = risk
        edge_risks[(tgt, src)] = risk
    
    # Get edge colors and widths
    edges = list(G.edges())
    edge_colors = [edge_risks.get((u, v), 0.5) for u, v in edges]
    edge_widths = [1 + 4 * edge_risks.get((u, v), 0.5) for u, v in edges]
    
    # Get node sizes based on load
    if node_features is not None:
        node_sizes = [
            config.NODE_SIZE_RANGE[0] + 
            (config.NODE_SIZE_RANGE[1] - config.NODE_SIZE_RANGE[0]) * 
            node_features.loc[n, 'load']
            for n in G.nodes()
        ]
        node_colors = [node_features.loc[n, 'load'] for n in G.nodes()]
    else:
        node_sizes = [500] * len(G.nodes())
        node_colors = [0.5] * len(G.nodes())
    
    # Draw edges
    cmap = plt.cm.get_cmap(config.RISK_COLORMAP)
    edge_collection = nx.draw_networkx_edges(
        G, positions,
        edge_color=edge_colors,
        edge_cmap=cmap,
        edge_vmin=0, edge_vmax=1,
        width=edge_widths,
        alpha=0.7,
        ax=ax
    )
    
    # Highlight path if provided
    if highlight_path:
        path_edges = [(highlight_path[i], highlight_path[i+1]) 
                      for i in range(len(highlight_path) - 1)]
        nx.draw_networkx_edges(
            G, positions,
            edgelist=path_edges,
            edge_color='blue',
            width=4,
            alpha=0.9,
            style='solid',
            ax=ax
        )
    
    # Draw nodes
    node_collection = nx.draw_networkx_nodes(
        G, positions,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.get_cmap('YlOrRd'),
        vmin=0, vmax=1,
        alpha=0.9,
        ax=ax
    )
    
    # Draw labels
    labels = {n: G.nodes[n].get('label', str(n))[:8] for n in G.nodes()}
    nx.draw_networkx_labels(
        G, positions,
        labels=labels,
        font_size=8,
        font_weight='bold',
        ax=ax
    )
    
    # Add colorbar for edges
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, label='Edge Risk Score')
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], color=cmap(0.2), linewidth=2, label='Low Risk'),
        Line2D([0], [0], color=cmap(0.5), linewidth=3, label='Medium Risk'),
        Line2D([0], [0], color=cmap(0.8), linewidth=4, label='High Risk'),
    ]
    if highlight_path:
        legend_elements.append(
            Line2D([0], [0], color='blue', linewidth=4, label='Highlighted Path')
        )
    ax.legend(handles=legend_elements, loc='upper left')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.close(fig)
    return fig


def plot_training_history(
    history: Dict[str, list],
    save_path: Optional[str] = None
):
    """
    Plot training and validation loss curves.
    
    Args:
        history: Dictionary with 'train_losses' and 'val_losses'
        save_path: Path to save figure
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    ax.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Mark best epoch
    best_epoch = np.argmin(history['val_losses']) + 1
    best_val_loss = min(history['val_losses'])
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label='Best Epoch')
    ax.scatter([best_epoch], [best_val_loss], color='g', s=100, zorder=5)
    ax.annotate(f'Best: {best_val_loss:.4f}', 
                xy=(best_epoch, best_val_loss),
                xytext=(best_epoch + 5, best_val_loss + 0.01),
                fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.close(fig)
    return fig


def plot_risk_distribution(
    edge_features: pd.DataFrame,
    predicted_risks: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Plot distribution of risk scores.
    
    Args:
        edge_features: DataFrame with edge features
        predicted_risks: Optional predicted risk scores
        save_path: Path to save figure
    """
    setup_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # True risk distribution
    ax1 = axes[0]
    sns.histplot(edge_features['risk_score'], bins=20, kde=True, ax=ax1, color='steelblue')
    ax1.set_xlabel('Risk Score', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('True Risk Score Distribution', fontsize=13, fontweight='bold')
    ax1.axvline(edge_features['risk_score'].mean(), color='red', linestyle='--', 
                label=f'Mean: {edge_features["risk_score"].mean():.3f}')
    ax1.legend()
    
    # Predicted vs True (if predictions available)
    ax2 = axes[1]
    if predicted_risks is not None:
        half_len = len(predicted_risks) // 2
        pred_risks = predicted_risks[:half_len]
        true_risks = edge_features['risk_score'].values
        
        ax2.scatter(true_risks, pred_risks, alpha=0.6, s=50)
        ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
        ax2.set_xlabel('True Risk Score', fontsize=12)
        ax2.set_ylabel('Predicted Risk Score', fontsize=12)
        ax2.set_title('Predicted vs True Risk', fontsize=13, fontweight='bold')
        
        # Add correlation
        corr = np.corrcoef(true_risks, pred_risks)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {corr:.4f}', 
                transform=ax2.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.legend()
    else:
        # Show feature correlations instead
        corr_data = edge_features[['risk_score', 'latency', 'utilization', 'loss_rate']].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=ax2)
        ax2.set_title('Feature Correlations', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.close(fig)
    return fig


def plot_critical_paths(
    G: nx.Graph,
    critical_paths: List[Dict],
    node_features: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None
):
    """
    Plot top critical paths on the network.
    
    Args:
        G: NetworkX graph
        critical_paths: List of critical path dictionaries
        node_features: Optional node features DataFrame
        save_path: Path to save figure
    """
    setup_style()
    
    num_paths = min(4, len(critical_paths))
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    # Get positions
    positions = {}
    has_geo = True
    for node in G.nodes():
        node_data = G.nodes[node]
        if 'Longitude' in node_data and 'Latitude' in node_data:
            positions[node] = (node_data['Longitude'], node_data['Latitude'])
        else:
            has_geo = False
            break
    
    if not has_geo:
        positions = nx.spring_layout(G, seed=config.RANDOM_SEED, k=2)
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    for i in range(num_paths):
        ax = axes[i]
        path_info = critical_paths[i]
        path = path_info['path']
        
        # Draw base network (gray)
        nx.draw_networkx_edges(G, positions, alpha=0.2, ax=ax)
        nx.draw_networkx_nodes(G, positions, node_size=200, 
                              node_color='lightgray', alpha=0.6, ax=ax)
        
        # Highlight path nodes
        path_node_sizes = [600 if n in path else 200 for n in G.nodes()]
        path_node_colors = [colors[i] if n in path else 'lightgray' for n in G.nodes()]
        
        nx.draw_networkx_nodes(G, positions, node_size=path_node_sizes,
                              node_color=path_node_colors, alpha=0.8, ax=ax)
        
        # Highlight path edges
        path_edges = [(path[j], path[j+1]) for j in range(len(path) - 1)]
        nx.draw_networkx_edges(G, positions, edgelist=path_edges,
                              edge_color=colors[i], width=4, alpha=0.9, ax=ax)
        
        # Labels for path nodes
        path_labels = {n: G.nodes[n].get('label', str(n))[:6] for n in path}
        nx.draw_networkx_labels(G, positions, labels=path_labels,
                               font_size=8, font_weight='bold', ax=ax)
        
        # Title
        ax.set_title(
            f"Critical Path {i+1}: {path_info['source']} → {path_info['target']}\n"
            f"Total Risk: {path_info['total_risk']:.3f} | "
            f"Max Edge Risk: {path_info['max_risk']:.3f} | "
            f"Length: {path_info['path_length']} hops",
            fontsize=11, fontweight='bold'
        )
        ax.axis('off')
    
    plt.suptitle('Top Critical Network Paths', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.close(fig)
    return fig


def plot_comparison_metrics(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot comparison between GNN predictions and static metrics.
    
    Args:
        comparison_df: DataFrame with comparison data
        save_path: Path to save figure
    """
    setup_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # GNN Risk vs True Risk
    ax1 = axes[0, 0]
    ax1.scatter(comparison_df['true_risk'], comparison_df['gnn_risk'], 
               alpha=0.6, s=50, c='steelblue')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2)
    ax1.set_xlabel('True Risk Score')
    ax1.set_ylabel('GNN Predicted Risk')
    ax1.set_title('GNN Prediction Accuracy')
    corr = comparison_df['true_risk'].corr(comparison_df['gnn_risk'])
    ax1.text(0.05, 0.95, f'r = {corr:.4f}', transform=ax1.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # GNN Risk vs Edge Betweenness
    ax2 = axes[0, 1]
    ax2.scatter(comparison_df['edge_betweenness'], comparison_df['gnn_risk'],
               alpha=0.6, s=50, c='coral')
    ax2.set_xlabel('Edge Betweenness Centrality')
    ax2.set_ylabel('GNN Predicted Risk')
    ax2.set_title('GNN Risk vs Betweenness Centrality')
    corr = comparison_df['edge_betweenness'].corr(comparison_df['gnn_risk'])
    ax2.text(0.05, 0.95, f'r = {corr:.4f}', transform=ax2.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Risk by Utilization
    ax3 = axes[1, 0]
    ax3.scatter(comparison_df['utilization'], comparison_df['gnn_risk'],
               alpha=0.6, s=50, c='seagreen')
    ax3.set_xlabel('Link Utilization')
    ax3.set_ylabel('GNN Predicted Risk')
    ax3.set_title('GNN Risk vs Link Utilization')
    corr = comparison_df['utilization'].corr(comparison_df['gnn_risk'])
    ax3.text(0.05, 0.95, f'r = {corr:.4f}', transform=ax3.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Feature importance (correlation bar chart)
    ax4 = axes[1, 1]
    correlations = {
        'Latency': comparison_df['latency'].corr(comparison_df['gnn_risk']),
        'Utilization': comparison_df['utilization'].corr(comparison_df['gnn_risk']),
        'Betweenness': comparison_df['edge_betweenness'].corr(comparison_df['gnn_risk']),
        'True Risk': comparison_df['true_risk'].corr(comparison_df['gnn_risk'])
    }
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    bars = ax4.bar(correlations.keys(), correlations.values(), color=colors)
    ax4.set_ylabel('Correlation with GNN Risk')
    ax4.set_title('Feature Correlations with GNN Predictions')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, correlations.values()):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.close(fig)
    return fig


def create_all_visualizations(
    G: nx.Graph,
    node_features: pd.DataFrame,
    edge_features: pd.DataFrame,
    predicted_risks: Optional[np.ndarray] = None,
    history: Optional[Dict] = None,
    critical_paths: Optional[List[Dict]] = None,
    comparison_df: Optional[pd.DataFrame] = None
):
    """
    Create all visualizations and save to output directory.
    
    Args:
        G: NetworkX graph
        node_features: Node features DataFrame
        edge_features: Edge features DataFrame
        predicted_risks: Predicted risk scores
        history: Training history
        critical_paths: Critical paths list
        comparison_df: Comparison DataFrame
    """
    print("\nGenerating visualizations...")
    
    # 1. Network risk map
    plot_network_risk_map(
        G, edge_features, predicted_risks, node_features,
        title="Network Risk Map (GNN Predictions)" if predicted_risks is not None else "Network Risk Map",
        save_path=os.path.join(config.OUTPUT_DIR, 'risk_map.png')
    )
    
    # 2. Training history
    if history:
        plot_training_history(
            history,
            save_path=os.path.join(config.OUTPUT_DIR, 'training_history.png')
        )
    
    # 3. Risk distribution
    plot_risk_distribution(
        edge_features, predicted_risks,
        save_path=os.path.join(config.OUTPUT_DIR, 'risk_distribution.png')
    )
    
    # 4. Critical paths
    if critical_paths:
        plot_critical_paths(
            G, critical_paths, node_features,
            save_path=os.path.join(config.OUTPUT_DIR, 'critical_paths.png')
        )
    
    # 5. Comparison metrics
    if comparison_df is not None:
        plot_comparison_metrics(
            comparison_df,
            save_path=os.path.join(config.OUTPUT_DIR, 'comparison_metrics.png')
        )
    
    print(f"\nAll visualizations saved to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    from data_loader import load_gml_graph
    from feature_generator import generate_all_features
    from path_analysis import analyze_paths
    
    # Test visualizations
    G = load_gml_graph(config.DEFAULT_NETWORK)
    node_features, edge_features = generate_all_features(G)
    
    # Plot basic risk map
    plot_network_risk_map(
        G, edge_features, None, node_features,
        title=f"Network Risk Map - {config.DEFAULT_NETWORK}",
        save_path=os.path.join(config.OUTPUT_DIR, 'test_risk_map.png')
    )
