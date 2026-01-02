"""
Feature generator module.
Generates synthetic node and edge features for network graphs.
"""

import numpy as np
import networkx as nx
from typing import Tuple, Dict
import pandas as pd

import config


def set_seed(seed: int = config.RANDOM_SEED):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def generate_node_features(G: nx.Graph) -> pd.DataFrame:
    """
    Generate synthetic node features.
    
    Features:
        - load: CPU/memory load on the node (0-1)
        - degree_centrality: Normalized degree
        - betweenness_centrality: Betweenness centrality
    
    Args:
        G: NetworkX graph
        
    Returns:
        DataFrame with node features indexed by node ID
    """
    set_seed()
    
    num_nodes = G.number_of_nodes()
    
    # Generate synthetic load (correlated with node degree for realism)
    degrees = np.array([G.degree(n) for n in G.nodes()])
    normalized_degrees = degrees / degrees.max()
    
    # Base load + degree-correlated component + noise
    base_load = np.random.uniform(0.2, 0.5, num_nodes)
    load = base_load + 0.3 * normalized_degrees + np.random.normal(0, 0.1, num_nodes)
    load = np.clip(load, config.NODE_LOAD_RANGE[0], config.NODE_LOAD_RANGE[1])
    
    # Calculate centrality metrics
    degree_centrality = np.array(list(nx.degree_centrality(G).values()))
    betweenness_centrality = np.array(list(nx.betweenness_centrality(G).values()))
    
    # Normalize betweenness centrality
    if betweenness_centrality.max() > 0:
        betweenness_centrality = betweenness_centrality / betweenness_centrality.max()
    
    node_features = pd.DataFrame({
        'node_id': list(G.nodes()),
        'load': load,
        'degree_centrality': degree_centrality,
        'betweenness_centrality': betweenness_centrality,
    }).set_index('node_id')
    
    return node_features


def generate_edge_features(G: nx.Graph, node_features: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic edge features.
    
    Features:
        - latency: Link latency in ms
        - bandwidth: Link bandwidth in Mbps
        - utilization: Current utilization rate (0-1)
        - loss_rate: Packet loss rate (0-1)
    
    Args:
        G: NetworkX graph
        node_features: DataFrame with node features
        
    Returns:
        DataFrame with edge features
    """
    set_seed()
    
    edges = list(G.edges())
    num_edges = len(edges)
    
    # Calculate geographic distance if coordinates available
    distances = []
    for u, v in edges:
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        if 'Longitude' in u_data and 'Latitude' in u_data:
            # Haversine distance approximation
            lon1, lat1 = u_data['Longitude'], u_data['Latitude']
            lon2, lat2 = v_data['Longitude'], v_data['Latitude']
            dist = np.sqrt((lon2-lon1)**2 + (lat2-lat1)**2) * 111  # Rough km conversion
            distances.append(dist)
        else:
            distances.append(None)
    
    # Generate latency (correlated with distance if available)
    if all(d is not None for d in distances):
        distances = np.array(distances)
        normalized_dist = distances / distances.max() if distances.max() > 0 else distances
        latency = 5 + 50 * normalized_dist + np.random.normal(0, 5, num_edges)
    else:
        latency = np.random.uniform(*config.EDGE_LATENCY_RANGE, num_edges)
    
    latency = np.clip(latency, config.EDGE_LATENCY_RANGE[0], config.EDGE_LATENCY_RANGE[1])
    
    # Generate bandwidth
    bandwidth = np.random.uniform(*config.EDGE_BANDWIDTH_RANGE, num_edges)
    
    # Generate utilization (higher for links between high-load nodes)
    utilization = []
    for i, (u, v) in enumerate(edges):
        avg_node_load = (node_features.loc[u, 'load'] + node_features.loc[v, 'load']) / 2
        # Higher node load leads to higher link utilization
        base_util = avg_node_load * 0.6 + np.random.uniform(0.1, 0.3)
        utilization.append(np.clip(base_util, *config.EDGE_UTILIZATION_RANGE))
    utilization = np.array(utilization)
    
    # Generate loss rate (correlated with utilization)
    loss_rate = 0.001 + utilization * 0.03 + np.random.exponential(0.005, num_edges)
    loss_rate = np.clip(loss_rate, *config.EDGE_LOSS_RATE_RANGE)
    
    edge_features = pd.DataFrame({
        'source': [e[0] for e in edges],
        'target': [e[1] for e in edges],
        'latency': latency,
        'bandwidth': bandwidth,
        'utilization': utilization,
        'loss_rate': loss_rate,
    })
    
    return edge_features


def calculate_edge_risk_score(edge_features: pd.DataFrame, 
                               node_features: pd.DataFrame) -> np.ndarray:
    """
    Calculate synthetic risk scores for edges.
    This serves as the ground truth for GNN training.
    
    Risk is calculated based on:
        - Normalized latency
        - Utilization rate
        - Loss rate
        - Average node load of endpoints
    
    Args:
        edge_features: DataFrame with edge features
        node_features: DataFrame with node features
        
    Returns:
        Array of risk scores (0-1)
    """
    # Normalize features
    latency_norm = (edge_features['latency'] - edge_features['latency'].min()) / \
                   (edge_features['latency'].max() - edge_features['latency'].min() + 1e-8)
    
    utilization = edge_features['utilization']
    loss_rate_norm = edge_features['loss_rate'] / config.EDGE_LOSS_RATE_RANGE[1]
    
    # Calculate average node load for each edge
    avg_node_load = []
    for _, row in edge_features.iterrows():
        src_load = node_features.loc[row['source'], 'load']
        tgt_load = node_features.loc[row['target'], 'load']
        avg_node_load.append((src_load + tgt_load) / 2)
    avg_node_load = np.array(avg_node_load)
    
    # Calculate weighted risk score
    weights = config.RISK_WEIGHTS
    risk_score = (
        weights['latency'] * latency_norm +
        weights['utilization'] * utilization +
        weights['loss_rate'] * loss_rate_norm +
        weights['node_load'] * avg_node_load
    )
    
    # Normalize to 0-1 range
    risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min() + 1e-8)
    
    return risk_score.values


def generate_all_features(G: nx.Graph) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate all features for a graph.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Tuple of (node_features, edge_features) DataFrames
    """
    node_features = generate_node_features(G)
    edge_features = generate_edge_features(G, node_features)
    edge_features['risk_score'] = calculate_edge_risk_score(edge_features, node_features)
    
    return node_features, edge_features


if __name__ == "__main__":
    from data_loader import load_gml_graph
    
    # Test feature generation
    G = load_gml_graph(config.DEFAULT_NETWORK)
    node_features, edge_features = generate_all_features(G)
    
    print("\nNode Features:")
    print(node_features.describe())
    
    print("\nEdge Features:")
    print(edge_features.describe())
