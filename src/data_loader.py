"""
Graph data loader module.
Loads network topology from GML files and creates NetworkX graphs.
"""

import os
import networkx as nx
import numpy as np
from typing import Optional, List, Tuple

import config


def load_gml_graph(network_name: str) -> nx.Graph:
    """
    Load a network topology from a GML file.
    
    Args:
        network_name: Name of the network (without extension)
        
    Returns:
        NetworkX graph object
    """
    gml_path = os.path.join(config.DATA_DIR, f"{network_name}.gml")
    
    if not os.path.exists(gml_path):
        raise FileNotFoundError(f"Network file not found: {gml_path}")
    
    # Load the graph
    G = nx.read_gml(gml_path, label='id')
    
    # Convert to undirected if directed
    if G.is_directed():
        G = G.to_undirected()
    
    # Ensure the graph is connected (take largest component)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"Warning: Graph was not connected. Using largest component with {len(G.nodes())} nodes.")
    
    # Relabel nodes to consecutive integers
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    
    print(f"Loaded network '{network_name}': {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G


def list_available_networks() -> List[str]:
    """
    List all available network topologies in the data directory.
    
    Returns:
        List of network names
    """
    networks = []
    for filename in os.listdir(config.DATA_DIR):
        if filename.endswith('.gml'):
            networks.append(filename[:-4])  # Remove .gml extension
    return sorted(networks)


def get_network_info(G: nx.Graph) -> dict:
    """
    Get basic information about a network graph.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary with network statistics
    """
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "diameter": nx.diameter(G) if nx.is_connected(G) else None,
        "avg_clustering": nx.average_clustering(G),
        "is_connected": nx.is_connected(G),
    }


def get_node_positions(G: nx.Graph) -> dict:
    """
    Get node positions for visualization.
    Uses geographic coordinates if available, otherwise spring layout.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary mapping node IDs to (x, y) positions
    """
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
        # Use spring layout as fallback
        positions = nx.spring_layout(G, seed=config.RANDOM_SEED)
    
    return positions


if __name__ == "__main__":
    # Test the data loader
    print("Available networks:")
    networks = list_available_networks()
    print(f"Found {len(networks)} networks")
    print(f"First 10: {networks[:10]}")
    
    print(f"\nLoading default network: {config.DEFAULT_NETWORK}")
    G = load_gml_graph(config.DEFAULT_NETWORK)
    info = get_network_info(G)
    print(f"Network info: {info}")
