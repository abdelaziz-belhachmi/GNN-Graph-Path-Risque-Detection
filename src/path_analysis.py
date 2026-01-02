"""
Path analysis module.
Identifies critical paths in the network based on GNN predictions.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from itertools import combinations
import heapq

import config


class PathAnalyzer:
    """
    Analyzes network paths and identifies critical/risky paths.
    """
    
    def __init__(
        self,
        G: nx.Graph,
        edge_features: pd.DataFrame,
        predicted_risks: Optional[np.ndarray] = None
    ):
        """
        Initialize path analyzer.
        
        Args:
            G: NetworkX graph
            edge_features: DataFrame with edge features and risk scores
            predicted_risks: GNN predicted risk scores (optional)
        """
        self.G = G.copy()
        self.edge_features = edge_features.copy()
        self.predicted_risks = predicted_risks
        
        # Add risk scores as edge weights
        self._add_edge_weights()
    
    def _add_edge_weights(self):
        """Add risk scores as edge weights to the graph."""
        # Create edge to risk mapping
        edge_risk_map = {}
        
        for idx, row in self.edge_features.iterrows():
            src, tgt = int(row['source']), int(row['target'])
            
            if self.predicted_risks is not None:
                # Use predicted risks (take only first half, since we have bidirectional)
                risk = self.predicted_risks[idx] if idx < len(self.predicted_risks) // 2 else \
                       self.predicted_risks[idx - len(self.predicted_risks) // 2]
            else:
                risk = row['risk_score']
            
            edge_risk_map[(src, tgt)] = risk
            edge_risk_map[(tgt, src)] = risk
        
        # Add to graph
        for u, v in self.G.edges():
            if (u, v) in edge_risk_map:
                self.G[u][v]['risk'] = edge_risk_map[(u, v)]
            else:
                self.G[u][v]['risk'] = 0.5  # Default risk
    
    def get_path_risk(self, path: List[int]) -> Dict[str, float]:
        """
        Calculate risk metrics for a path.
        
        Args:
            path: List of node IDs representing a path
            
        Returns:
            Dictionary with risk metrics
        """
        if len(path) < 2:
            return {'total_risk': 0, 'max_risk': 0, 'avg_risk': 0}
        
        edge_risks = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.G.has_edge(u, v):
                edge_risks.append(self.G[u][v].get('risk', 0.5))
        
        if not edge_risks:
            return {'total_risk': 0, 'max_risk': 0, 'avg_risk': 0}
        
        return {
            'total_risk': sum(edge_risks),
            'max_risk': max(edge_risks),
            'avg_risk': np.mean(edge_risks),
            'path_length': len(path) - 1
        }
    
    def find_shortest_path(self, source: int, target: int) -> Tuple[List[int], float]:
        """
        Find shortest path (by hop count) between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Tuple of (path, path_length)
        """
        try:
            path = nx.shortest_path(self.G, source, target)
            return path, len(path) - 1
        except nx.NetworkXNoPath:
            return [], float('inf')
    
    def find_riskiest_path(self, source: int, target: int) -> Tuple[List[int], float]:
        """
        Find the path with highest cumulative risk between two nodes.
        Uses negative weights to find longest path in terms of risk.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Tuple of (path, total_risk)
        """
        # Use Dijkstra with inverted weights
        G_inv = self.G.copy()
        for u, v in G_inv.edges():
            G_inv[u][v]['weight'] = 1 - G_inv[u][v].get('risk', 0.5)
        
        try:
            path = nx.shortest_path(G_inv, source, target, weight='weight')
            risk_metrics = self.get_path_risk(path)
            return path, risk_metrics['total_risk']
        except nx.NetworkXNoPath:
            return [], 0
    
    def find_safest_path(self, source: int, target: int) -> Tuple[List[int], float]:
        """
        Find the path with lowest cumulative risk between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Tuple of (path, total_risk)
        """
        G_risk = self.G.copy()
        for u, v in G_risk.edges():
            G_risk[u][v]['weight'] = G_risk[u][v].get('risk', 0.5)
        
        try:
            path = nx.shortest_path(G_risk, source, target, weight='weight')
            risk_metrics = self.get_path_risk(path)
            return path, risk_metrics['total_risk']
        except nx.NetworkXNoPath:
            return [], float('inf')
    
    def find_top_critical_paths(
        self,
        k: int = config.TOP_K_CRITICAL_PATHS,
        sample_pairs: int = 50
    ) -> List[Dict]:
        """
        Find top-k most critical paths in the network.
        
        Args:
            k: Number of top paths to return
            sample_pairs: Number of node pairs to sample
            
        Returns:
            List of dictionaries with path information
        """
        nodes = list(self.G.nodes())
        
        # Sample node pairs
        if len(nodes) <= sample_pairs:
            pairs = list(combinations(nodes, 2))
        else:
            np.random.seed(config.RANDOM_SEED)
            sampled_nodes = np.random.choice(nodes, min(sample_pairs, len(nodes)), replace=False)
            pairs = list(combinations(sampled_nodes, 2))
        
        # Calculate risk for all paths
        path_risks = []
        
        for source, target in pairs:
            path, _ = self.find_safest_path(source, target)  # Even safest path might be risky
            if path:
                metrics = self.get_path_risk(path)
                path_risks.append({
                    'source': source,
                    'target': target,
                    'path': path,
                    **metrics
                })
        
        # Sort by total risk (descending)
        path_risks.sort(key=lambda x: x['total_risk'], reverse=True)
        
        return path_risks[:k]
    
    def find_critical_edges(self, top_k: int = 10) -> pd.DataFrame:
        """
        Find edges that appear most frequently in critical paths.
        
        Args:
            top_k: Number of top edges to return
            
        Returns:
            DataFrame with critical edge information
        """
        critical_paths = self.find_top_critical_paths(k=50)
        
        edge_counts = {}
        edge_total_risk = {}
        
        for path_info in critical_paths:
            path = path_info['path']
            for i in range(len(path) - 1):
                edge = tuple(sorted([path[i], path[i + 1]]))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
                if edge not in edge_total_risk:
                    edge_total_risk[edge] = self.G[edge[0]][edge[1]].get('risk', 0.5)
        
        # Create DataFrame
        critical_edges = []
        for edge, count in edge_counts.items():
            critical_edges.append({
                'source': edge[0],
                'target': edge[1],
                'critical_path_count': count,
                'risk_score': edge_total_risk[edge]
            })
        
        df = pd.DataFrame(critical_edges)
        if not df.empty:
            df = df.sort_values('critical_path_count', ascending=False).head(top_k)
        
        return df
    
    def compare_with_static_metrics(self) -> pd.DataFrame:
        """
        Compare GNN predictions with static graph metrics.
        
        Returns:
            DataFrame with comparison
        """
        # Calculate edge betweenness centrality
        edge_betweenness = nx.edge_betweenness_centrality(self.G)
        
        comparison = []
        
        for idx, row in self.edge_features.iterrows():
            src, tgt = int(row['source']), int(row['target'])
            
            # Get betweenness (try both edge directions)
            betweenness = edge_betweenness.get((src, tgt), 
                                                edge_betweenness.get((tgt, src), 0))
            
            if self.predicted_risks is not None and idx < len(self.predicted_risks) // 2:
                pred_risk = self.predicted_risks[idx]
            else:
                pred_risk = row['risk_score']
            
            comparison.append({
                'source': src,
                'target': tgt,
                'gnn_risk': pred_risk,
                'true_risk': row['risk_score'],
                'edge_betweenness': betweenness,
                'latency': row['latency'],
                'utilization': row['utilization'],
            })
        
        df = pd.DataFrame(comparison)
        
        # Add correlation analysis
        if len(df) > 1:
            print("\nCorrelation Analysis:")
            print(f"  GNN Risk vs True Risk: {df['gnn_risk'].corr(df['true_risk']):.4f}")
            print(f"  GNN Risk vs Betweenness: {df['gnn_risk'].corr(df['edge_betweenness']):.4f}")
            print(f"  GNN Risk vs Latency: {df['gnn_risk'].corr(df['latency']):.4f}")
            print(f"  GNN Risk vs Utilization: {df['gnn_risk'].corr(df['utilization']):.4f}")
        
        return df


def analyze_paths(
    G: nx.Graph,
    edge_features: pd.DataFrame,
    predicted_risks: Optional[np.ndarray] = None
) -> Dict:
    """
    Perform full path analysis.
    
    Args:
        G: NetworkX graph
        edge_features: DataFrame with edge features
        predicted_risks: Optional predicted risk scores
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = PathAnalyzer(G, edge_features, predicted_risks)
    
    print("\n" + "=" * 50)
    print("PATH ANALYSIS RESULTS")
    print("=" * 50)
    
    # Find critical paths
    print("\nTop Critical Paths:")
    critical_paths = analyzer.find_top_critical_paths()
    for i, path_info in enumerate(critical_paths[:5], 1):
        print(f"  {i}. {path_info['source']} -> {path_info['target']}")
        print(f"     Path: {' -> '.join(map(str, path_info['path']))}")
        print(f"     Total Risk: {path_info['total_risk']:.4f}, "
              f"Max Risk: {path_info['max_risk']:.4f}")
    
    # Find critical edges
    print("\nTop Critical Edges:")
    critical_edges = analyzer.find_critical_edges()
    print(critical_edges.to_string(index=False))
    
    # Compare with static metrics
    comparison = analyzer.compare_with_static_metrics()
    
    return {
        'critical_paths': critical_paths,
        'critical_edges': critical_edges,
        'comparison': comparison,
        'analyzer': analyzer
    }


if __name__ == "__main__":
    from data_loader import load_gml_graph
    from feature_generator import generate_all_features
    
    # Test path analysis
    G = load_gml_graph(config.DEFAULT_NETWORK)
    node_features, edge_features = generate_all_features(G)
    
    results = analyze_paths(G, edge_features)
