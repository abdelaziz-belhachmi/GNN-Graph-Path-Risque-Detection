"""
Training module for the edge risk prediction GNN.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.models import EdgeRiskGNN, SimpleEdgeMLP


class Trainer:
    """
    Trainer class for edge risk prediction models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = config.LEARNING_RATE,
        weight_decay: float = config.WEIGHT_DECAY,
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, data: Data) -> float:
        """
        Train for one epoch.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Training loss
        """
        self.model.train()
        data = data.to(self.device)
        
        self.optimizer.zero_grad()
        
        predictions = self.model(data)
        loss = self.criterion(predictions[data.train_mask], data.edge_labels[data.train_mask])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, data: Data, mask: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model on a subset of edges.
        
        Args:
            data: PyTorch Geometric Data object
            mask: Boolean mask for edges to evaluate
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        data = data.to(self.device)
        
        predictions = self.model(data)
        predictions = predictions[mask].cpu().numpy()
        labels = data.edge_labels[mask].cpu().numpy()
        
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2
        }
    
    def train(
        self,
        data: Data,
        epochs: int = config.EPOCHS,
        patience: int = config.EARLY_STOPPING_PATIENCE,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Full training loop with early stopping.
        
        Args:
            data: PyTorch Geometric Data object
            epochs: Maximum number of epochs
            patience: Early stopping patience
            verbose: Whether to print progress
            
        Returns:
            Dictionary of training history
        """
        data = data.to(self.device)
        
        pbar = tqdm(range(epochs), desc="Training", disable=not verbose)
        
        for epoch in pbar:
            train_loss = self.train_epoch(data)
            val_metrics = self.evaluate(data, data.val_mask)
            val_loss = val_metrics['mse']
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                self.save_model(os.path.join(config.MODELS_DIR, 'best_model.pt'))
            else:
                self.patience_counter += 1
            
            pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_r2': f'{val_metrics["r2"]:.4f}'
            })
            
            if self.patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        self.load_model(os.path.join(config.MODELS_DIR, 'best_model.pt'))
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }, path)
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
    
    @torch.no_grad()
    def predict(self, data: Data) -> np.ndarray:
        """
        Get predictions for all edges.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Array of risk predictions
        """
        self.model.eval()
        data = data.to(self.device)
        predictions = self.model(data)
        return predictions.cpu().numpy()


def train_model(
    data: Data,
    model_type: str = 'gnn',
    gnn_type: str = 'sage',
    **kwargs
) -> Tuple[nn.Module, Trainer, Dict]:
    """
    Train a model on the given data.
    
    Args:
        data: PyTorch Geometric Data object
        model_type: 'gnn' or 'mlp'
        gnn_type: GNN architecture ('gcn', 'gat', 'sage')
        **kwargs: Additional training arguments
        
    Returns:
        Tuple of (trained model, trainer, training history)
    """
    if model_type == 'gnn':
        model = EdgeRiskGNN(gnn_type=gnn_type)
    else:
        model = SimpleEdgeMLP()
    
    trainer = Trainer(model, **kwargs)
    history = trainer.train(data)
    
    # Final evaluation
    test_metrics = trainer.evaluate(data, data.test_mask)
    print(f"\nTest Results:")
    print(f"  MSE: {test_metrics['mse']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  R²: {test_metrics['r2']:.4f}")
    
    # Save metrics
    with open(os.path.join(config.OUTPUT_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    return model, trainer, history


if __name__ == "__main__":
    from data_loader import load_gml_graph
    from feature_generator import generate_all_features
    from dataset import create_pyg_data, create_edge_splits
    
    # Prepare data
    G = load_gml_graph(config.DEFAULT_NETWORK)
    node_features, edge_features = generate_all_features(G)
    data = create_pyg_data(G, node_features, edge_features)
    data = create_edge_splits(data)
    
    # Train model
    model, trainer, history = train_model(data, model_type='gnn', gnn_type='sage')
