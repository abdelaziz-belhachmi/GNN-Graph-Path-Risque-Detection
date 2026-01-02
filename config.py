"""
Configuration file for the network risk path detection project.
Contains all hyperparameters and settings.
"""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "archive")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Dataset settings
DEFAULT_NETWORK = "Abilene"  # Default network to use
RANDOM_SEED = 42

# Synthetic feature generation parameters
NODE_LOAD_RANGE = (0.1, 0.95)  # Min and max node load (0-1)
EDGE_LATENCY_RANGE = (1, 100)  # Latency in ms
EDGE_BANDWIDTH_RANGE = (100, 10000)  # Bandwidth in Mbps
EDGE_UTILIZATION_RANGE = (0.1, 0.95)  # Utilization rate (0-1)
EDGE_LOSS_RATE_RANGE = (0.0, 0.05)  # Packet loss rate (0-1)

# Risk calculation weights (for synthetic labels)
RISK_WEIGHTS = {
    "latency": 0.25,
    "utilization": 0.35,
    "loss_rate": 0.20,
    "node_load": 0.20,  # Average of source and target node loads
}

# GNN Model hyperparameters
NODE_FEATURE_DIM = 3  # load, degree_centrality, betweenness_centrality
EDGE_FEATURE_DIM = 4  # latency, bandwidth, utilization, loss_rate
HIDDEN_DIM = 64
NUM_GNN_LAYERS = 3
DROPOUT = 0.2

# Training hyperparameters
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 200
BATCH_SIZE = 1  # Full graph training
EARLY_STOPPING_PATIENCE = 20
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Path analysis settings
TOP_K_CRITICAL_PATHS = 10
PATH_MAX_LENGTH = 10

# Visualization settings
FIGURE_SIZE = (14, 10)
NODE_SIZE_RANGE = (300, 1500)
EDGE_WIDTH_RANGE = (1, 5)
RISK_COLORMAP = "RdYlGn_r"  # Red = high risk, Green = low risk
