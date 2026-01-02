# SUJET 6 - Detection de chemins a risque (Graph + GNN)

## Project Overview

This project implements a Graph Neural Network (GNN) based system for detecting critical and risky paths in network topologies. It uses real network data from the Internet Topology Zoo and synthetic performance metrics to train a GNN model that learns to predict link risk scores.

## Objectives

- Identify critical paths in network infrastructure
- Exploit risk propagation through graph structure
- Use GNN to score network links based on multiple features
- Compare GNN predictions with traditional static metrics

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Execution Flow](#execution-flow)
4. [Algorithms and Methods](#algorithms-and-methods)
5. [Commands Reference](#commands-reference)
6. [Output Files](#output-files)
7. [Configuration](#configuration)

---

## Project Structure

```
Prjt_virtualisation/
|-- main.py                    # Main entry point
|-- config.py                  # Configuration parameters
|-- requirements.txt           # Python dependencies
|-- src/
|   |-- __init__.py           # Package initialization
|   |-- data_loader.py        # GML file loader
|   |-- feature_generator.py  # Synthetic feature generation
|   |-- dataset.py            # PyTorch Geometric dataset
|   |-- models.py             # GNN model architectures
|   |-- train.py              # Training pipeline
|   |-- path_analysis.py      # Critical path detection
|   |-- visualization.py      # Visualization functions
|   |-- utils.py              # Utility functions
|-- notebooks/
|   |-- risk_path_detection.ipynb  # Interactive notebook
|-- archive/                   # Internet Topology Zoo data (GML files)
|-- graphs/                    # Condensed graph files
|-- output/                    # Generated results and visualizations
|-- models/                    # Saved model checkpoints
|-- venv/                      # Python virtual environment
```

---

## Installation

### Step 1: Create Virtual Environment

```powershell
cd D:\Prjt_virtualisation
python -m venv venv
```

### Step 2: Activate Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```powershell
pip install networkx numpy pandas matplotlib seaborn scikit-learn tqdm
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
```

Or install all at once:

```powershell
pip install -r requirements.txt
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
```

---

## Execution Flow

The pipeline executes in 5 main steps:

### Step 1: Load Network Topology

**File:** `src/data_loader.py`

**What it does:**
- Reads GML (Graph Modeling Language) files from the Internet Topology Zoo dataset
- Parses network topology into a NetworkX graph object
- Extracts node metadata (city names, geographic coordinates)
- Ensures graph connectivity by extracting the largest connected component
- Relabels nodes to consecutive integers for tensor compatibility

**Algorithms/Methods:**
- NetworkX `read_gml()` parser
- Connected component extraction using BFS/DFS
- Graph relabeling for node indexing

**Input:** GML file path (e.g., `archive/Abilene.gml`)
**Output:** NetworkX undirected graph object

---

### Step 2: Generate Synthetic Features

**File:** `src/feature_generator.py`

**What it does:**
- Generates realistic synthetic node features:
  - `load`: CPU/memory utilization (0-1), correlated with node degree
  - `degree_centrality`: Normalized degree of each node
  - `betweenness_centrality`: Importance of node in shortest paths
- Generates synthetic edge features:
  - `latency`: Link delay in milliseconds, correlated with geographic distance
  - `bandwidth`: Link capacity in Mbps
  - `utilization`: Current usage rate (0-1)
  - `loss_rate`: Packet loss percentage
- Calculates ground truth risk scores using weighted combination of features

**Algorithms/Methods:**
- Degree centrality: `C_D(v) = deg(v) / (n-1)`
- Betweenness centrality: Counts shortest paths passing through each node
- Haversine distance approximation for geographic coordinates
- Risk score formula:
  ```
  risk = 0.25 * latency_norm + 0.35 * utilization + 0.20 * loss_rate + 0.20 * avg_node_load
  ```

**Input:** NetworkX graph
**Output:** Node features DataFrame, Edge features DataFrame with risk scores

---

### Step 3: Prepare PyTorch Geometric Data

**File:** `src/dataset.py`

**What it does:**
- Converts NetworkX graph to PyTorch Geometric Data object
- Creates bidirectional edge index in COO (Coordinate) format
- Normalizes node and edge features using z-score standardization
- Creates train/validation/test splits for edges (70%/15%/15%)

**Algorithms/Methods:**
- Z-score normalization: `x_norm = (x - mean) / std`
- Stratified random splitting for edge masks
- COO sparse matrix format for edge connectivity

**Input:** NetworkX graph, node features, edge features
**Output:** PyTorch Geometric Data object with train/val/test masks

---

### Step 4: Train GNN Model

**File:** `src/models.py`, `src/train.py`

**What it does:**
- Builds a Graph Neural Network architecture for edge risk prediction
- Trains the model using node features, edge features, and graph structure
- Uses message passing to propagate information between neighboring nodes
- Predicts risk scores for each edge

**Model Architecture:**

```
Input: Node features (3-dim) + Edge features (4-dim)
    |
    v
Node Encoder (Linear: 3 -> 64)
    |
    v
GNN Layers (3x GraphSAGE/GCN/GAT with residual connections)
    - Message passing: aggregate neighbor information
    - Batch normalization
    - ReLU activation
    - Dropout (0.2)
    |
    v
Edge Encoder (MLP: 4 -> 64 -> 64)
    |
    v
Edge Predictor (Concat source + target + edge features)
    - MLP: 192 -> 64 -> 32 -> 1
    - Sigmoid activation (output 0-1)
    |
    v
Output: Risk score per edge
```

**Algorithms/Methods:**
- GraphSAGE (default): Samples and aggregates neighbor features
  ```
  h_v = sigma(W * CONCAT(h_v, AGG({h_u : u in N(v)})))
  ```
- GCN (Graph Convolutional Network): Spectral convolution approximation
  ```
  H = sigma(D^(-1/2) * A * D^(-1/2) * H * W)
  ```
- GAT (Graph Attention Network): Attention-weighted neighbor aggregation
  ```
  h_v = sigma(sum(alpha_vu * W * h_u))
  ```
- Adam optimizer with learning rate 0.001
- MSE loss function for regression
- ReduceLROnPlateau scheduler
- Early stopping with patience of 20 epochs

**Input:** PyTorch Geometric Data object
**Output:** Trained model, risk predictions for all edges

---

### Step 5: Analyze Critical Paths

**File:** `src/path_analysis.py`

**What it does:**
- Identifies paths with highest cumulative risk scores
- Finds safest paths between node pairs using risk-weighted shortest paths
- Detects edges that appear most frequently in critical paths
- Compares GNN predictions with static graph metrics (edge betweenness centrality)

**Algorithms/Methods:**
- Modified Dijkstra algorithm for risk-weighted shortest paths
  - Safest path: minimize sum of edge risks
  - Riskiest path: maximize sum of edge risks (using inverted weights)
- Edge betweenness centrality: fraction of shortest paths passing through edge
- Pearson correlation for comparing GNN vs static metrics
- Top-K path extraction with path risk metrics:
  - `total_risk`: Sum of edge risks along path
  - `max_risk`: Maximum single edge risk
  - `avg_risk`: Average edge risk

**Input:** Graph, edge features, predicted risks
**Output:** Critical paths list, critical edges DataFrame, comparison metrics

---

### Step 6: Generate Visualizations

**File:** `src/visualization.py`

**What it does:**
- Creates network risk map with edges colored by risk score
- Plots training history (loss curves)
- Generates risk score distribution histograms
- Visualizes top critical paths on the network
- Creates comparison charts between GNN and static metrics

**Visualization Types:**
- `risk_map.png`: Network graph with color-coded edges (red=high risk, green=low risk)
- `training_history.png`: Training and validation loss over epochs
- `risk_distribution.png`: Histogram of true vs predicted risk scores
- `critical_paths.png`: Top 4 critical paths highlighted on network
- `comparison_metrics.png`: Scatter plots and correlation analysis

---

## Algorithms and Methods

### Graph Neural Network Architectures

| Architecture | Description | Use Case |
|-------------|-------------|----------|
| GraphSAGE | Samples and aggregates neighbor features | Default, works well on all graphs |
| GCN | Spectral graph convolution | Homogeneous graphs |
| GAT | Attention-weighted aggregation | When edge importance varies |

### Risk Calculation

The synthetic risk score is computed as a weighted sum:

```
risk_score = w1 * latency_normalized + 
             w2 * utilization + 
             w3 * loss_rate_normalized + 
             w4 * average_endpoint_load
```

Default weights: `w1=0.25, w2=0.35, w3=0.20, w4=0.20`

### Path Finding Algorithms

| Algorithm | Purpose |
|-----------|---------|
| Dijkstra (min weight) | Find safest path (minimum total risk) |
| Dijkstra (inverted weight) | Find riskiest path (maximum total risk) |
| BFS | Find shortest path (minimum hops) |

### Centrality Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Degree Centrality | `deg(v) / (n-1)` | Node connectivity |
| Betweenness Centrality | `sum(shortest_paths_through_v) / total_shortest_paths` | Node importance in routing |
| Edge Betweenness | Same concept applied to edges | Link importance in routing |

---

## Commands Reference

### Basic Execution

Run with default network (Abilene):
```powershell
D:\Prjt_virtualisation\venv\Scripts\python.exe D:\Prjt_virtualisation\main.py
```

Run without visualizations (faster):
```powershell
D:\Prjt_virtualisation\venv\Scripts\python.exe D:\Prjt_virtualisation\main.py --no-visualize
```

### Network Selection

Use a different network topology:
```powershell
python main.py --network Geant2012
python main.py --network Cogentco
python main.py --network AttMpls
python main.py --network Cesnet201006
```

List all available networks:
```powershell
python main.py --list-networks
```

### GNN Architecture Selection

Use different GNN architectures:
```powershell
python main.py --gnn-type sage    # GraphSAGE (default)
python main.py --gnn-type gcn     # Graph Convolutional Network
python main.py --gnn-type gat     # Graph Attention Network
```

### Training Parameters

Adjust number of training epochs:
```powershell
python main.py --epochs 100
python main.py --epochs 500
```

Skip training and load existing model:
```powershell
python main.py --skip-training
```

### Combined Options

Full example with multiple options:
```powershell
python main.py --network Geant2012 --gnn-type gat --epochs 300
```

### Jupyter Notebook

Run interactive analysis:
```powershell
jupyter notebook notebooks/risk_path_detection.ipynb
```

---

## Output Files

After execution, the following files are generated in the `output/` directory:

| File | Format | Description |
|------|--------|-------------|
| `edge_predictions.csv` | CSV | Edge features with GNN predicted risk scores |
| `critical_paths.json` | JSON | Top 10 critical paths with risk metrics |
| `critical_edges.csv` | CSV | Most critical edges ranked by frequency |
| `test_metrics.json` | JSON | Model performance (MSE, RMSE, MAE, R2) |
| `risk_map.png` | PNG | Network visualization with risk coloring |
| `training_history.png` | PNG | Loss curves during training |
| `risk_distribution.png` | PNG | Risk score distributions |
| `critical_paths.png` | PNG | Visualization of top critical paths |
| `comparison_metrics.png` | PNG | GNN vs static metrics comparison |

---

## Configuration

All parameters can be modified in `config.py`:

### Paths
```python
DATA_DIR = "archive"           # Topology Zoo GML files
OUTPUT_DIR = "output"          # Results directory
MODELS_DIR = "models"          # Model checkpoints
```

### Feature Generation
```python
NODE_LOAD_RANGE = (0.1, 0.95)
EDGE_LATENCY_RANGE = (1, 100)      # milliseconds
EDGE_BANDWIDTH_RANGE = (100, 10000) # Mbps
EDGE_UTILIZATION_RANGE = (0.1, 0.95)
EDGE_LOSS_RATE_RANGE = (0.0, 0.05)
```

### Model Hyperparameters
```python
NODE_FEATURE_DIM = 3
EDGE_FEATURE_DIM = 4
HIDDEN_DIM = 64
NUM_GNN_LAYERS = 3
DROPOUT = 0.2
```

### Training Parameters
```python
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
```

---

## Dataset

### Internet Topology Zoo

Source: https://topology-zoo.org

Contains 261+ real-world network topologies in GML format including:
- Research networks (Abilene, Geant, Internet2)
- Commercial ISPs (Cogentco, AT&T, Sprint)
- Regional networks (Cesnet, Belnet, Arnes)

Each GML file contains:
- Node definitions with labels and coordinates
- Edge definitions with link types
- Network metadata (country, date, type)

### Synthetic Features

Since Topology Zoo only provides structure, we generate synthetic performance metrics:

**Node Features:**
- Load: Simulated CPU/memory usage
- Centrality: Computed from graph structure

**Edge Features:**
- Latency: Based on geographic distance
- Bandwidth: Random within realistic range
- Utilization: Correlated with node loads
- Loss rate: Correlated with utilization

---

## Performance Metrics

The model is evaluated using:

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| MSE | Mean Squared Error | Lower is better |
| RMSE | Root Mean Squared Error | Lower is better |
| MAE | Mean Absolute Error | Lower is better |
| R2 | Coefficient of Determination | Closer to 1 is better |

Typical results on Abilene network:
- RMSE: 0.12 to 0.18
- R2: 0.3 to 0.6 (varies with random features)
- GNN vs True Risk Correlation: 0.5 to 0.9

---

## Deliverables

As specified in the project requirements:

1. **Risk Path Map**: Generated as `output/risk_map.png`
2. **Critical Path Analysis**: Saved in `output/critical_paths.json`
3. **Model Comparison**: Saved in `output/comparison_metrics.png`
4. **Complete Codebase**: All source files in `src/` directory
5. **Interactive Notebook**: Available in `notebooks/`

---

## References

- Topology Zoo: https://topology-zoo.org
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io
- GraphSAGE Paper: Hamilton et al., "Inductive Representation Learning on Large Graphs"
- GCN Paper: Kipf and Welling, "Semi-Supervised Classification with Graph Convolutional Networks"
- GAT Paper: Velickovic et al., "Graph Attention Networks"