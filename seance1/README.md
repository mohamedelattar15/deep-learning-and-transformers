# SOM Credit Card Fraud Detection (from scratch)

Detect potential fraud in credit card applications using a **Self-Organizing Map
(SOM)** implemented **from scratch** (no MiniSom), then flag the most suspicious
customers and report whether each card was **approved** or **not approved**.

## Repository structure

```
seance1/
├── som/                          # from-scratch SOM package
│   ├── __init__.py
│   └── som.py                    # the SOM algorithm (NumPy only)
├── som-credit-card-fraud.ipynb   # notebook: full credit-card fraud workflow
├── requirements.txt
└── README.md
```

## The algorithm (`som/som.py`)

A dependency-light implementation of Kohonen's SOM:

- 2D grid of neurons, each with a weight vector of the input dimension
- **BMU** search with Euclidean distance (`find_bmu` / `winner`)
- **Gaussian neighborhood** influence `h(d) = exp(-d² / (2σ²))`
- **Exponential decay** of the learning rate `α(t)` and radius `σ(t)`
- Update rule `w += α(t) · h(d) · (x − w)`
- Extras: `distance_map()` (U-Matrix), `winner_distance()` (outlier score),
  `win_map()`, `quantization_error()`

## Usage

```python
from som import SOM

som = SOM(grid=(10, 10), input_dim=X.shape[1], learning_rate=0.5, n_iter=1000)
som.train(X_scaled)

umatrix = som.distance_map()          # U-Matrix
scores = som.winner_distance(X_scaled)  # outlier score per sample
```

## Notebook workflow

1. Import libraries
2. Load & explore `Credit_Card_Applications.csv`
3. Prepare features (scale to `[0, 1]` with `MinMaxScaler`)
4. Import the from-scratch `SOM`
5. Train the map
6. Plot the U-Matrix
7. Score customers (distance to BMU) and flag fraud candidates
8. Overlay approved / not approved applications on the map
9. Rank and list the most likely fraudulent customers with their status

## Setup

```bash
pip install -r requirements.txt
```

Then open `som-credit-card-fraud.ipynb` and run all cells.
