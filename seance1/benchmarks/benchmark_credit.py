"""
Performance benchmark of the from-scratch SOM on the CREDIT CARD FRAUD case.

This mirrors the notebook `som-credit-card-fraud.ipynb` but measures timings
and quality metrics on the real dataset.

Run:  python benchmarks/benchmark_credit.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from som import SOM  # noqa: E402


def _time(fn, repeat: int = 5) -> float:
    """Best (min) wall-clock time of `repeat` runs, in seconds."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def load_credit_data():
    """Load and scale the credit card dataset (same steps as the notebook)."""
    here = os.path.dirname(__file__)
    csv_path = os.path.join(here, "..", "Credit_Card_Applications.csv")
    dataset = pd.read_csv(csv_path)

    customer_ids = dataset["CustomerID"].values
    y = dataset["Class"].values
    X = dataset.drop(columns=["CustomerID", "Class"]).values.astype(float)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    return dataset, customer_ids, y, X_scaled


def main():
    dataset, customer_ids, y, X_scaled = load_credit_data()
    n_samples, n_features = X_scaled.shape

    print("=" * 64)
    print("SOM performance on the CREDIT CARD FRAUD case")
    print("=" * 64)
    print(f"Dataset shape : {dataset.shape}")
    print(f"Samples       : {n_samples}")
    print(f"Features      : {n_features}")
    print(f"Approved (1)  : {int((y == 1).sum())}")
    print(f"Not appr. (0) : {int((y == 0).sum())}")

    # ---------------------------------------------------------------- #
    # 1. Full pipeline timing (train + score + U-Matrix)
    # ---------------------------------------------------------------- #
    print("\n[1] Full pipeline timing (10x10 grid, 1000 iterations)")
    print("-" * 64)

    t_train = _time(lambda: SOM(grid=(10, 10), input_dim=n_features,
                                n_iter=1000, seed=42).train(X_scaled, verbose=False))
    som = SOM(grid=(10, 10), input_dim=n_features, n_iter=1000, seed=42)
    som.train(X_scaled, verbose=False)

    t_score = _time(lambda: som.winner_distance(X_scaled))
    t_umatrix = _time(lambda: som.distance_map())

    print(f"  Training                : {t_train*1000:8.2f} ms")
    print(f"  Scoring (winner_distance): {t_score*1000:8.2f} ms")
    print(f"  U-Matrix (distance_map) : {t_umatrix*1000:8.2f} ms")
    print(f"  TOTAL                   : {(t_train+t_score+t_umatrix)*1000:8.2f} ms")

    # ---------------------------------------------------------------- #
    # 2. Grid size sensitivity
    # ---------------------------------------------------------------- #
    print("\n[2] Training time vs. grid size (1000 iter)")
    print(f"{'grid':>8} | {'neurons':>8} | {'train (ms)':>11} | {'quant. err':>11}")
    print("-" * 48)
    for g in [5, 10, 15, 20, 30]:
        s = SOM(grid=(g, g), input_dim=n_features, n_iter=1000, seed=42)
        t = _time(lambda: s.train(X_scaled, verbose=False), repeat=3)
        qe = s.quantization_error(X_scaled)
        print(f"{g:>3}x{g:<4} | {g*g:>8} | {t*1000:>11.2f} | {qe:>11.6f}")

    # ---------------------------------------------------------------- #
    # 3. Iterations sensitivity
    # ---------------------------------------------------------------- #
    print("\n[3] Training time vs. iterations (10x10)")
    print(f"{'n_iter':>8} | {'train (ms)':>11} | {'quant. err':>11}")
    print("-" * 36)
    for n in [100, 500, 1000, 2000, 5000]:
        s = SOM(grid=(10, 10), input_dim=n_features, n_iter=n, seed=42)
        t = _time(lambda: s.train(X_scaled, verbose=False), repeat=3)
        qe = s.quantization_error(X_scaled)
        print(f"{n:>8} | {t*1000:>11.2f} | {qe:>11.6f}")

    # ---------------------------------------------------------------- #
    # 4. Fraud detection result + quality
    # ---------------------------------------------------------------- #
    print("\n[4] Fraud detection output (10x10, 1000 iter)")
    print("-" * 64)
    scores = som.winner_distance(X_scaled)
    threshold = scores.mean() + scores.std()
    candidates = scores > threshold

    print(f"  Quantization error      : {som.quantization_error(X_scaled):.6f}")
    print(f"  Threshold (mean+1std)   : {threshold:.4f}")
    print(f"  Fraud candidates        : {int(candidates.sum())} / {n_samples} "
          f"({candidates.mean()*100:.1f}%)")
    print(f"  Among them approved     : {int((y[candidates] == 1).sum())}")
    print(f"  Among them not approved : {int((y[candidates] == 0).sum())}")

    # ---------------------------------------------------------------- #
    # 5. Reproducibility check (same seed -> same result)
    # ---------------------------------------------------------------- #
    print("\n[5] Reproducibility (same seed)")
    print("-" * 64)
    s1 = SOM(grid=(10, 10), input_dim=n_features, n_iter=1000, seed=42)
    s2 = SOM(grid=(10, 10), input_dim=n_features, n_iter=1000, seed=42)
    s1.train(X_scaled, verbose=False)
    s2.train(X_scaled, verbose=False)
    identical = np.allclose(s1.weights, s2.weights)
    print(f"  Two runs with seed=42 identical: {identical}")

    print("\nDone.")


if __name__ == "__main__":
    main()
