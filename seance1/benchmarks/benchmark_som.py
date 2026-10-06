"""
Benchmark for the from-scratch SOM module (som/som.py).

Measures:
  1. Training time vs. grid size
  2. Training time vs. number of iterations
  3. Training time vs. dataset size
  4. Prediction / scoring throughput (find_bmu)
  5. Map quality (quantization error)

Run:  python benchmarks/benchmark_som.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# Make the `som` package importable when run from the repo root or elsewhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from som import SOM  # noqa: E402


def _time(fn, repeat: int = 3) -> float:
    """Return the best (min) wall-clock time of `repeat` runs, in seconds."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_grid_size():
    print("\n[1] Training time vs. grid size (1000 iter, 690 samples, dim=14)")
    print(f"{'grid':>10} | {'neurons':>8} | {'time (s)':>10} | {'ms/iter':>10}")
    print("-" * 48)
    rng = np.random.default_rng(0)
    data = rng.random((690, 14))
    for g in [5, 10, 20, 30]:
        som = SOM(grid=(g, g), input_dim=14, n_iter=1000, seed=1)
        t = _time(lambda: som.train(data, verbose=False))
        print(f"{g}x{g:>6} | {g*g:>8} | {t:>10.3f} | {t/1000*1000:>10.3f}")


def bench_iterations():
    print("\n[2] Training time vs. number of iterations (10x10, 690 samples)")
    print(f"{'n_iter':>10} | {'time (s)':>10} | {'ms/iter':>10}")
    print("-" * 38)
    rng = np.random.default_rng(0)
    data = rng.random((690, 14))
    for n in [100, 500, 1000, 5000]:
        som = SOM(grid=(10, 10), input_dim=14, n_iter=n, seed=1)
        t = _time(lambda: som.train(data, verbose=False))
        print(f"{n:>10} | {t:>10.3f} | {t/n*1000:>10.3f}")


def bench_dataset_size():
    print("\n[3] Training time vs. dataset size (10x10, 1000 iter)")
    print(f"{'samples':>10} | {'time (s)':>10}")
    print("-" * 26)
    rng = np.random.default_rng(0)
    for n in [100, 690, 5000, 20000]:
        data = rng.random((n, 14))
        som = SOM(grid=(10, 10), input_dim=14, n_iter=1000, seed=1)
        t = _time(lambda: som.train(data, verbose=False))
        print(f"{n:>10} | {t:>10.3f}")


def bench_prediction():
    print("\n[4] Prediction throughput: find_bmu on 690 samples (10x10)")
    rng = np.random.default_rng(0)
    data = rng.random((690, 14))
    som = SOM(grid=(10, 10), input_dim=14, n_iter=100, seed=1).train(data, verbose=False)

    t = _time(lambda: [som.find_bmu(x) for x in data], repeat=5)
    print(f"    {t*1000:.2f} ms for 690 samples  ->  {690/t:,.0f} samples/s")

    t = _time(lambda: som.winner_distance(data), repeat=5)
    print(f"    winner_distance: {t*1000:.2f} ms for 690 samples")

    t = _time(lambda: som.distance_map(), repeat=5)
    print(f"    distance_map (U-Matrix): {t*1000:.2f} ms")


def bench_quality():
    print("\n[5] Map quality (quantization error, lower is better)")
    print(f"{'n_iter':>10} | {'quant. error':>14}")
    print("-" * 28)
    rng = np.random.default_rng(0)
    data = rng.random((690, 14))
    for n in [100, 500, 1000, 5000]:
        som = SOM(grid=(10, 10), input_dim=14, n_iter=n, seed=1)
        som.train(data, verbose=False)
        qe = som.quantization_error(data)
        print(f"{n:>10} | {qe:>14.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("SOM from-scratch benchmark")
    print("=" * 60)
    bench_grid_size()
    bench_iterations()
    bench_dataset_size()
    bench_prediction()
    bench_quality()
    print("\nDone.")
