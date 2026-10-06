"""
Self-Organizing Map (SOM) implemented from scratch.

This module provides a dependency-light implementation of Kohonen's
Self-Organizing Map. It only relies on NumPy for the numerical work, so it can
be used as a drop-in replacement for third-party libraries such as MiniSom.

Author: (your name)
Reference: Kohonen, T. (1982). Self-organized formation of topologically correct
feature maps. Biological Cybernetics, 43(1), 59-69.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SOM"]


class SOM:
    """A 2D Self-Organizing Map (Kohonen network).

    Parameters
    ----------
    grid : tuple[int, int], default=(10, 10)
        Grid size of the map as ``(rows, cols)``.
    input_dim : int
        Dimensionality of the input vectors.
    learning_rate : float, default=0.5
        Initial learning rate ``alpha_0``.
    sigma : float or None, default=None
        Initial neighborhood radius ``sigma_0`` (in grid units). When ``None``
        it defaults to half of the largest grid dimension.
    n_iter : int, default=1000
        Number of training iterations.
    seed : int or None, default=42
        Random seed used for weight initialization, for reproducibility.
    """

    def __init__(
        self,
        grid: tuple[int, int] = (10, 10),
        input_dim: int = 2,
        learning_rate: float = 0.5,
        sigma: float | None = None,
        n_iter: int = 1000,
        seed: int | None = 42,
    ) -> None:
        self.rows, self.cols = grid
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.n_iter = n_iter

        # Default initial radius = half the largest grid dimension.
        self.sigma = sigma if sigma is not None else max(self.rows, self.cols) / 2.0

        # Time constant used by the exponential decay of sigma.
        # Guard against sigma <= 1 (log would be <= 0).
        self.time_constant = (
            self.n_iter / np.log(self.sigma) if self.sigma > 1 else 1.0
        )

        self._rng = np.random.default_rng(seed)
        # Weight vectors: shape (rows, cols, input_dim), initialized in [0, 1].
        self.weights = self._rng.random((self.rows, self.cols, input_dim))

        # Precompute the grid coordinates of every neuron (used for the
        # neighborhood distance in grid space).
        self._grid_coords = np.array(
            [[[i, j] for j in range(self.cols)] for i in range(self.rows)],
            dtype=float,
        )

    # ------------------------------------------------------------------ #
    # Decay functions
    # ------------------------------------------------------------------ #
    def _decay_learning_rate(self, t: int) -> float:
        """Exponential decay of the learning rate: ``alpha_0 * exp(-t / n_iter)``."""
        return self.learning_rate * np.exp(-t / self.n_iter)

    def _decay_radius(self, t: int) -> float:
        """Exponential decay of the neighborhood radius using a time constant."""
        return self.sigma * np.exp(-t / self.time_constant)

    def _influence(self, dist: np.ndarray, radius: float) -> np.ndarray:
        """Gaussian neighborhood function ``h(d) = exp(-d^2 / (2 * radius^2))``."""
        return np.exp(-(dist ** 2) / (2.0 * radius ** 2))

    # ------------------------------------------------------------------ #
    # Best Matching Unit
    # ------------------------------------------------------------------ #
    def find_bmu(self, x: np.ndarray) -> tuple[int, int]:
        """Return the ``(row, col)`` coordinates of the Best Matching Unit for ``x``."""
        diff = self.weights - x                      # (rows, cols, input_dim)
        dist = np.sqrt(np.sum(diff ** 2, axis=2))    # (rows, cols)
        return np.unravel_index(np.argmin(dist), dist.shape)

    def winner(self, x: np.ndarray) -> tuple[int, int]:
        """Alias for :meth:`find_bmu` (MiniSom-compatible name)."""
        return self.find_bmu(x)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, data: np.ndarray, verbose: bool = True) -> "SOM":
        """Train the map on ``data`` of shape ``(n_samples, input_dim)``."""
        data = np.asarray(data, dtype=float)
        n_samples = data.shape[0]
        report_every = max(1, self.n_iter // 10)

        for t in range(self.n_iter):
            x = data[self._rng.integers(0, n_samples)]
            bmu = self.find_bmu(x)

            alpha = self._decay_learning_rate(t)
            radius = self._decay_radius(t)

            # Distance of every neuron to the BMU, in grid coordinates.
            bmu_coord = np.array(bmu, dtype=float)
            grid_dist = np.sqrt(
                np.sum((self._grid_coords - bmu_coord) ** 2, axis=2)
            )

            # Gaussian neighborhood influence.
            influence = self._influence(grid_dist, radius)

            # Update rule: w += alpha * h(d) * (x - w)
            self.weights += (alpha * influence)[..., np.newaxis] * (x - self.weights)

            if verbose and (t + 1) % report_every == 0:
                print(
                    f"Iteration {t + 1}/{self.n_iter}  "
                    f"alpha={alpha:.4f}  radius={radius:.4f}"
                )

        if verbose:
            print("Training complete.")
        return self

    # ------------------------------------------------------------------ #
    # Scoring / visualization helpers
    # ------------------------------------------------------------------ #
    def winner_distance(self, data: np.ndarray) -> np.ndarray:
        """Distance of each sample to its BMU (used as an outlier score)."""
        data = np.asarray(data, dtype=float)
        scores = np.zeros(data.shape[0])
        for i, x in enumerate(data):
            bmu = self.find_bmu(x)
            scores[i] = np.linalg.norm(x - self.weights[bmu])
        return scores

    def distance_map(self) -> np.ndarray:
        """U-Matrix: mean Euclidean distance of each neuron to its grid neighbors."""
        umatrix = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                neighbors = []
                if i > 0:
                    neighbors.append(self.weights[i - 1, j])
                if i < self.rows - 1:
                    neighbors.append(self.weights[i + 1, j])
                if j > 0:
                    neighbors.append(self.weights[i, j - 1])
                if j < self.cols - 1:
                    neighbors.append(self.weights[i, j + 1])
                umatrix[i, j] = np.mean(
                    [np.linalg.norm(self.weights[i, j] - n) for n in neighbors]
                )
        return umatrix

    def win_map(self, data: np.ndarray) -> dict[tuple[int, int], list[np.ndarray]]:
        """Map each neuron coordinate to the list of samples that hit it."""
        winmap: dict[tuple[int, int], list[np.ndarray]] = {}
        for x in np.asarray(data, dtype=float):
            winmap.setdefault(self.find_bmu(x), []).append(x)
        return winmap

    def quantization_error(self, data: np.ndarray) -> float:
        """Average distance between each sample and its BMU (map quality)."""
        return float(np.mean(self.winner_distance(data)))
