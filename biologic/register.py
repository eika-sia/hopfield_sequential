"""State-register cleanup implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biologic.encodings import sign


@dataclass(frozen=True)
class NearestAttractorRegister:
    """Idealized cleanup by highest overlap with a stored codebook state."""

    codebook: np.ndarray

    def __post_init__(self) -> None:
        codebook = np.asarray(self.codebook, dtype=int)
        if codebook.ndim != 2:
            raise ValueError("codebook must be 2D")
        object.__setattr__(self, "codebook", codebook)

    def overlap_scores(self, x: np.ndarray) -> np.ndarray:
        """Return overlap score with each stored state."""
        x = np.asarray(x, dtype=int)
        return (self.codebook @ x) / self.codebook.shape[1]

    def cleanup(self, x: np.ndarray) -> tuple[int, np.ndarray]:
        """Return the closest stored state by normalized bipolar overlap."""
        scores = self.overlap_scores(x)
        state_idx = int(np.argmax(scores))
        return state_idx, self.codebook[state_idx].copy()


@dataclass(frozen=True)
class HopfieldRegister:
    """Classical Hebbian Hopfield state register."""

    codebook: np.ndarray
    weights: np.ndarray
    thresholds: np.ndarray

    @classmethod
    def from_codebook(cls, codebook: np.ndarray) -> "HopfieldRegister":
        """Build Hebbian weights with zero diagonal and zero thresholds."""
        codebook = np.asarray(codebook, dtype=int)
        if codebook.ndim != 2:
            raise ValueError("codebook must be 2D")
        dim = codebook.shape[1]
        weights = (codebook.T @ codebook) / dim
        np.fill_diagonal(weights, 0.0)
        thresholds = np.zeros(dim)
        return cls(codebook=codebook, weights=weights, thresholds=thresholds)

    def __post_init__(self) -> None:
        codebook = np.asarray(self.codebook, dtype=int)
        weights = np.asarray(self.weights, dtype=float)
        thresholds = np.asarray(self.thresholds, dtype=float)
        dim = codebook.shape[1]
        if weights.shape != (dim, dim):
            raise ValueError("weights must have shape (dim, dim)")
        if thresholds.shape != (dim,):
            raise ValueError("thresholds must have shape (dim,)")
        object.__setattr__(self, "codebook", codebook)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "thresholds", thresholds)

    def step(self, x: np.ndarray, synchronous: bool = False) -> np.ndarray:
        """Run one Hopfield update."""
        x_next = np.asarray(x, dtype=int).copy()
        if synchronous:
            return sign(self.weights @ x_next - self.thresholds)

        for idx in range(len(x_next)):
            field = float(self.weights[idx] @ x_next - self.thresholds[idx])
            x_next[idx] = 1 if field >= 0 else -1
        return x_next

    def run(
        self,
        x: np.ndarray,
        max_steps: int = 50,
        synchronous: bool = False,
    ) -> np.ndarray:
        """Run until convergence or ``max_steps``."""
        current = np.asarray(x, dtype=int).copy()
        for _ in range(max_steps):
            nxt = self.step(current, synchronous=synchronous)
            if np.array_equal(nxt, current):
                return nxt
            current = nxt
        return current

    def cleanup(self, x: np.ndarray) -> tuple[int, np.ndarray]:
        """Run Hopfield dynamics, then return nearest stored codebook state."""
        settled = self.run(x)
        scores = (self.codebook @ settled) / self.codebook.shape[1]
        state_idx = int(np.argmax(scores))
        return state_idx, self.codebook[state_idx].copy()

    def energy(self, x: np.ndarray) -> float:
        """Return Hopfield energy ``-0.5 x^T W x + theta^T x``."""
        x = np.asarray(x, dtype=float)
        return float(-0.5 * x @ self.weights @ x + self.thresholds @ x)
