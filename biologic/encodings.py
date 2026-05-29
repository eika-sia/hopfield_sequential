"""Encoding helpers for bipolar state and input vectors."""

from __future__ import annotations

import numpy as np


def sign(x: np.ndarray) -> np.ndarray:
    """Return bipolar sign: +1 for x >= 0, -1 otherwise."""
    return np.where(np.asarray(x) >= 0, 1, -1).astype(int)


def random_bipolar(
    shape: int | tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    """Generate random {-1, +1} array."""
    return rng.choice(np.array([-1, 1], dtype=int), size=shape)


def _random_unique_codebook(
    num_codes: int, dim: int, rng: np.random.Generator
) -> np.ndarray:
    if dim < 63 and num_codes > 2**dim:
        raise ValueError("num_codes exceeds the number of unique bipolar vectors")

    rows: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    batch_size = max(64, num_codes)
    while len(rows) < num_codes:
        batch = random_bipolar((batch_size, dim), rng)
        for row in batch:
            key = tuple(int(v) for v in row)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row.copy())
            if len(rows) == num_codes:
                break
    return np.stack(rows, axis=0)


def random_state_codebook(
    num_states: int,
    dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an array of shape (num_states, dim) with entries in {-1, +1}."""
    if num_states <= 0:
        raise ValueError("num_states must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")
    return _random_unique_codebook(num_states, dim, rng)


def random_input_codebook(
    num_inputs: int,
    dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an array of shape (num_inputs, dim) with entries in {-1, +1}."""
    if num_inputs <= 0:
        raise ValueError("num_inputs must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")
    return _random_unique_codebook(num_inputs, dim, rng)


def pairwise_overlap(codebook: np.ndarray) -> np.ndarray:
    """Return normalized bipolar overlap matrix for the rows of ``codebook``."""
    codebook = np.asarray(codebook)
    if codebook.ndim != 2:
        raise ValueError("codebook must be a 2D array")
    return (codebook @ codebook.T) / codebook.shape[1]


def mean_abs_offdiag_overlap(codebook: np.ndarray) -> float:
    """Return mean absolute normalized overlap excluding the diagonal."""
    overlaps = pairwise_overlap(codebook)
    n = overlaps.shape[0]
    if n <= 1:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(np.abs(overlaps[mask])))
