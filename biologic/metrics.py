"""Metrics and perturbation helpers for BioLogic simulations."""

from __future__ import annotations

import numpy as np

from biologic.fsm import FiniteStateMachine


def hamming_distance(x: np.ndarray, y: np.ndarray) -> int:
    """Return the number of differing coordinates."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    return int(np.sum(x != y))


def normalized_overlap(x: np.ndarray, y: np.ndarray) -> float:
    """Return mean bipolar overlap ``mean(x * y)``."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    return float(np.mean(x * y))


def corrupt_vector(
    x: np.ndarray,
    flip_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Flip exactly ``round(flip_fraction * len(x))`` coordinates."""
    if not 0.0 <= flip_fraction <= 1.0:
        raise ValueError("flip_fraction must be in [0, 1]")
    x = np.asarray(x).copy()
    n_flip = int(round(flip_fraction * len(x)))
    if n_flip == 0:
        return x
    indices = rng.choice(len(x), size=n_flip, replace=False)
    x[indices] *= -1
    return x


def transition_accuracy(
    fsm: FiniteStateMachine,
    transition,
    register,
    state_codebook: np.ndarray,
    input_codebook: np.ndarray,
) -> float:
    """Evaluate all FSM transitions after transition proposal and cleanup."""
    correct = 0
    total = 0
    for state_idx, input_idx, target_idx in fsm.all_transitions():
        proposal = transition.propose(state_codebook[state_idx], input_codebook[input_idx])
        recovered_idx, _ = register.cleanup(proposal)
        correct += int(recovered_idx == target_idx)
        total += 1
    return correct / total if total else 0.0


def recovery_accuracy(
    register,
    codebook: np.ndarray,
    flip_fraction: float,
    trials_per_state: int,
    rng: np.random.Generator,
) -> float:
    """Corrupt each state multiple times and test cleanup."""
    if trials_per_state <= 0:
        raise ValueError("trials_per_state must be positive")
    correct = 0
    total = 0
    for state_idx, state_vec in enumerate(codebook):
        for _ in range(trials_per_state):
            corrupted = corrupt_vector(state_vec, flip_fraction, rng)
            recovered_idx, _ = register.cleanup(corrupted)
            correct += int(recovered_idx == state_idx)
            total += 1
    return correct / total if total else 0.0


def sparse_transition_accuracy(
    fsm,
    transition,
    register,
    state_codebook,
    input_codebook,
) -> float:
    """Evaluate a sparse transition object with cleanup."""
    return transition_accuracy(fsm, transition, register, state_codebook, input_codebook)
