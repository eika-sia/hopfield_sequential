"""BioLogic-style coordinate transition mechanisms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biologic.encodings import random_bipolar
from biologic.fsm import FiniteStateMachine


def _nearest_index(codebook: np.ndarray, x: np.ndarray) -> int:
    scores = (codebook @ np.asarray(x)) / codebook.shape[1]
    return int(np.argmax(scores))


@dataclass(frozen=True)
class ExactCoordinateTransition:
    """Transition lookup equivalent to constructed coordinate Boolean maps."""

    fsm: FiniteStateMachine
    state_codebook: np.ndarray
    input_codebook: np.ndarray

    def propose(
        self, current_state_vec: np.ndarray, input_vec: np.ndarray
    ) -> np.ndarray:
        """Identify current state and input by overlap and return target vector."""
        state_idx = _nearest_index(self.state_codebook, current_state_vec)
        input_idx = _nearest_index(self.input_codebook, input_vec)
        target_idx = self.fsm.next_state(state_idx, input_idx)
        return self.state_codebook[target_idx].copy()

    def theoretical_minterm_count(self) -> int:
        """Return the number of state-input minterms."""
        return self.fsm.num_states * self.fsm.num_inputs

    def coordinate_positive_counts(self) -> np.ndarray:
        """Count transitions requiring +1 for each output coordinate."""
        counts = np.zeros(self.state_codebook.shape[1], dtype=int)
        for _, _, target_idx in self.fsm.all_transitions():
            counts += self.state_codebook[target_idx] == 1
        return counts


@dataclass
class SparseCoordinateTransition:
    """Transition proposal that writes only part of the target vector."""

    fsm: FiniteStateMachine
    state_codebook: np.ndarray
    input_codebook: np.ndarray
    write_fraction: float
    rng: np.random.Generator
    mode: str = "keep_current"
    fixed_masks: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.write_fraction <= 1.0:
            raise ValueError("write_fraction must be in [0, 1]")
        if self.mode not in {"keep_current", "random_noise", "zero_unknown"}:
            raise ValueError("unsupported sparse transition mode")
        self.state_codebook = np.asarray(self.state_codebook, dtype=int)
        self.input_codebook = np.asarray(self.input_codebook, dtype=int)
        self._fixed_masks: dict[tuple[int, int], np.ndarray] = {}
        if self.fixed_masks:
            for state_idx in range(self.fsm.num_states):
                for input_idx in range(self.fsm.num_inputs):
                    self._fixed_masks[(state_idx, input_idx)] = self._sample_mask()

    def _sample_mask(self) -> np.ndarray:
        dim = self.state_codebook.shape[1]
        n_write = int(round(self.write_fraction * dim))
        mask = np.zeros(dim, dtype=bool)
        if n_write > 0:
            indices = self.rng.choice(dim, size=n_write, replace=False)
            mask[indices] = True
        return mask

    def propose(
        self, current_state_vec: np.ndarray, input_vec: np.ndarray
    ) -> np.ndarray:
        """Write a fraction of target coordinates and fill the rest by mode."""
        state_idx = _nearest_index(self.state_codebook, current_state_vec)
        input_idx = _nearest_index(self.input_codebook, input_vec)
        target_idx = self.fsm.next_state(state_idx, input_idx)
        target = self.state_codebook[target_idx]
        mask = (
            self._fixed_masks[(state_idx, input_idx)]
            if self.fixed_masks
            else self._sample_mask()
        )

        if self.mode == "random_noise":
            proposal = random_bipolar(target.shape, self.rng)
        else:
            # Bipolar vectors cannot represent unknown coordinates with zero.
            # For zero_unknown, current values are retained as a deterministic proxy.
            proposal = np.asarray(current_state_vec, dtype=int).copy()
        proposal[mask] = target[mask]
        return proposal
