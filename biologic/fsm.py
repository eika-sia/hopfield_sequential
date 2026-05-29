"""Finite-state machine representation and generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class FiniteStateMachine:
    """A finite-state transition table with optional Moore-style outputs."""

    num_states: int
    num_inputs: int
    transition_table: np.ndarray
    output_table: np.ndarray | None = None

    def __post_init__(self) -> None:
        table = np.asarray(self.transition_table, dtype=int)
        if table.shape != (self.num_states, self.num_inputs):
            raise ValueError(
                "transition_table must have shape (num_states, num_inputs)"
            )
        if np.any(table < 0) or np.any(table >= self.num_states):
            raise ValueError("transition_table contains invalid state indices")
        object.__setattr__(self, "transition_table", table)

        if self.output_table is None:
            output = np.arange(self.num_states, dtype=int)
        else:
            output = np.asarray(self.output_table, dtype=int)
            if output.shape != (self.num_states,):
                raise ValueError("output_table must have shape (num_states,)")
        object.__setattr__(self, "output_table", output)

    @classmethod
    def random(
        cls,
        num_states: int,
        num_inputs: int,
        rng: np.random.Generator,
        num_outputs: int | None = None,
    ) -> "FiniteStateMachine":
        """Generate a random Moore-type FSM."""
        if num_states <= 0:
            raise ValueError("num_states must be positive")
        if num_inputs <= 0:
            raise ValueError("num_inputs must be positive")
        transition_table = rng.integers(0, num_states, size=(num_states, num_inputs))
        if num_outputs is None:
            output_table = np.arange(num_states, dtype=int)
        else:
            if num_outputs <= 0:
                raise ValueError("num_outputs must be positive")
            output_table = rng.integers(0, num_outputs, size=num_states)
        return cls(num_states, num_inputs, transition_table, output_table)

    def next_state(self, state_idx: int, input_idx: int) -> int:
        """Return ``transition_table[state_idx, input_idx]``."""
        return int(self.transition_table[state_idx, input_idx])

    def all_transitions(self) -> Iterator[tuple[int, int, int]]:
        """Yield ``(state_idx, input_idx, next_state_idx)`` for every transition."""
        for state_idx in range(self.num_states):
            for input_idx in range(self.num_inputs):
                yield state_idx, input_idx, self.next_state(state_idx, input_idx)
