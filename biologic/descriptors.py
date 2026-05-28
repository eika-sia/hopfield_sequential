"""Toy descriptor/payload protocol for operation-content separation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

STORE = "STORE"
COMPARE = "COMPARE"
ROUTE_A = "ROUTE_A"
ROUTE_B = "ROUTE_B"
START = "START"
END = "END"

IDLE = "IDLE"
EXPECT_PAYLOAD_STORE = "EXPECT_PAYLOAD_STORE"
EXPECT_PAYLOAD_COMPARE = "EXPECT_PAYLOAD_COMPARE"
STORED = "STORED"
COMPARE_TRUE = "COMPARE_TRUE"
COMPARE_FALSE = "COMPARE_FALSE"
ERROR = "ERROR"


@dataclass
class DescriptorPayloadMachine:
    """Transparent finite-state protocol separating descriptors from payloads."""

    state: str = IDLE
    stored_payload: np.ndarray | None = field(default=None, repr=False)

    def reset(self) -> None:
        """Reset to IDLE while retaining no stored payload."""
        self.state = IDLE
        self.stored_payload = None

    def feed_descriptor(self, descriptor: str) -> str:
        """Feed a descriptor token and return the new state name."""
        if descriptor == STORE:
            self.state = EXPECT_PAYLOAD_STORE
        elif descriptor == COMPARE:
            self.state = EXPECT_PAYLOAD_COMPARE
        else:
            self.state = ERROR
        return self.state

    def feed_payload(self, payload: np.ndarray) -> str:
        """Feed a payload vector and return the new state name."""
        payload = np.asarray(payload, dtype=int).copy()
        if self.state == EXPECT_PAYLOAD_STORE:
            self.stored_payload = payload
            self.state = STORED
        elif self.state == EXPECT_PAYLOAD_COMPARE:
            if self.stored_payload is not None and np.array_equal(payload, self.stored_payload):
                self.state = COMPARE_TRUE
            else:
                self.state = COMPARE_FALSE
        else:
            self.state = ERROR
        return self.state

    def run_sequence(self, sequence: list[tuple[str, Any]]) -> list[str]:
        """Run descriptor/payload tokens and return states after each input."""
        states = []
        for kind, value in sequence:
            if kind == "descriptor":
                states.append(self.feed_descriptor(value))
            elif kind == "payload":
                states.append(self.feed_payload(value))
            else:
                self.state = ERROR
                states.append(self.state)
        return states
