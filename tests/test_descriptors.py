import numpy as np

from biologic.descriptors import (
    COMPARE,
    COMPARE_FALSE,
    COMPARE_TRUE,
    ERROR,
    STORE,
    STORED,
    DescriptorPayloadMachine,
)
from biologic.encodings import random_bipolar


def test_payload_without_descriptor_errors():
    rng = np.random.default_rng(0)
    p1 = random_bipolar(16, rng)
    machine = DescriptorPayloadMachine()
    assert machine.feed_payload(p1) == ERROR


def test_store_payload_stores():
    rng = np.random.default_rng(1)
    p1 = random_bipolar(16, rng)
    machine = DescriptorPayloadMachine()
    assert machine.feed_descriptor(STORE) != ERROR
    assert machine.feed_payload(p1) == STORED


def test_compare_same_payload_after_store_is_true():
    rng = np.random.default_rng(2)
    p1 = random_bipolar(16, rng)
    machine = DescriptorPayloadMachine()
    states = machine.run_sequence(
        [("descriptor", STORE), ("payload", p1), ("descriptor", COMPARE), ("payload", p1)]
    )
    assert states[-1] == COMPARE_TRUE


def test_compare_different_payload_after_store_is_false():
    rng = np.random.default_rng(3)
    p1, p2 = random_bipolar((2, 16), rng)
    machine = DescriptorPayloadMachine()
    states = machine.run_sequence(
        [("descriptor", STORE), ("payload", p1), ("descriptor", COMPARE), ("payload", p2)]
    )
    assert states[-1] == COMPARE_FALSE


def test_same_payload_with_store_vs_compare_has_different_trajectory():
    rng = np.random.default_rng(4)
    p1 = random_bipolar(16, rng)
    store_machine = DescriptorPayloadMachine()
    compare_machine = DescriptorPayloadMachine()
    compare_machine.run_sequence([("descriptor", STORE), ("payload", p1)])

    store_states = store_machine.run_sequence([("descriptor", STORE), ("payload", p1)])
    compare_states = compare_machine.run_sequence([("descriptor", COMPARE), ("payload", p1)])
    assert store_states[-1] == STORED
    assert compare_states[-1] == COMPARE_TRUE
    assert store_states != compare_states
