import numpy as np

from biologic.encodings import random_state_codebook
from biologic.metrics import corrupt_vector
from biologic.register import HopfieldRegister, NearestAttractorRegister


def test_nearest_register_recovers_exact_vectors() -> None:
    rng = np.random.default_rng(0)
    codebook = random_state_codebook(8, 64, rng)
    register = NearestAttractorRegister(codebook)
    for idx, state in enumerate(codebook):
        recovered_idx, recovered = register.cleanup(state)
        assert recovered_idx == idx
        assert np.array_equal(recovered, state)


def test_zero_corruption_recovers_exactly() -> None:
    rng = np.random.default_rng(1)
    codebook = random_state_codebook(8, 64, rng)
    register = NearestAttractorRegister(codebook)
    for idx, state in enumerate(codebook):
        corrupted = corrupt_vector(state, 0.0, rng)
        recovered_idx, _ = register.cleanup(corrupted)
        assert recovered_idx == idx


def test_hopfield_weights_symmetric_and_zero_diagonal() -> None:
    rng = np.random.default_rng(2)
    codebook = random_state_codebook(6, 32, rng)
    register = HopfieldRegister.from_codebook(codebook)
    assert np.allclose(register.weights, register.weights.T)
    assert np.allclose(np.diag(register.weights), 0.0)


def test_hopfield_energy_does_not_increase_under_async_update() -> None:
    rng = np.random.default_rng(3)
    codebook = random_state_codebook(4, 32, rng)
    register = HopfieldRegister.from_codebook(codebook)
    x = corrupt_vector(codebook[0], 0.25, rng)
    before = register.energy(x)
    after_state = register.step(x, synchronous=False)
    after = register.energy(after_state)
    assert after <= before + 1e-9
