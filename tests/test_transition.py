import numpy as np

from biologic.encodings import random_input_codebook, random_state_codebook
from biologic.fsm import FiniteStateMachine
from biologic.metrics import transition_accuracy
from biologic.register import NearestAttractorRegister
from biologic.transition import ExactCoordinateTransition, SparseCoordinateTransition


def _fixture() -> tuple[
    np.random.Generator,
    FiniteStateMachine,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(0)
    fsm = FiniteStateMachine.random(6, 3, rng)
    state_codebook = random_state_codebook(6, 64, rng)
    input_codebook = random_input_codebook(3, 16, rng)
    return rng, fsm, state_codebook, input_codebook


def test_exact_transition_targets_all_fsm_transitions() -> None:
    _, fsm, state_codebook, input_codebook = _fixture()
    transition = ExactCoordinateTransition(fsm, state_codebook, input_codebook)
    for state_idx, input_idx, target_idx in fsm.all_transitions():
        proposal = transition.propose(
            state_codebook[state_idx], input_codebook[input_idx]
        )
        assert np.array_equal(proposal, state_codebook[target_idx])


def test_transition_accuracy_is_one_with_nearest_register() -> None:
    _, fsm, state_codebook, input_codebook = _fixture()
    register = NearestAttractorRegister(state_codebook)
    transition = ExactCoordinateTransition(fsm, state_codebook, input_codebook)
    assert (
        transition_accuracy(fsm, transition, register, state_codebook, input_codebook)
        == 1.0
    )


def test_sparse_full_write_equals_exact_transition() -> None:
    rng, fsm, state_codebook, input_codebook = _fixture()
    exact = ExactCoordinateTransition(fsm, state_codebook, input_codebook)
    sparse = SparseCoordinateTransition(
        fsm,
        state_codebook,
        input_codebook,
        write_fraction=1.0,
        rng=rng,
        mode="random_noise",
        fixed_masks=True,
    )
    for state_idx, input_idx, _target_idx in fsm.all_transitions():
        assert np.array_equal(
            sparse.propose(state_codebook[state_idx], input_codebook[input_idx]),
            exact.propose(state_codebook[state_idx], input_codebook[input_idx]),
        )
