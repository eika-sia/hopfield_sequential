import inspect

import numpy as np

from biologic.encodings import random_input_codebook, random_state_codebook
from biologic.fsm import FiniteStateMachine
from biologic.learning import (
    EligibilityTraceTransitionLearner,
    LearnedAssociativeTransition,
    StateInputPairEncoder,
)
from biologic.register import NearestAttractorRegister
from experiments.exp06_learned_transitions import (
    _evaluate_writer,
    _learner_examples,
    _seen_pairs,
    _training_examples,
    generate_transition_demonstrations,
)


def test_exact_pair_encoder_shape() -> None:
    encoder = StateInputPairEncoder(num_states=3, num_inputs=2)
    phi = encoder.encode_indices(1, 0)
    assert encoder.feature_dim == 6
    assert phi.shape == (6,)
    assert int(np.sum(phi)) == 1


def test_exact_pair_unique() -> None:
    encoder = StateInputPairEncoder(num_states=3, num_inputs=2)
    active = {
        encoder.active_indices(state_idx, input_idx)
        for state_idx in range(3)
        for input_idx in range(2)
    }
    assert len(active) == 6


def test_learned_transition_one_example() -> None:
    rng = np.random.default_rng(0)
    target = np.array([1, -1, 1, -1], dtype=int)
    writer = LearnedAssociativeTransition(
        state_dim=4,
        feature_dim=1,
        rng=rng,
        update_rule="hebbian",
    )
    phi = np.array([1.0])
    writer.update(phi, target)
    assert np.array_equal(writer.propose_features(phi), target)


def test_full_table_one_epoch_nearest() -> None:
    rng = np.random.default_rng(1)
    transition_table = np.array(
        [
            [1, 2],
            [2, 3],
            [3, 0],
            [0, 1],
        ],
        dtype=int,
    )
    fsm = FiniteStateMachine(4, 2, transition_table)
    state_codebook = random_state_codebook(4, 64, rng)
    input_codebook = random_input_codebook(2, 16, rng)
    encoder = StateInputPairEncoder(4, 2)
    writer = LearnedAssociativeTransition(
        state_dim=64,
        feature_dim=encoder.feature_dim,
        rng=rng,
        update_rule="delta",
    )
    learner = EligibilityTraceTransitionLearner(writer)
    demonstrations = generate_transition_demonstrations(
        fsm,
        state_codebook,
        input_codebook,
        mode="full_table",
        rng=rng,
    )
    items = _training_examples(demonstrations, encoder, state_codebook, input_codebook)
    learner.train_epoch(_learner_examples(items), rng=rng, shuffle=False)
    metrics = _evaluate_writer(
        fsm,
        writer,
        encoder,
        NearestAttractorRegister(state_codebook),
        state_codebook,
        input_codebook,
        _seen_pairs(items),
    )
    assert metrics["all_transition_accuracy"] == 1.0


def test_untrained_baseline_not_perfect() -> None:
    rng = np.random.default_rng(2)
    transition_table = np.array(
        [
            [1, 2],
            [2, 3],
            [3, 0],
            [0, 1],
        ],
        dtype=int,
    )
    fsm = FiniteStateMachine(4, 2, transition_table)
    state_codebook = random_state_codebook(4, 64, rng)
    input_codebook = random_input_codebook(2, 16, rng)
    encoder = StateInputPairEncoder(4, 2)
    writer = LearnedAssociativeTransition(
        state_dim=64,
        feature_dim=encoder.feature_dim,
        rng=rng,
    )
    metrics = _evaluate_writer(
        fsm,
        writer,
        encoder,
        NearestAttractorRegister(state_codebook),
        state_codebook,
        input_codebook,
        set(),
    )
    assert metrics["all_transition_accuracy"] < 1.0


def test_hashed_pair_collision_degrades_or_collides() -> None:
    rng = np.random.default_rng(3)
    encoder = StateInputPairEncoder(
        num_states=4,
        num_inputs=3,
        mode="hashed_pair",
        hidden_dim=2,
        rng=rng,
    )
    assert encoder.feature_dim == 2
    assert encoder.collision_count() > 0


def test_no_direct_transition_table_use_in_learner() -> None:
    for cls in [
        StateInputPairEncoder,
        LearnedAssociativeTransition,
        EligibilityTraceTransitionLearner,
    ]:
        signature = inspect.signature(cls)
        assert "fsm" not in signature.parameters
        assert "transition_table" not in signature.parameters

    update_signature = inspect.signature(LearnedAssociativeTransition.update)
    assert "phi" in update_signature.parameters
    assert "target_state_vec" in update_signature.parameters
    assert "fsm" not in update_signature.parameters
