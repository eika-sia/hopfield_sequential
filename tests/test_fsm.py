import numpy as np

from biologic.fsm import FiniteStateMachine


def test_random_fsm_shape_and_valid_indices() -> None:
    rng = np.random.default_rng(0)
    fsm = FiniteStateMachine.random(5, 3, rng)
    assert fsm.transition_table.shape == (5, 3)
    assert np.all(fsm.transition_table >= 0)
    assert np.all(fsm.transition_table < 5)


def test_all_transitions_count() -> None:
    rng = np.random.default_rng(1)
    fsm = FiniteStateMachine.random(4, 2, rng)
    transitions = list(fsm.all_transitions())
    assert len(transitions) == 8
    for state_idx, input_idx, next_state_idx in transitions:
        assert next_state_idx == fsm.next_state(state_idx, input_idx)
