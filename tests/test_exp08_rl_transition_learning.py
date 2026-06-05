import inspect

import numpy as np

from biologic.grammars import even_ones_task
from biologic.rl_transition import RLTransitionAgent, update_transition_writer_rl
from experiments.exp08_rl_transition_learning import (
    ALL_COLUMNS,
    Condition,
    DataBundle,
    _run_condition_with_bundle,
    _with_row_type,
)


def test_rl_update_does_not_accept_target_state_vec() -> None:
    signature = inspect.signature(update_transition_writer_rl)
    assert "target_state_vec" not in signature.parameters
    assert "true_next_state_vec" not in signature.parameters


def test_rl_update_reinforces_executed_state_only() -> None:
    weights = np.zeros((3, 4), dtype=float)
    executed = np.array([1.0, -1.0, 1.0])
    trace = np.array([0.0, 1.0, 0.5, 0.0])

    update_transition_writer_rl(weights, executed, trace, delta=2.0, eta=0.1)

    expected = 0.2 * np.outer(executed, trace)
    np.testing.assert_allclose(weights, expected)


def test_rl_sparse_update_masks_rows() -> None:
    weights = np.zeros((3, 2), dtype=float)
    executed = np.array([1.0, -1.0, 1.0])
    trace = np.array([1.0, 1.0])
    mask = np.array([True, False, True])

    update_transition_writer_rl(
        weights,
        executed,
        trace,
        delta=1.0,
        eta=0.5,
        mask=mask,
    )

    assert np.all(weights[1] == 0.0)
    np.testing.assert_allclose(weights[0], [0.5, 0.5])
    np.testing.assert_allclose(weights[2], [0.5, 0.5])


def test_rl_agent_evaluate_returns_metrics() -> None:
    rng = np.random.default_rng(1)
    task = even_ones_task()
    agent = RLTransitionAgent(task=task, state_dim=32, input_dim=8, rng=rng)
    strings = [[], list("0"), list("1"), list("11")]

    metrics = agent.evaluate(strings, train_strings=strings)

    assert 0.0 <= metrics.string_accuracy <= 1.0
    assert 0.0 <= metrics.transition_accuracy <= 1.0


def test_exp08_tiny_condition_produces_requested_rows() -> None:
    condition = Condition(
        seed=0,
        grammar="even_ones",
        condition="state_shaped_rl",
        cleanup="nearest",
        feature_type="exact_pair",
        hidden_dim=None,
        state_dim=32,
        input_dim=8,
        episodes=4,
        eval_every=2,
        train_max_len=4,
        test_max_len=6,
        test_strings=4,
        eta=0.01,
        alpha=0.01,
        gamma=0.95,
        lambda_trace=0.8,
        noise_start=0.1,
        noise_end=0.0,
        write_fraction=1.0,
    )
    task = even_ones_task()
    bundle = DataBundle(
        task=task,
        train_strings=[list("0"), list("1"), list("11"), list("00")],
        test_strings=[[], list("1"), list("11"), list("101")],
    )

    result = _run_condition_with_bundle(condition, bundle)

    assert len(result.summary_rows) == 1
    assert len(result.curve_rows) >= 2
    assert len(result.transition_rows) == task.dfa.num_states * len(task.alphabet)


def test_exp08_all_csv_schema_has_row_type_and_all_row_fields() -> None:
    assert ALL_COLUMNS[0] == "row_type"
    for column in [
        "final_test_accuracy",
        "episode",
        "test_accuracy",
        "state_id",
        "symbol",
        "correct",
    ]:
        assert column in ALL_COLUMNS

    row = _with_row_type({"seed": 0, "final_test_accuracy": 1.0}, "summary")
    assert row["row_type"] == "summary"
    assert row["final_test_accuracy"] == 1.0
