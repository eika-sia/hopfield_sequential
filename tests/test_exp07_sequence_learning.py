import numpy as np

from biologic.grammars import even_ones_task, strings_covering_transitions
from biologic.sequence_eval import BioLogicSequenceLearner, MaskedNearestRegister


def _trained_even_ones() -> tuple[BioLogicSequenceLearner, list[list[str]]]:
    rng = np.random.default_rng(10)
    task = even_ones_task()
    train_strings = strings_covering_transitions(task, (0, 4), rng, max_strings=16)
    learner = BioLogicSequenceLearner(
        task=task,
        state_dim=64,
        input_dim=8,
        rng=rng,
        register_type="nearest",
    )
    learner.fit_from_dfa_traces(train_strings, epochs=1, shuffle=False)
    return learner, train_strings


def test_exact_pair_nearest_learns_even_parity_after_full_transition_exposure() -> None:
    learner, _train_strings = _trained_even_ones()
    strings = [
        [],
        list("0"),
        list("11"),
        list("1010"),
        list("1111"),
        list("101010"),
    ]
    metrics = learner.evaluate(strings)
    assert metrics["string_accuracy"] == 1.0
    assert metrics["extracted_transition_table_accuracy"] == 1.0


def test_autonomous_evaluation_works_without_teacher_forcing() -> None:
    learner, _train_strings = _trained_even_ones()
    trace = learner.predict_trace(list("1011"))
    assert len(trace) == 5
    assert learner.predict_accept(list("1011")) is False
    assert learner.predict_accept(list("1010")) is True


def test_length_generalization_after_full_transition_coverage() -> None:
    learner, _train_strings = _trained_even_ones()
    long_strings = [list("1" * length) for length in range(5, 13)]
    metrics = learner.evaluate(long_strings)
    assert metrics["string_accuracy"] == 1.0
    assert metrics["state_tracking_accuracy"] == 1.0


def test_missing_transition_coverage_lowers_relevant_strings() -> None:
    rng = np.random.default_rng(11)
    task = even_ones_task()
    learner = BioLogicSequenceLearner(
        task=task,
        state_dim=64,
        input_dim=8,
        rng=rng,
        register_type="nearest",
    )
    learner.fit_from_dfa_traces([list("0"), list("00")], epochs=1, shuffle=False)
    metrics = learner.evaluate([list("1"), list("11"), list("101")])
    assert metrics["string_accuracy"] < 1.0


def test_masked_topk_cleanup_reaches_high_accuracy_with_sufficient_k() -> None:
    learner, _train_strings = _trained_even_ones()
    topk = learner.topk_masked_transition_accuracy([1, 2, 4, 8])
    assert topk[8] == 1.0

    masked = MaskedNearestRegister(learner.state_codebook)
    proposal = learner.state_codebook[0].copy()
    mask = np.ones(learner.state_dim, dtype=bool)
    recovered_idx, _ = masked.cleanup_masked(proposal, mask)
    assert recovered_idx == 0
