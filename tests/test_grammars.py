import numpy as np

from biologic.grammars import (
    contains_101_task,
    even_ones_task,
    no_substring_11_task,
    reber_task,
    tomita_tasks,
)


def test_dfa_step_run_trace_works() -> None:
    task = even_ones_task()
    assert task.dfa.step("even", "1") == "odd"
    assert task.dfa.run(["1", "1"]) == ("even", True)
    assert task.dfa.trace(["1", "0", "1"]) == ["even", "odd", "odd", "even"]


def test_even_number_of_ones_examples() -> None:
    task = even_ones_task()
    assert task.dfa.accepts([])
    assert task.dfa.accepts(["1", "1"])
    assert task.dfa.accepts(["1", "0", "1"])
    assert not task.dfa.accepts(["1"])
    assert not task.dfa.accepts(["0", "1", "0"])


def test_no_substring_11_examples() -> None:
    task = no_substring_11_task()
    assert task.dfa.accepts([])
    assert task.dfa.accepts(["1", "0", "1", "0"])
    assert task.dfa.accepts(["0", "0", "1"])
    assert not task.dfa.accepts(["1", "1"])
    assert not task.dfa.accepts(["0", "1", "1", "0"])


def test_contains_101_examples() -> None:
    task = contains_101_task()
    assert task.dfa.accepts(["1", "0", "1"])
    assert task.dfa.accepts(["0", "1", "0", "1", "0"])
    assert not task.dfa.accepts([])
    assert not task.dfa.accepts(["1", "1", "0", "0"])


def test_tomita_grammars_known_examples() -> None:
    tasks = {task.name: task for task in tomita_tasks()}
    assert tasks["tomita_1"].dfa.accepts(list("111"))
    assert not tasks["tomita_1"].dfa.accepts(list("101"))
    assert tasks["tomita_2"].dfa.accepts(list("1010"))
    assert not tasks["tomita_2"].dfa.accepts(list("100"))
    assert tasks["tomita_3"].dfa.accepts(list("100"))
    assert not tasks["tomita_3"].dfa.accepts(list("10"))
    assert tasks["tomita_4"].dfa.accepts(list("001001"))
    assert not tasks["tomita_4"].dfa.accepts(list("10001"))
    assert tasks["tomita_5"].dfa.accepts(list("0011"))
    assert not tasks["tomita_5"].dfa.accepts(list("01"))
    assert tasks["tomita_6"].dfa.accepts(list("000"))
    assert not tasks["tomita_6"].dfa.accepts(list("001"))
    assert tasks["tomita_7"].dfa.accepts(list("00110011"))
    assert not tasks["tomita_7"].dfa.accepts(list("1010"))


def test_reber_generator_produces_accepted_strings() -> None:
    rng = np.random.default_rng(0)
    task = reber_task()
    for _ in range(20):
        symbols = task.sample_positive((0, 20), rng)
        assert task.dfa.accepts(symbols)


def test_corrupted_reber_strings_are_often_rejected() -> None:
    rng = np.random.default_rng(1)
    task = reber_task()
    rejected = 0
    for _ in range(50):
        symbols = task.sample_positive((0, 20), rng)
        if len(symbols) > 2:
            symbols[1] = "E"
        rejected += int(not task.dfa.accepts(symbols))
    assert rejected >= 40
