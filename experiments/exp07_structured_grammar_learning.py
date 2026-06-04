"""Experiment 7: structured grammar learning from string demonstrations."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from multiprocessing.context import BaseContext
from pathlib import Path
from time import monotonic
from typing import Literal, Mapping

import numpy as np
import pandas as pd

from biologic.grammars import (
    GrammarTask,
    all_tasks,
    strings_covering_transitions,
    task_by_name,
    transition_coverage,
)
from biologic.learning import FeatureMode, OutputMode, UnwrittenMode
from biologic.sequence_eval import (
    BioLogicSequenceLearner,
    first_k_reaching,
    majority_label_baseline,
    make_length_splits,
    theoretical_topk_bound,
)
from experiments.common import (
    CSV_DIR,
    ensure_output_dirs,
    print_progress,
    progress_interval,
    resolve_jobs,
)
from scripts.plot_results import plot_structured_grammar_results

Profile = Literal["quick", "full"]
Row = dict[str, object]
ConditionKey = tuple[str, ...]
DataKey = tuple[str, ...]

CONDITION_KEY_COLUMNS: tuple[str, ...] = (
    "seed",
    "grammar_name",
    "train_max_len",
    "test_max_len",
    "train_num_strings",
    "test_num_strings",
    "regime",
    "feature_mode",
    "hidden_dim",
    "state_dim",
    "input_dim",
    "register_type",
    "output_mode",
    "write_fraction",
    "unwritten_mode",
    "epochs",
    "learning_rate",
)

CSV_COLUMNS: list[str] = [
    "experiment",
    "seed",
    "grammar_name",
    "alphabet_size",
    "num_dfa_states",
    "num_accept_states",
    "train_max_len",
    "test_max_len",
    "train_num_strings",
    "test_num_strings",
    "transition_coverage_train",
    "transition_coverage_test",
    "regime",
    "feature_mode",
    "feature_dim",
    "hidden_dim",
    "state_dim",
    "input_dim",
    "register_type",
    "output_mode",
    "write_fraction",
    "unwritten_mode",
    "epochs",
    "learning_rate",
    "theoretical_k_delta_05",
    "theoretical_k_delta_01",
    "notes",
    "row_type",
    "split",
    "length",
    "train_string_accuracy",
    "test_string_accuracy",
    "length_generalization_accuracy",
    "accept_accuracy",
    "reject_accuracy",
    "transition_accuracy_teacher_forced",
    "transition_accuracy_autonomous",
    "state_tracking_accuracy",
    "final_state_accuracy",
    "valid_next_symbol_accuracy",
    "extracted_transition_table_accuracy",
    "seen_transition_accuracy",
    "unseen_transition_accuracy",
    "ngram_baseline_accuracy",
    "majority_long_baseline_accuracy",
    "topk_k",
    "topk_fraction",
    "masked_topk_accuracy",
    "random_noise_topk_accuracy",
    "keep_current_topk_accuracy",
    "first_k_95",
    "first_k_99",
]


@dataclass(frozen=True)
class Condition:
    seed: int
    task_name: str
    train_max_len: int
    test_max_len: int
    train_num_strings: int
    test_num_strings: int
    regime: str
    feature_mode: FeatureMode
    hidden_dim: int | None
    state_dim: int
    input_dim: int
    register_type: str
    output_mode: OutputMode
    write_fraction: float
    unwritten_mode: UnwrittenMode
    epochs: int
    learning_rate: float
    with_topk: bool


def _select_tasks(profile: Profile) -> list[GrammarTask]:
    if profile == "quick":
        selected = [
            "even_ones",
            "no_substring_11",
            "contains_101",
            "ones_mod3_zero",
            "tomita_1",
            "tomita_2",
            "tomita_4",
        ]
        return [task_by_name(name) for name in selected]
    return all_tasks()


def _conditions(profile: Profile) -> list[Condition]:
    tasks = _select_tasks(profile)
    if profile == "quick":
        seeds = range(3)
        train_max_len = 8
        test_max_len = 16
        train_num_strings = 200
        test_num_strings = 500
        state_dim_values = [128]
        feature_configs: list[tuple[FeatureMode, int | None]] = [("exact_pair", None)]
        register_types = ["nearest"]
        output_configs: list[tuple[OutputMode, float, UnwrittenMode]] = [
            ("dense", 1.0, "random_noise")
        ]
    else:
        seeds = range(10)
        train_max_len = 12
        test_max_len = 64
        train_num_strings = 1000
        test_num_strings = 3000
        state_dim_values = [256, 512]
        feature_configs = [
            ("exact_pair", None),
            ("hashed_pair", 32),
            ("hashed_pair", 64),
            ("hashed_pair", 128),
            ("hashed_pair", 256),
        ]
        register_types = ["nearest", "hopfield"]
        output_configs = [
            ("dense", 1.0, "random_noise"),
            ("sparse_topk", 0.30, "random_noise"),
            ("sparse_topk", 0.50, "random_noise"),
            ("sparse_topk", 0.50, "keep_current"),
        ]
    input_dim = 16

    conditions: list[Condition] = []
    for seed, task, state_dim, feature_config, register_type, output_config in product(
        seeds,
        tasks,
        state_dim_values,
        feature_configs,
        register_types,
        output_configs,
    ):
        feature_mode, hidden_dim = feature_config
        output_mode, write_fraction, unwritten_mode = output_config
        if output_mode == "sparse_topk" and feature_mode != "exact_pair":
            continue
        conditions.append(
            Condition(
                seed=seed,
                task_name=task.name,
                train_max_len=train_max_len,
                test_max_len=test_max_len,
                train_num_strings=train_num_strings,
                test_num_strings=test_num_strings,
                regime="full_transition_exposure",
                feature_mode=feature_mode,
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                input_dim=input_dim,
                register_type=register_type,
                output_mode=output_mode,
                write_fraction=write_fraction,
                unwritten_mode=unwritten_mode,
                epochs=1
                if feature_mode == "exact_pair"
                else (10 if profile == "quick" else 50),
                learning_rate=1.0,
                with_topk=(
                    profile == "full"
                    and feature_mode == "exact_pair"
                    and register_type == "nearest"
                    and output_mode == "dense"
                ),
            )
        )
    # This is a separate coverage-failure regime. It intentionally shares most
    # model settings with exact-pair nearest dense rows, but differs by
    # ``regime`` and by the small training string budget.
    if profile == "full":
        limited_train_num_strings = 8
        for seed, task, state_dim in product(seeds, tasks, state_dim_values):
            conditions.append(
                Condition(
                    seed=seed,
                    task_name=task.name,
                    train_max_len=train_max_len,
                    test_max_len=test_max_len,
                    train_num_strings=limited_train_num_strings,
                    test_num_strings=test_num_strings,
                    regime="limited_string_exposure",
                    feature_mode="exact_pair",
                    hidden_dim=None,
                    state_dim=state_dim,
                    input_dim=input_dim,
                    register_type="nearest",
                    output_mode="dense",
                    write_fraction=1.0,
                    unwritten_mode="random_noise",
                    epochs=1,
                    learning_rate=1.0,
                    with_topk=False,
                )
            )
    return conditions


def _limited_string_exposure(
    task: GrammarTask,
    condition: Condition,
    rng: np.random.Generator,
) -> list[list[str]]:
    """Create a small string set that intentionally leaves some input contexts unseen."""
    if not task.alphabet:
        return [[]]
    visible_alphabet = task.alphabet[: max(1, len(task.alphabet) // 2)]
    max_len = max(1, condition.train_max_len)
    strings: list[list[str]] = []
    for index in range(condition.train_num_strings):
        length = 1 + (index % max_len)
        strings.append([str(rng.choice(visible_alphabet)) for _ in range(length)])
    return strings


def _make_strings(
    task: GrammarTask,
    condition: Condition,
    rng: np.random.Generator,
) -> tuple[list[list[str]], list[list[str]], list[list[str]], float, float]:
    sampled_train_count = (
        0
        if condition.regime == "limited_string_exposure"
        else condition.train_num_strings
    )
    sampled_train, in_dist, long = make_length_splits(
        task,
        condition.train_max_len,
        condition.test_max_len,
        sampled_train_count,
        condition.test_num_strings,
        rng,
    )
    if condition.regime == "limited_string_exposure":
        train_strings = _limited_string_exposure(task, condition, rng)
    else:
        train_strings = strings_covering_transitions(
            task,
            (0, condition.train_max_len),
            rng,
            condition.train_num_strings,
        )
        if len(train_strings) < condition.train_num_strings:
            train_strings.extend(
                sampled_train[: condition.train_num_strings - len(train_strings)]
            )
        train_strings = train_strings[: condition.train_num_strings]
    _, train_coverage = transition_coverage(task.dfa, train_strings)
    _, test_coverage = transition_coverage(task.dfa, [*in_dist, *long])
    return train_strings, in_dist, long, train_coverage, test_coverage


def _condition_data_key(condition: Condition) -> DataKey:
    values: tuple[object, ...] = (
        condition.seed,
        condition.task_name,
        condition.train_max_len,
        condition.test_max_len,
        condition.train_num_strings,
        condition.test_num_strings,
        condition.regime,
    )
    return tuple(_key_value(value) for value in values)


def _group_conditions(conditions: list[Condition]) -> list[list[Condition]]:
    groups: list[list[Condition]] = []
    by_key: dict[DataKey, list[Condition]] = {}
    for condition in conditions:
        key = _condition_data_key(condition)
        group = by_key.get(key)
        if group is None:
            group = []
            by_key[key] = group
            groups.append(group)
        group.append(condition)
    return groups


def _run_condition_with_data(
    condition: Condition,
    task: GrammarTask,
    train_strings: list[list[str]],
    in_dist: list[list[str]],
    long: list[list[str]],
    train_coverage: float,
    test_coverage: float,
    model_rng_state: dict[str, object],
) -> list[Row]:
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(model_rng_state)
    learner = BioLogicSequenceLearner(
        task=task,
        state_dim=condition.state_dim,
        input_dim=condition.input_dim,
        rng=rng,
        feature_mode=condition.feature_mode,
        hidden_dim=condition.hidden_dim,
        register_type=condition.register_type,
        output_mode=condition.output_mode,
        write_fraction=condition.write_fraction,
        unwritten_mode=condition.unwritten_mode,
        update_rule="delta",
        learning_rate=condition.learning_rate,
    )
    learner.fit_from_dfa_traces(
        train_strings,
        epochs=condition.epochs,
        deduplicate_transitions=False,
    )

    train_metrics = learner.evaluate(train_strings)
    in_dist_metrics = learner.evaluate(in_dist)
    long_metrics = learner.evaluate(long)
    majority_in_dist = majority_label_baseline(train_strings, in_dist, task)
    majority_long = majority_label_baseline(train_strings, long, task)
    seen_unseen = _transition_seen_unseen_accuracy(learner, train_strings)

    base = _base_row(condition, task, train_coverage, test_coverage)
    rows: list[Row] = [
        {
            **base,
            "row_type": "main",
            "split": "aggregate",
            "length": math.nan,
            "train_string_accuracy": train_metrics["string_accuracy"],
            "test_string_accuracy": in_dist_metrics["string_accuracy"],
            "length_generalization_accuracy": long_metrics["string_accuracy"],
            "accept_accuracy": in_dist_metrics["accept_accuracy"],
            "reject_accuracy": in_dist_metrics["reject_accuracy"],
            "transition_accuracy_teacher_forced": in_dist_metrics[
                "transition_accuracy_teacher_forced"
            ],
            "transition_accuracy_autonomous": in_dist_metrics[
                "transition_accuracy_autonomous"
            ],
            "state_tracking_accuracy": in_dist_metrics["state_tracking_accuracy"],
            "final_state_accuracy": in_dist_metrics["final_state_accuracy"],
            "valid_next_symbol_accuracy": in_dist_metrics["valid_next_symbol_accuracy"],
            "extracted_transition_table_accuracy": in_dist_metrics[
                "extracted_transition_table_accuracy"
            ],
            "seen_transition_accuracy": seen_unseen["seen_transition_accuracy"],
            "unseen_transition_accuracy": seen_unseen["unseen_transition_accuracy"],
            "ngram_baseline_accuracy": majority_in_dist,
            "majority_long_baseline_accuracy": majority_long,
            "topk_k": math.nan,
            "topk_fraction": math.nan,
            "masked_topk_accuracy": math.nan,
            "random_noise_topk_accuracy": math.nan,
            "keep_current_topk_accuracy": math.nan,
            "first_k_95": math.nan,
            "first_k_99": math.nan,
        }
    ]

    for length, accuracy in learner.accuracy_by_length([*in_dist, *long]).items():
        rows.append(
            {
                **base,
                "row_type": "length",
                "split": "by_length",
                "length": length,
                "train_string_accuracy": math.nan,
                "test_string_accuracy": accuracy,
                "length_generalization_accuracy": accuracy,
                "accept_accuracy": math.nan,
                "reject_accuracy": math.nan,
                "transition_accuracy_teacher_forced": math.nan,
                "transition_accuracy_autonomous": math.nan,
                "state_tracking_accuracy": math.nan,
                "final_state_accuracy": math.nan,
                "valid_next_symbol_accuracy": math.nan,
                "extracted_transition_table_accuracy": math.nan,
                "seen_transition_accuracy": math.nan,
                "unseen_transition_accuracy": math.nan,
                "ngram_baseline_accuracy": math.nan,
                "majority_long_baseline_accuracy": math.nan,
                "topk_k": math.nan,
                "topk_fraction": math.nan,
                "masked_topk_accuracy": math.nan,
                "random_noise_topk_accuracy": math.nan,
                "keep_current_topk_accuracy": math.nan,
                "first_k_95": math.nan,
                "first_k_99": math.nan,
            }
        )

    if condition.with_topk and condition.feature_mode == "exact_pair":
        k_values = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32]
        topk = learner.topk_masked_transition_accuracy(k_values)
        first_95 = first_k_reaching(topk, 0.95)
        first_99 = first_k_reaching(topk, 0.99)
        for k, accuracy in topk.items():
            rows.append(
                {
                    **base,
                    "row_type": "topk",
                    "split": "masked_topk",
                    "length": math.nan,
                    "train_string_accuracy": math.nan,
                    "test_string_accuracy": math.nan,
                    "length_generalization_accuracy": math.nan,
                    "accept_accuracy": math.nan,
                    "reject_accuracy": math.nan,
                    "transition_accuracy_teacher_forced": math.nan,
                    "transition_accuracy_autonomous": math.nan,
                    "state_tracking_accuracy": math.nan,
                    "final_state_accuracy": math.nan,
                    "valid_next_symbol_accuracy": math.nan,
                    "extracted_transition_table_accuracy": math.nan,
                    "seen_transition_accuracy": math.nan,
                    "unseen_transition_accuracy": math.nan,
                    "ngram_baseline_accuracy": math.nan,
                    "majority_long_baseline_accuracy": math.nan,
                    "topk_k": k,
                    "topk_fraction": k / condition.state_dim,
                    "masked_topk_accuracy": accuracy,
                    "random_noise_topk_accuracy": math.nan,
                    "keep_current_topk_accuracy": math.nan,
                    "first_k_95": first_95 if first_95 is not None else math.nan,
                    "first_k_99": first_99 if first_99 is not None else math.nan,
                }
            )
    return rows


def _run_condition_group(conditions: list[Condition]) -> list[Row]:
    if not conditions:
        return []
    reference = conditions[0]
    task = task_by_name(reference.task_name)
    data_rng = np.random.default_rng(reference.seed)
    train_strings, in_dist, long, train_coverage, test_coverage = _make_strings(
        task,
        reference,
        data_rng,
    )
    model_rng_state: dict[str, object] = copy.deepcopy(data_rng.bit_generator.state)
    rows: list[Row] = []
    for condition in conditions:
        rows.extend(
            _run_condition_with_data(
                condition,
                task,
                train_strings,
                in_dist,
                long,
                train_coverage,
                test_coverage,
                model_rng_state,
            )
        )
    return rows


def _run_condition(condition: Condition) -> list[Row]:
    return _run_condition_group([condition])


def _base_row(
    condition: Condition,
    task: GrammarTask,
    train_coverage: float,
    test_coverage: float,
) -> Row:
    theoretical_05 = theoretical_topk_bound(task.dfa.num_states, 0.05)
    theoretical_01 = theoretical_topk_bound(task.dfa.num_states, 0.01)
    return {
        "experiment": "exp07_structured_grammar_learning",
        "seed": condition.seed,
        "grammar_name": task.name,
        "alphabet_size": len(task.alphabet),
        "num_dfa_states": task.dfa.num_states,
        "num_accept_states": len(task.dfa.accept_states),
        "train_max_len": condition.train_max_len,
        "test_max_len": condition.test_max_len,
        "train_num_strings": condition.train_num_strings,
        "test_num_strings": condition.test_num_strings,
        "transition_coverage_train": train_coverage,
        "transition_coverage_test": test_coverage,
        "regime": condition.regime,
        "feature_mode": condition.feature_mode,
        "feature_dim": (
            task.dfa.num_states * len(task.alphabet)
            if condition.feature_mode == "exact_pair"
            else condition.hidden_dim
        ),
        "hidden_dim": (
            task.dfa.num_states * len(task.alphabet)
            if condition.hidden_dim is None
            else condition.hidden_dim
        ),
        "state_dim": condition.state_dim,
        "input_dim": condition.input_dim,
        "register_type": condition.register_type,
        "output_mode": condition.output_mode,
        "write_fraction": condition.write_fraction,
        "unwritten_mode": condition.unwritten_mode,
        "epochs": condition.epochs,
        "learning_rate": condition.learning_rate,
        "theoretical_k_delta_05": theoretical_05,
        "theoretical_k_delta_01": theoretical_01,
        "notes": task.description,
    }


def _key_value(value: object) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return ""
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.12g}"


def _condition_hidden_dim(condition: Condition) -> int:
    if condition.hidden_dim is not None:
        return condition.hidden_dim
    task = task_by_name(condition.task_name)
    return task.dfa.num_states * len(task.alphabet)


def _condition_key(condition: Condition) -> ConditionKey:
    values: dict[str, object] = {
        "seed": condition.seed,
        "grammar_name": condition.task_name,
        "train_max_len": condition.train_max_len,
        "test_max_len": condition.test_max_len,
        "train_num_strings": condition.train_num_strings,
        "test_num_strings": condition.test_num_strings,
        "regime": condition.regime,
        "feature_mode": condition.feature_mode,
        "hidden_dim": _condition_hidden_dim(condition),
        "state_dim": condition.state_dim,
        "input_dim": condition.input_dim,
        "register_type": condition.register_type,
        "output_mode": condition.output_mode,
        "write_fraction": condition.write_fraction,
        "unwritten_mode": condition.unwritten_mode,
        "epochs": condition.epochs,
        "learning_rate": condition.learning_rate,
    }
    return tuple(_key_value(values[column]) for column in CONDITION_KEY_COLUMNS)


def _row_condition_key(row: Mapping[str, object]) -> ConditionKey | None:
    if any(column not in row for column in CONDITION_KEY_COLUMNS):
        return None
    return tuple(_key_value(row[column]) for column in CONDITION_KEY_COLUMNS)


def _completed_condition_keys(path: Path) -> set[ConditionKey]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except (OSError, pd.errors.ParserError) as exc:
        print(f"Warning: cannot read existing CSV for continue mode: {exc}")
        return set()
    if "row_type" not in df.columns:
        print(
            "Warning: existing CSV has no row_type column; continue mode cannot skip rows."
        )
        return set()
    missing = [column for column in CONDITION_KEY_COLUMNS if column not in df.columns]
    if missing:
        joined = ", ".join(missing)
        print(f"Warning: existing CSV is missing continue key columns: {joined}")
        return set()
    completed: set[ConditionKey] = set()
    main_rows = df[df["row_type"] == "main"]
    for row in main_rows.to_dict("records"):
        key = _row_condition_key(row)
        if key is not None:
            completed.add(key)
    return completed


def _remaining_conditions(
    conditions: list[Condition],
    completed: set[ConditionKey],
) -> list[Condition]:
    return [
        condition
        for condition in conditions
        if _condition_key(condition) not in completed
    ]


def _transition_seen_unseen_accuracy(
    learner: BioLogicSequenceLearner,
    train_strings: list[list[str]],
) -> dict[str, float]:
    covered, _coverage = transition_coverage(learner.task.dfa, train_strings)
    seen_correct = 0
    seen_total = 0
    unseen_correct = 0
    unseen_total = 0
    for state, symbol, target in learner.task.dfa.transition_items():
        state_idx = learner.state_to_idx[state]
        input_idx = learner.symbol_to_idx[symbol]
        target_idx = learner.state_to_idx[target]
        pred_idx = learner._transition_step(state_idx, input_idx)
        if (state, symbol) in covered:
            seen_correct += int(pred_idx == target_idx)
            seen_total += 1
        else:
            unseen_correct += int(pred_idx == target_idx)
            unseen_total += 1
    return {
        "seen_transition_accuracy": (
            seen_correct / seen_total if seen_total else math.nan
        ),
        "unseen_transition_accuracy": (
            unseen_correct / unseen_total if unseen_total else math.nan
        ),
    }


def _write_rows(path: Path, condition_rows: list[Row], append: bool) -> None:
    with path.open("a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if not append:
            writer.writeheader()
        for row in condition_rows:
            writer.writerow(
                {column: row.get(column, math.nan) for column in CSV_COLUMNS}
            )


def _initialize_csv(path: Path, append_existing: bool) -> None:
    if append_existing and path.exists() and path.stat().st_size > 0:
        return
    _write_rows(path, [], append=False)


def _effective_jobs(jobs: int | None) -> int:
    return resolve_jobs(jobs)


def _run_conditions_to_csv(
    conditions: list[Condition],
    path: Path,
    jobs: int,
    append_existing: bool = False,
) -> None:
    _initialize_csv(path, append_existing)
    condition_groups = _group_conditions(conditions)
    if jobs <= 1 or len(conditions) <= 1:
        start_time = monotonic()
        total = len(conditions)
        completed = 0
        progress_every = progress_interval(total)
        for group in condition_groups:
            _write_rows(path, _run_condition_group(group), append=True)
            completed += len(group)
            if completed == total or completed % progress_every == 0:
                print_progress(
                    "Exp07 structured grammars",
                    completed,
                    total,
                    start_time,
                )
        return

    start_time = monotonic()
    total = len(conditions)
    completed = 0
    progress_every = progress_interval(total)
    context: BaseContext | None = (
        mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
    )
    with ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(_run_condition_group, group): len(group)
            for group in condition_groups
        }
        for future in as_completed(futures):
            _write_rows(path, future.result(), append=True)
            completed += futures[future]
            if completed == total or completed % progress_every == 0:
                print_progress(
                    "Exp07 structured grammars",
                    completed,
                    total,
                    start_time,
                )


def run_experiment(
    quick: bool = False,
    jobs: int | None = 1,
    continue_run: bool = False,
) -> pd.DataFrame:
    selected_profile: Profile = "quick" if quick else "full"
    conditions = _conditions(selected_profile)
    ensure_output_dirs()
    csv_path = CSV_DIR / "exp07_structured_grammar_learning.csv"
    total_conditions = len(conditions)
    append_existing = False
    if continue_run:
        completed = _completed_condition_keys(csv_path)
        skipped = sum(
            1 for condition in conditions if _condition_key(condition) in completed
        )
        conditions = _remaining_conditions(conditions, completed)
        append_existing = csv_path.exists() and csv_path.stat().st_size > 0
        print(
            "Continue mode: "
            f"found {len(completed)} completed condition keys, "
            f"skipping {skipped}/{total_conditions}, "
            f"remaining {len(conditions)}."
        )
    effective_jobs = _effective_jobs(jobs)
    print(
        "Running experiment 7: structured grammar learning "
        f"profile={selected_profile}, conditions={len(conditions)}/{total_conditions}, "
        f"jobs={effective_jobs}"
    )
    if conditions:
        _run_conditions_to_csv(
            conditions,
            csv_path,
            jobs=effective_jobs,
            append_existing=append_existing,
        )
    elif not csv_path.exists():
        _initialize_csv(csv_path, append_existing=False)
    else:
        print("No remaining Exp07 conditions to run.")
    df = pd.read_csv(csv_path)
    figure_paths: list[Path] = plot_structured_grammar_results(df)
    print(f"Saved CSV to {csv_path}")
    print(f"Saved figures to results/figures ({len(figure_paths)} files)")
    main_rows = df[df["row_type"] == "main"]
    full_rows = main_rows[main_rows["regime"] == "full_transition_exposure"]
    limited_rows = main_rows[main_rows["regime"] == "limited_string_exposure"]
    print(
        "Summary:\n"
        f"  full-transition mean test string accuracy = {full_rows['test_string_accuracy'].mean():.3f}\n"
        f"  full-transition mean length generalization accuracy = {full_rows['length_generalization_accuracy'].mean():.3f}"
    )
    if not limited_rows.empty:
        print(
            "  limited-exposure mean test string accuracy = "
            f"{limited_rows['test_string_accuracy'].mean():.3f}"
        )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run a small smoke test.")
    mode.add_argument(
        "--full",
        action="store_true",
        help="Run the full structured grammar sweep.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Worker processes; 0 uses all CPUs.",
    )
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Append missing conditions by skipping completed main rows in the existing CSV.",
    )
    args = parser.parse_args()
    profile: Profile = "full" if args.full else "quick"
    run_experiment(
        quick=profile == "quick",
        jobs=args.jobs,
        continue_run=args.continue_run,
    )


if __name__ == "__main__":
    main()
