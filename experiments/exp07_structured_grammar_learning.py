"""Experiment 7: structured grammar learning from string demonstrations."""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from multiprocessing.context import BaseContext
from typing import Literal

import numpy as np
import pandas as pd

from biologic.grammars import (
    GrammarTask,
    all_tasks,
    custom_tasks,
    reber_task,
    strings_covering_transitions,
    task_by_name,
    tomita_tasks,
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
from experiments.common import CSV_DIR, ensure_output_dirs, resolve_jobs
from scripts.plot_results import plot_structured_grammar_results

TaskGroup = Literal["all", "custom", "tomita", "reber"]
Profile = Literal["quick", "full", "advanced"]
Row = dict[str, object]
MAX_EXP07_WORKERS: int = 4

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


def _select_tasks(task_group: TaskGroup, profile: Profile) -> list[GrammarTask]:
    if task_group == "custom":
        return custom_tasks()
    if task_group == "tomita":
        tasks = tomita_tasks()
        return tasks[:4] if profile == "quick" else tasks
    if task_group == "reber":
        return [reber_task()]
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


def _conditions(
    profile: Profile,
    task_group: TaskGroup,
    with_topk: bool,
    with_hopfield: bool,
) -> list[Condition]:
    tasks = _select_tasks(task_group, profile)
    if profile == "quick":
        seeds = range(3)
        train_max_len = 8
        test_max_len = 16
        train_num_strings = 200
        test_num_strings = 500
        state_dim_values = [128]
    elif profile == "advanced":
        seeds = range(10)
        train_max_len = 12
        test_max_len = 64
        train_num_strings = 1000
        test_num_strings = 3000
        state_dim_values = [256, 512]
    else:
        seeds = range(5)
        train_max_len = 12
        test_max_len = 32
        train_num_strings = 500
        test_num_strings = 1000
        state_dim_values = [256]
    input_dim = 16
    if profile == "quick":
        feature_configs: list[tuple[FeatureMode, int | None]] = [
            ("exact_pair", None)
        ]
    elif profile == "advanced":
        feature_configs = [
            ("exact_pair", None),
            ("hashed_pair", 32),
            ("hashed_pair", 64),
            ("hashed_pair", 128),
            ("hashed_pair", 256),
        ]
    else:
        feature_configs = [
            ("exact_pair", None),
            ("hashed_pair", 32),
            ("hashed_pair", 128),
        ]
    register_types = ["nearest"]
    if with_hopfield:
        register_types.append("hopfield")
    output_configs: list[tuple[OutputMode, float, UnwrittenMode]] = [
        ("dense", 1.0, "random_noise")
    ]
    if with_topk and profile != "quick":
        output_configs.extend(
            [
                ("sparse_topk", 0.30, "random_noise"),
                ("sparse_topk", 0.50, "random_noise"),
                ("sparse_topk", 0.50, "keep_current"),
            ]
        )

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
                with_topk=with_topk,
                )
            )
    if profile in {"full", "advanced"}:
        limited_train_num_strings = 4 if profile == "full" else 8
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
        strings.append(
            [str(rng.choice(visible_alphabet)) for _ in range(length)]
        )
    return strings


def _make_strings(
    task: GrammarTask,
    condition: Condition,
    rng: np.random.Generator,
) -> tuple[list[list[str]], list[list[str]], list[list[str]], float, float]:
    sampled_train, in_dist, long = make_length_splits(
        task,
        condition.train_max_len,
        condition.test_max_len,
        condition.train_num_strings,
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


def _run_condition(condition: Condition) -> list[Row]:
    rng = np.random.default_rng(condition.seed)
    task = task_by_name(condition.task_name)
    train_strings, in_dist, long, train_coverage, test_coverage = _make_strings(
        task,
        condition,
        rng,
    )
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
    learner.fit_from_dfa_traces(train_strings, epochs=condition.epochs)

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
            "valid_next_symbol_accuracy": in_dist_metrics[
                "valid_next_symbol_accuracy"
            ],
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


def _limit_worker_memory(worker_memory_gb: float | None) -> None:
    if worker_memory_gb is None or worker_memory_gb <= 0.0:
        return
    try:
        import resource
    except ImportError:
        return
    limit_bytes = int(worker_memory_gb * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))


def _write_rows(path: Path, condition_rows: list[Row], append: bool) -> None:
    with path.open("a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if not append:
            writer.writeheader()
        for row in condition_rows:
            writer.writerow({column: row.get(column, math.nan) for column in CSV_COLUMNS})


def _effective_jobs(
    jobs: int | None,
    max_workers: int | None,
    profile: Profile,
) -> int:
    resolved_jobs = resolve_jobs(jobs)
    if max_workers == 0:
        return resolved_jobs
    if max_workers is not None:
        return max(1, min(resolved_jobs, max_workers))
    if profile == "advanced":
        return resolved_jobs
    return min(resolved_jobs, MAX_EXP07_WORKERS)


def _run_conditions_to_csv(
    conditions: list[Condition],
    path: Path,
    jobs: int,
    worker_memory_gb: float | None,
) -> None:
    if jobs <= 1 or len(conditions) <= 1:
        _write_rows(path, [], append=False)
        for condition in conditions:
            _write_rows(path, _run_condition(condition), append=True)
        return

    _write_rows(path, [], append=False)
    context: BaseContext | None = (
        mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
    )
    with ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=context,
        initializer=_limit_worker_memory,
        initargs=(worker_memory_gb,),
    ) as executor:
        futures = [executor.submit(_run_condition, condition) for condition in conditions]
        for future in as_completed(futures):
            _write_rows(path, future.result(), append=True)


def run_experiment(
    quick: bool = False,
    jobs: int | None = 1,
    task_group: TaskGroup = "all",
    with_topk: bool = True,
    with_hopfield: bool = False,
    profile: Profile | None = None,
    max_workers: int | None = None,
    worker_memory_gb: float | None = None,
    total_memory_gb: float | None = None,
) -> pd.DataFrame:
    selected_profile: Profile = profile if profile is not None else ("quick" if quick else "full")
    conditions = _conditions(
        profile=selected_profile,
        task_group=task_group,
        with_topk=with_topk,
        with_hopfield=with_hopfield,
    )
    effective_jobs = _effective_jobs(jobs, max_workers, selected_profile)
    effective_worker_memory_gb = worker_memory_gb
    if total_memory_gb is not None and total_memory_gb > 0.0:
        per_worker = total_memory_gb / effective_jobs
        effective_worker_memory_gb = (
            per_worker
            if effective_worker_memory_gb is None
            else min(effective_worker_memory_gb, per_worker)
        )
    print(
        "Running experiment 7: structured grammar learning "
        f"({len(conditions)} conditions, jobs={effective_jobs})..."
    )
    ensure_output_dirs()
    csv_path = CSV_DIR / "exp07_structured_grammar_learning.csv"
    _run_conditions_to_csv(
        conditions,
        csv_path,
        jobs=effective_jobs,
        worker_memory_gb=effective_worker_memory_gb,
    )
    df = pd.read_csv(csv_path)
    figure_paths: list[Path] = plot_structured_grammar_results(df)
    print(f"Saved CSV to {csv_path}")
    print(f"Saved figures to results/figures ({len(figure_paths)} files)")
    main_rows = df[df["row_type"] == "main"]
    print(
        "Summary:\n"
        f"  mean test string accuracy = {main_rows['test_string_accuracy'].mean():.3f}\n"
        f"  mean length generalization accuracy = {main_rows['length_generalization_accuracy'].mean():.3f}"
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    mode.add_argument(
        "--advanced",
        action="store_true",
        help="Run harder grammar/model sweeps intended for long jobs.",
    )
    parser.add_argument(
        "--task",
        choices=["all", "custom", "tomita", "reber"],
        default="all",
    )
    parser.add_argument("--with-topk", action="store_true", default=False)
    parser.add_argument("--with-hopfield", action="store_true", default=False)
    parser.add_argument(
        "--with-torch-baselines",
        action="store_true",
        help="Accepted for CLI compatibility; no torch baseline is run.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Worker processes; 0 uses all CPUs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Cap Exp07 worker processes. Use 0 for no cap; quick/full default to "
            "a RAM-safe cap, advanced defaults to all requested jobs."
        ),
    )
    parser.add_argument(
        "--worker-memory-gb",
        type=float,
        default=None,
        help="Optional per-worker address-space limit in GiB on Unix.",
    )
    parser.add_argument(
        "--total-memory-gb",
        type=float,
        default=None,
        help=(
            "Optional total RAM budget in GiB; divided across effective Exp07 "
            "workers as a per-worker address-space limit."
        ),
    )
    args = parser.parse_args()
    del args.with_torch_baselines
    if args.advanced:
        profile: Profile = "advanced"
    elif args.full:
        profile = "full"
    else:
        profile = "quick"
    run_experiment(
        quick=profile == "quick",
        jobs=args.jobs,
        task_group=args.task,
        with_topk=args.with_topk or profile in {"quick", "advanced"},
        with_hopfield=args.with_hopfield or profile == "advanced",
        profile=profile,
        max_workers=args.max_workers,
        worker_memory_gb=args.worker_memory_gb,
        total_memory_gb=args.total_memory_gb,
    )


if __name__ == "__main__":
    main()
