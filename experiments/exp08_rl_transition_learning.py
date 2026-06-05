"""Experiment 8: reward-modulated transition learning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from multiprocessing.context import BaseContext
from pathlib import Path
from time import monotonic
from typing import Literal, Mapping

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from biologic.grammars import GrammarTask, task_by_name
from biologic.learning import FeatureMode
from biologic.rl_transition import (
    CleanupType,
    OutputMode,
    RLCondition,
    RLEvaluation,
    RLTransitionAgent,
)
from experiments.common import print_progress, progress_interval, resolve_jobs

Profile = Literal["quick", "full"]
Row = dict[str, object]
ConditionKey = tuple[str, ...]
DataKey = tuple[str, ...]

CSV_DIR = Path("results/csv")
FIGURE_DIR = Path("results/figures/exp08_rl_transition_learning")

SUMMARY_COLUMNS: list[str] = [
    "seed",
    "grammar",
    "condition",
    "cleanup",
    "feature_type",
    "state_dim",
    "num_states",
    "feature_dim",
    "episodes",
    "eta",
    "alpha",
    "gamma",
    "lambda",
    "noise_start",
    "noise_end",
    "write_fraction",
    "final_test_accuracy",
    "final_return",
    "state_tracking_accuracy",
    "transition_accuracy",
    "seen_transition_accuracy",
    "unseen_transition_accuracy",
    "episodes_to_80",
    "episodes_to_90",
]

CURVE_COLUMNS: list[str] = [
    "seed",
    "grammar",
    "condition",
    "cleanup",
    "feature_type",
    "episode",
    "mean_train_return",
    "test_accuracy",
    "state_tracking_accuracy",
    "transition_accuracy",
    "avg_td_error",
    "avg_trace_norm",
]

TRANSITION_COLUMNS: list[str] = [
    "seed",
    "grammar",
    "condition",
    "cleanup",
    "feature_type",
    "state_id",
    "symbol",
    "seen_in_training",
    "true_next_state",
    "pred_next_state",
    "correct",
]

ALL_COLUMNS: list[str] = ["row_type", *SUMMARY_COLUMNS]
for _column in [*CURVE_COLUMNS, *TRANSITION_COLUMNS]:
    if _column not in ALL_COLUMNS:
        ALL_COLUMNS.append(_column)

CONDITION_KEY_COLUMNS: tuple[str, ...] = (
    "seed",
    "grammar",
    "condition",
    "cleanup",
    "feature_type",
    "state_dim",
    "episodes",
    "eta",
    "alpha",
    "gamma",
    "lambda",
    "noise_start",
    "noise_end",
    "write_fraction",
)


@dataclass(frozen=True)
class Condition:
    seed: int
    grammar: str
    condition: RLCondition
    cleanup: CleanupType
    feature_type: FeatureMode
    hidden_dim: int | None
    state_dim: int
    input_dim: int
    episodes: int
    eval_every: int
    train_max_len: int
    test_max_len: int
    test_strings: int
    eta: float
    alpha: float
    gamma: float
    lambda_trace: float
    noise_start: float
    noise_end: float
    write_fraction: float


@dataclass
class ConditionResult:
    summary_rows: list[Row]
    curve_rows: list[Row]
    transition_rows: list[Row]


@dataclass(frozen=True)
class DataBundle:
    task: GrammarTask
    train_strings: list[list[str]]
    test_strings: list[list[str]]


def _format_value(value: object) -> str:
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


def _stable_seed(*values: object) -> int:
    hasher = hashlib.blake2b(digest_size=8)
    for value in values:
        hasher.update(str(value).encode("utf-8"))
        hasher.update(b"\0")
    return int.from_bytes(hasher.digest(), byteorder="little", signed=False)


def _task_names(profile: Profile) -> list[str]:
    if profile == "quick":
        return ["even_ones", "contains_101"]
    return [
        "even_ones",
        "contains_101",
        "no_substring_11",
        "tomita_1",
        "tomita_2",
        "reber",
        "tier_alternating_12",
    ]


def _conditions(profile: Profile) -> list[Condition]:
    if profile == "quick":
        seeds = range(2)
        episodes = 300
        eval_every = 50
        train_max_len = 8
        test_max_len = 16
        test_strings = 300
        state_dims = [64]
        cleanups: list[CleanupType] = ["nearest"]
        features: list[tuple[FeatureMode, int | None]] = [("exact_pair", None)]
        condition_names: list[RLCondition] = [
            "supervised_baseline",
            "state_shaped_rl",
            "terminal_rl",
            "sparse_terminal_rl",
        ]
        sparse_write_fractions = [0.50]
    else:
        seeds = range(5)
        episodes = 5000
        eval_every = 250
        train_max_len = 12
        test_max_len = 32
        test_strings = 1000
        state_dims = [128]
        cleanups = ["nearest", "hopfield"]
        features = [("exact_pair", None), ("hashed_pair", 32)]
        condition_names = [
            "supervised_baseline",
            "state_shaped_rl",
            "terminal_rl",
            "sparse_terminal_rl",
        ]
        sparse_write_fractions = [0.30, 0.50]

    conditions: list[Condition] = []
    for seed, grammar, state_dim, cleanup, feature in product(
        seeds,
        _task_names(profile),
        state_dims,
        cleanups,
        features,
    ):
        feature_type, hidden_dim = feature
        for condition_name in condition_names:
            if condition_name == "sparse_terminal_rl":
                if feature_type != "exact_pair":
                    continue
                for write_fraction in sparse_write_fractions:
                    conditions.append(
                        _make_condition(
                            seed,
                            grammar,
                            condition_name,
                            cleanup,
                            feature_type,
                            hidden_dim,
                            state_dim,
                            episodes,
                            eval_every,
                            train_max_len,
                            test_max_len,
                            test_strings,
                            write_fraction,
                        )
                    )
                continue
            conditions.append(
                _make_condition(
                    seed,
                    grammar,
                    condition_name,
                    cleanup,
                    feature_type,
                    hidden_dim,
                    state_dim,
                    episodes,
                    eval_every,
                    train_max_len,
                    test_max_len,
                    test_strings,
                    1.0,
                )
            )
    return conditions


def _make_condition(
    seed: int,
    grammar: str,
    condition_name: RLCondition,
    cleanup: CleanupType,
    feature_type: FeatureMode,
    hidden_dim: int | None,
    state_dim: int,
    episodes: int,
    eval_every: int,
    train_max_len: int,
    test_max_len: int,
    test_strings: int,
    write_fraction: float,
) -> Condition:
    return Condition(
        seed=seed,
        grammar=grammar,
        condition=condition_name,
        cleanup=cleanup,
        feature_type=feature_type,
        hidden_dim=hidden_dim,
        state_dim=state_dim,
        input_dim=16,
        episodes=episodes,
        eval_every=eval_every,
        train_max_len=train_max_len,
        test_max_len=test_max_len,
        test_strings=test_strings,
        eta=0.005,
        alpha=0.01,
        gamma=0.95,
        lambda_trace=0.8,
        noise_start=1.0,
        noise_end=0.05,
        write_fraction=write_fraction,
    )


def _condition_key(condition: Condition) -> ConditionKey:
    values: dict[str, object] = {
        "seed": condition.seed,
        "grammar": condition.grammar,
        "condition": condition.condition,
        "cleanup": condition.cleanup,
        "feature_type": condition.feature_type,
        "state_dim": condition.state_dim,
        "episodes": condition.episodes,
        "eta": condition.eta,
        "alpha": condition.alpha,
        "gamma": condition.gamma,
        "lambda": condition.lambda_trace,
        "noise_start": condition.noise_start,
        "noise_end": condition.noise_end,
        "write_fraction": condition.write_fraction,
    }
    return tuple(_format_value(values[column]) for column in CONDITION_KEY_COLUMNS)


def _row_key(row: Mapping[str, object]) -> ConditionKey | None:
    if any(column not in row for column in CONDITION_KEY_COLUMNS):
        return None
    return tuple(_format_value(row[column]) for column in CONDITION_KEY_COLUMNS)


def _completed_keys(path: Path) -> set[ConditionKey]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except (OSError, pd.errors.ParserError) as exc:
        print(f"Warning: cannot read existing Exp08 CSV for continue: {exc}")
        return set()
    if "row_type" in df.columns:
        df = df[df["row_type"] == "summary"]
    missing = [column for column in CONDITION_KEY_COLUMNS if column not in df.columns]
    if missing:
        print(f"Warning: Exp08 CSV missing continue columns: {', '.join(missing)}")
        return set()
    completed: set[ConditionKey] = set()
    for row in df.to_dict("records"):
        key = _row_key(row)
        if key is not None:
            completed.add(key)
    return completed


def _data_key(condition: Condition) -> DataKey:
    values: tuple[object, ...] = (
        condition.seed,
        condition.grammar,
        condition.episodes,
        condition.train_max_len,
        condition.test_max_len,
        condition.test_strings,
    )
    return tuple(_format_value(value) for value in values)


def _group_conditions(conditions: list[Condition]) -> list[list[Condition]]:
    groups: list[list[Condition]] = []
    by_key: dict[DataKey, list[Condition]] = {}
    for condition in conditions:
        key = _data_key(condition)
        group = by_key.get(key)
        if group is None:
            group = []
            by_key[key] = group
            groups.append(group)
        group.append(condition)
    return groups


def _make_bundle(condition: Condition) -> DataBundle:
    rng = np.random.default_rng(condition.seed)
    task = task_by_name(condition.grammar)
    train = [
        symbols
        for symbols, _label in task.sample_balanced_dataset(
            condition.episodes,
            (0, condition.train_max_len),
            rng,
        )
    ]
    test = [
        symbols
        for symbols, _label in task.sample_balanced_dataset(
            condition.test_strings,
            (0, condition.test_max_len),
            rng,
        )
    ]
    return DataBundle(task=task, train_strings=train, test_strings=test)


def _noise_std(condition: Condition, episode: int) -> float:
    if condition.episodes <= 1:
        return condition.noise_end
    progress = (episode - 1) / (condition.episodes - 1)
    return condition.noise_start + progress * (
        condition.noise_end - condition.noise_start
    )


def _agent_rng(condition: Condition) -> np.random.Generator:
    return np.random.default_rng(
        _stable_seed(
            condition.seed,
            condition.grammar,
            condition.condition,
            condition.cleanup,
            condition.feature_type,
            condition.hidden_dim,
            condition.write_fraction,
        )
    )


def _run_condition_with_bundle(
    condition: Condition,
    bundle: DataBundle,
) -> ConditionResult:
    agent = RLTransitionAgent(
        task=bundle.task,
        state_dim=condition.state_dim,
        input_dim=condition.input_dim,
        rng=_agent_rng(condition),
        feature_type=condition.feature_type,
        hidden_dim=condition.hidden_dim,
        cleanup=condition.cleanup,
        output_mode="sparse_topk"
        if condition.condition == "sparse_terminal_rl"
        else "dense",
        write_fraction=condition.write_fraction,
        eta=condition.eta,
        alpha=condition.alpha,
        gamma=condition.gamma,
        lambda_trace=condition.lambda_trace,
    )
    curve_rows: list[Row] = []
    returns: list[float] = []
    td_errors: list[float] = []
    trace_norms: list[float] = []
    final_eval: RLEvaluation | None = None

    for episode, symbols in enumerate(bundle.train_strings, start=1):
        stats = agent.train_episode(
            symbols,
            condition.condition,
            noise_std=_noise_std(condition, episode),
        )
        returns.append(stats.episode_return)
        td_errors.append(stats.mean_td_error)
        trace_norms.append(stats.mean_trace_norm)
        if episode == 1 or episode % condition.eval_every == 0 or episode == condition.episodes:
            evaluation = agent.evaluate(bundle.test_strings, bundle.train_strings)
            final_eval = evaluation
            curve_rows.append(
                _curve_row(
                    condition,
                    episode,
                    returns,
                    td_errors,
                    trace_norms,
                    evaluation,
                )
            )
            returns = []
            td_errors = []
            trace_norms = []
    if final_eval is None:
        final_eval = agent.evaluate(bundle.test_strings, bundle.train_strings)
    summary = _summary_row(condition, bundle.task, agent, final_eval, curve_rows)
    transition_rows = [
        _transition_row(condition, item)
        for item in agent.transition_eval_items(bundle.train_strings)
    ]
    return ConditionResult(
        summary_rows=[summary],
        curve_rows=curve_rows,
        transition_rows=transition_rows,
    )


def _curve_row(
    condition: Condition,
    episode: int,
    returns: list[float],
    td_errors: list[float],
    trace_norms: list[float],
    evaluation: RLEvaluation,
) -> Row:
    return {
        "seed": condition.seed,
        "grammar": condition.grammar,
        "condition": condition.condition,
        "cleanup": condition.cleanup,
        "feature_type": condition.feature_type,
        "episode": episode,
        "mean_train_return": float(np.mean(returns)) if returns else math.nan,
        "test_accuracy": evaluation.string_accuracy,
        "state_tracking_accuracy": evaluation.state_tracking_accuracy,
        "transition_accuracy": evaluation.transition_accuracy,
        "avg_td_error": float(np.mean(td_errors)) if td_errors else math.nan,
        "avg_trace_norm": float(np.mean(trace_norms)) if trace_norms else math.nan,
    }


def _summary_row(
    condition: Condition,
    task: GrammarTask,
    agent: RLTransitionAgent,
    evaluation: RLEvaluation,
    curve_rows: list[Row],
) -> Row:
    return {
        "seed": condition.seed,
        "grammar": condition.grammar,
        "condition": condition.condition,
        "cleanup": condition.cleanup,
        "feature_type": condition.feature_type,
        "state_dim": condition.state_dim,
        "num_states": task.dfa.num_states,
        "feature_dim": agent.feature_dim,
        "episodes": condition.episodes,
        "eta": condition.eta,
        "alpha": condition.alpha,
        "gamma": condition.gamma,
        "lambda": condition.lambda_trace,
        "noise_start": condition.noise_start,
        "noise_end": condition.noise_end,
        "write_fraction": condition.write_fraction,
        "final_test_accuracy": evaluation.string_accuracy,
        "final_return": curve_rows[-1]["mean_train_return"] if curve_rows else math.nan,
        "state_tracking_accuracy": evaluation.state_tracking_accuracy,
        "transition_accuracy": evaluation.transition_accuracy,
        "seen_transition_accuracy": evaluation.seen_transition_accuracy,
        "unseen_transition_accuracy": evaluation.unseen_transition_accuracy,
        "episodes_to_80": _episodes_to(curve_rows, 0.80),
        "episodes_to_90": _episodes_to(curve_rows, 0.90),
    }


def _transition_row(condition: Condition, item: object) -> Row:
    return {
        "seed": condition.seed,
        "grammar": condition.grammar,
        "condition": condition.condition,
        "cleanup": condition.cleanup,
        "feature_type": condition.feature_type,
        "state_id": getattr(item, "state_id"),
        "symbol": getattr(item, "symbol"),
        "seen_in_training": getattr(item, "seen_in_training"),
        "true_next_state": getattr(item, "true_next_state"),
        "pred_next_state": getattr(item, "pred_next_state"),
        "correct": getattr(item, "correct"),
    }


def _episodes_to(curve_rows: list[Row], threshold: float) -> int | float:
    for row in curve_rows:
        value = row.get("test_accuracy", math.nan)
        if isinstance(value, (float, int)) and float(value) >= threshold:
            return int(row["episode"])
    return math.nan


def _run_condition_group(conditions: list[Condition]) -> ConditionResult:
    if not conditions:
        return ConditionResult([], [], [])
    bundle = _make_bundle(conditions[0])
    result = ConditionResult([], [], [])
    for condition in conditions:
        condition_result = _run_condition_with_bundle(condition, bundle)
        result.summary_rows.extend(condition_result.summary_rows)
        result.curve_rows.extend(condition_result.curve_rows)
        result.transition_rows.extend(condition_result.transition_rows)
    return result


def _ensure_dirs() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _csv_paths() -> dict[str, Path]:
    return {
        "all": CSV_DIR / "exp08_rl_transition_learning.csv",
    }


def _write_rows(path: Path, rows: list[Row], columns: list[str], append: bool) -> None:
    with path.open("a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not append:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, math.nan) for column in columns})


def _initialize_output(paths: dict[str, Path], append_existing: bool) -> None:
    if append_existing:
        if not paths["all"].exists():
            _write_rows(paths["all"], [], ALL_COLUMNS, append=False)
        return
    _write_rows(paths["all"], [], ALL_COLUMNS, append=False)


def _append_result(paths: dict[str, Path], result: ConditionResult) -> None:
    all_rows = [
        *(_with_row_type(row, "summary") for row in result.summary_rows),
        *(_with_row_type(row, "learning_curve") for row in result.curve_rows),
        *(_with_row_type(row, "transition_eval") for row in result.transition_rows),
    ]
    _write_rows(paths["all"], all_rows, ALL_COLUMNS, append=True)


def _with_row_type(row: Row, row_type: str) -> Row:
    return {"row_type": row_type, **row}


def _run_groups_to_csv(
    condition_groups: list[list[Condition]],
    paths: dict[str, Path],
    jobs: int,
    append_existing: bool,
) -> None:
    _initialize_output(paths, append_existing)
    total = sum(len(group) for group in condition_groups)
    completed = 0
    start_time = monotonic()
    progress_every = progress_interval(total)
    if jobs <= 1 or len(condition_groups) <= 1:
        for group in condition_groups:
            result = _run_condition_group(group)
            _append_result(paths, result)
            completed += len(group)
            if completed == total or completed % progress_every == 0:
                print_progress("Exp08 RL transition learning", completed, total, start_time)
        return

    context: BaseContext | None = (
        mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
    )
    with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as executor:
        futures = {
            executor.submit(_run_condition_group, group): len(group)
            for group in condition_groups
        }
        for future in as_completed(futures):
            _append_result(paths, future.result())
            completed += futures[future]
            if completed == total or completed % progress_every == 0:
                print_progress("Exp08 RL transition learning", completed, total, start_time)


def _plot_results(paths: dict[str, Path]) -> list[Path]:
    if not paths["all"].exists():
        return []
    data = pd.read_csv(paths["all"])
    summary = data[data["row_type"] == "summary"].copy()
    curves = data[data["row_type"] == "learning_curve"].copy()
    output_paths: list[Path] = []
    if not curves.empty:
        output_paths.append(_plot_learning_curves(curves))
    if not summary.empty:
        output_paths.append(_plot_final_accuracy(summary))
        output_paths.append(_plot_seen_unseen(summary))
        sparse = summary[summary["condition"] == "sparse_terminal_rl"]
        if not sparse.empty:
            output_paths.append(_plot_sparse_write_fraction(sparse))
    return output_paths


def _plot_learning_curves(curves: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=300, constrained_layout=True)
    grouped = (
        curves.groupby(["condition", "episode"], dropna=False)["test_accuracy"]
        .mean()
        .reset_index()
    )
    for condition, part in grouped.groupby("condition"):
        part = part.sort_values("episode")
        ax.plot(part["episode"], part["test_accuracy"], marker="o", markersize=3, label=condition)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Test string accuracy")
    ax.set_ylim(0.0, 1.03)
    ax.set_title("Exp08 RL transition learning curves")
    ax.legend(fontsize=8)
    path = FIGURE_DIR / "exp08_learning_curves.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_final_accuracy(summary: pd.DataFrame) -> Path:
    grouped = (
        summary.groupby("condition", dropna=False)["final_test_accuracy"]
        .mean()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=300, constrained_layout=True)
    ax.bar(range(len(grouped)), grouped.to_numpy())
    ax.set_xticks(range(len(grouped)), grouped.index, rotation=25, ha="right")
    ax.set_ylabel("Final test string accuracy")
    ax.set_ylim(0.0, 1.03)
    ax.set_title("Final accuracy by training signal")
    path = FIGURE_DIR / "exp08_final_accuracy_by_condition.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_seen_unseen(summary: pd.DataFrame) -> Path:
    grouped = summary.groupby("condition", dropna=False)[
        ["seen_transition_accuracy", "unseen_transition_accuracy"]
    ].mean()
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=300, constrained_layout=True)
    x = np.arange(len(grouped))
    width = 0.36
    ax.bar(x - width / 2, grouped["seen_transition_accuracy"], width, label="seen")
    ax.bar(x + width / 2, grouped["unseen_transition_accuracy"], width, label="unseen")
    ax.set_xticks(x, grouped.index, rotation=25, ha="right")
    ax.set_ylabel("Transition accuracy")
    ax.set_ylim(0.0, 1.03)
    ax.set_title("Seen vs unseen hidden transitions")
    ax.legend(fontsize=8)
    path = FIGURE_DIR / "exp08_seen_unseen_transition_accuracy.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_sparse_write_fraction(summary: pd.DataFrame) -> Path:
    grouped = (
        summary.groupby("write_fraction", dropna=False)["final_test_accuracy"]
        .mean()
        .reset_index()
        .sort_values("write_fraction")
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=300, constrained_layout=True)
    ax.plot(
        grouped["write_fraction"],
        grouped["final_test_accuracy"],
        marker="o",
        linewidth=1.8,
    )
    ax.set_xlabel("Write fraction")
    ax.set_ylabel("Final test string accuracy")
    ax.set_ylim(0.0, 1.03)
    ax.set_title("Sparse terminal RL")
    path = FIGURE_DIR / "exp08_sparse_write_fraction.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def run_experiment(
    quick: bool = False,
    jobs: int | None = 1,
    continue_run: bool = False,
) -> pd.DataFrame:
    profile: Profile = "quick" if quick else "full"
    _ensure_dirs()
    paths = _csv_paths()
    conditions = _conditions(profile)
    total_conditions = len(conditions)
    append_existing = False
    if continue_run:
        completed = _completed_keys(paths["all"])
        skipped = sum(1 for condition in conditions if _condition_key(condition) in completed)
        conditions = [
            condition for condition in conditions if _condition_key(condition) not in completed
        ]
        append_existing = paths["all"].exists() and paths["all"].stat().st_size > 0
        print(
            "Continue mode: "
            f"found {len(completed)} completed keys, "
            f"skipping {skipped}/{total_conditions}, "
            f"remaining {len(conditions)}."
        )
    effective_jobs = resolve_jobs(jobs)
    groups = _group_conditions(conditions)
    print(
        "Running experiment 8: RL transition learning "
        f"profile={profile}, conditions={len(conditions)}/{total_conditions}, "
        f"data_groups={len(groups)}, jobs={effective_jobs}"
    )
    if conditions:
        _run_groups_to_csv(groups, paths, effective_jobs, append_existing)
    elif not paths["all"].exists():
        _initialize_output(paths, append_existing=False)
    else:
        print("No remaining Exp08 conditions to run.")
    figure_paths = _plot_results(paths)
    data = pd.read_csv(paths["all"])
    summary = data[data["row_type"] == "summary"].copy()
    print(f"Saved Exp08 CSV outputs to {CSV_DIR}")
    print(f"Saved Exp08 figures to {FIGURE_DIR} ({len(figure_paths)} files)")
    if not summary.empty:
        print(
            "Summary:\n"
            f"  mean final test accuracy = {summary['final_test_accuracy'].mean():.3f}\n"
            f"  mean transition accuracy = {summary['transition_accuracy'].mean():.3f}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run a small Exp08 smoke run.")
    mode.add_argument("--full", action="store_true", help="Run the full Exp08 sweep.")
    parser.add_argument("--jobs", type=int, default=1, help="Worker processes; 0 uses all CPUs.")
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Append missing conditions by skipping completed summary rows.",
    )
    args = parser.parse_args()
    run_experiment(
        quick=not args.full,
        jobs=args.jobs,
        continue_run=args.continue_run,
    )


if __name__ == "__main__":
    main()
