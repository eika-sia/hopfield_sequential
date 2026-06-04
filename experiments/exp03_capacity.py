"""Experiment 3: capacity sweep for state-register recovery."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from biologic.encodings import mean_abs_offdiag_overlap, random_state_codebook
from biologic.metrics import recovery_accuracy
from experiments.common import make_register, parallel_map, save_csv
from scripts.plot_results import plot_capacity

Row = dict[str, object]
Condition = tuple[int, int, float, str, float, int]


def _run_condition(args: Condition) -> Row:
    seed, state_dim, capacity_ratio, register_type, flip_fraction, trials_per_state = (
        args
    )
    rng = np.random.default_rng(seed)
    num_states = max(2, int(capacity_ratio * state_dim))
    state_codebook = random_state_codebook(num_states, state_dim, rng)
    register = make_register(register_type, state_codebook)
    return {
        "experiment": "exp03_capacity",
        "seed": seed,
        "state_dim": state_dim,
        "num_states": num_states,
        "capacity_ratio": capacity_ratio,
        "register_type": register_type,
        "flip_fraction": flip_fraction,
        "recovery_accuracy": recovery_accuracy(
            register, state_codebook, flip_fraction, trials_per_state, rng
        ),
        "mean_abs_overlap": mean_abs_offdiag_overlap(state_codebook),
    }


def run_experiment(quick: bool = False, jobs: int | None = 1) -> pd.DataFrame:
    state_dim_list: list[int] = [64, 128] if quick else [64, 128, 256, 512, 1024]
    capacity_ratios: list[float] = (
        [0.05, 0.138, 0.3]
        if quick
        else [0.025, 0.05, 0.10, 0.138, 0.20, 0.30, 0.50, 0.75, 1.00]
    )
    flip_fraction: float = 0.10
    trials_per_state: int = 5 if quick else 20
    seeds: range = range(2) if quick else range(10)
    register_types: list[str] = ["nearest", "hopfield"]

    conditions: list[Condition] = [
        (
            seed,
            state_dim,
            capacity_ratio,
            register_type,
            flip_fraction,
            trials_per_state,
        )
        for seed, state_dim, capacity_ratio, register_type in product(
            seeds, state_dim_list, capacity_ratios, register_types
        )
    ]
    print(
        f"Running experiment 3: capacity ({len(conditions)} conditions, jobs={jobs})..."
    )
    rows: list[Row] = parallel_map(
        _run_condition,
        conditions,
        jobs=jobs,
        progress_label="Exp03 capacity",
    )
    df: pd.DataFrame = pd.DataFrame(rows)
    csv_path = save_csv(df, "exp03_capacity.csv")
    figure_paths: list[Path] = plot_capacity(df)
    print(f"Saved CSV to {csv_path}")
    print(f"Saved figures to results/figures ({len(figure_paths)} files)")
    print(f"Summary:\n  mean recovery = {df['recovery_accuracy'].mean():.3f}")
    return df


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--jobs", type=int, default=1, help="Worker processes; 0 uses all CPUs."
    )
    args: argparse.Namespace = parser.parse_args()
    run_experiment(quick=args.quick, jobs=args.jobs)


if __name__ == "__main__":
    main()
