"""Experiment 1: exact FSM transition accuracy."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from biologic.encodings import (
    mean_abs_offdiag_overlap,
    random_input_codebook,
    random_state_codebook,
)
from biologic.fsm import FiniteStateMachine
from biologic.metrics import transition_accuracy
from biologic.transition import ExactCoordinateTransition
from experiments.common import make_register, parallel_map, save_csv
from scripts.plot_results import plot_transition_accuracy

Row = dict[str, object]
Condition = tuple[int, int, int, int, str, int]


def _run_condition(args: Condition) -> Row:
    seed, num_states, num_inputs, state_dim, register_type, input_dim = args
    rng = np.random.default_rng(seed)
    fsm = FiniteStateMachine.random(num_states, num_inputs, rng)
    state_codebook = random_state_codebook(num_states, state_dim, rng)
    input_codebook = random_input_codebook(num_inputs, input_dim, rng)
    register = make_register(register_type, state_codebook)
    transition = ExactCoordinateTransition(fsm, state_codebook, input_codebook)
    return {
        "experiment": "exp01_transition_accuracy",
        "seed": seed,
        "num_states": num_states,
        "num_inputs": num_inputs,
        "state_dim": state_dim,
        "input_dim": input_dim,
        "register_type": register_type,
        "transition_accuracy": transition_accuracy(
            fsm, transition, register, state_codebook, input_codebook
        ),
        "mean_abs_overlap": mean_abs_offdiag_overlap(state_codebook),
    }


def run_experiment(quick: bool = False, jobs: int | None = 1) -> pd.DataFrame:
    num_states_list: list[int] = [4, 8] if quick else [4, 8, 16, 32, 64]
    num_inputs_list: list[int] = [2] if quick else [2, 4, 8]
    state_dim_list: list[int] = [64, 128] if quick else [64, 128, 256, 512]
    input_dim: int = 16
    seeds: range = range(2) if quick else range(10)
    register_types: list[str] = ["nearest", "hopfield"]

    conditions: list[Condition] = [
        (seed, num_states, num_inputs, state_dim, register_type, input_dim)
        for seed, num_states, num_inputs, state_dim, register_type in product(
            seeds, num_states_list, num_inputs_list, state_dim_list, register_types
        )
    ]
    print(
        f"Running experiment 1: transition accuracy ({len(conditions)} conditions, jobs={jobs})..."
    )
    rows: list[Row] = parallel_map(_run_condition, conditions, jobs=jobs)
    df: pd.DataFrame = pd.DataFrame(rows)
    csv_path = save_csv(df, "exp01_transition_accuracy.csv")
    figure_paths: list[Path] = plot_transition_accuracy(df)
    print(f"Saved CSV to {csv_path}")
    print(f"Saved figures to results/figures ({len(figure_paths)} files)")
    print(f"Summary:\n  mean accuracy = {df['transition_accuracy'].mean():.3f}")
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
