"""Experiment 4: sparse basin-based transitions."""

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
from biologic.metrics import sparse_transition_accuracy
from biologic.transition import SparseCoordinateTransition
from experiments.common import make_register, parallel_map, save_csv
from scripts.plot_results import plot_sparse_transitions

Row = dict[str, object]
Condition = tuple[int, int, int, str, str, float, int, int]


def _run_condition(
    args: Condition,
) -> Row:
    (
        seed,
        num_states,
        state_dim,
        register_type,
        mode,
        write_fraction,
        num_inputs,
        input_dim,
    ) = args
    rng = np.random.default_rng(seed)
    fsm = FiniteStateMachine.random(num_states, num_inputs, rng)
    state_codebook = random_state_codebook(num_states, state_dim, rng)
    input_codebook = random_input_codebook(num_inputs, input_dim, rng)
    register = make_register(register_type, state_codebook)
    transition = SparseCoordinateTransition(
        fsm=fsm,
        state_codebook=state_codebook,
        input_codebook=input_codebook,
        write_fraction=write_fraction,
        rng=rng,
        mode=mode,
        fixed_masks=True,
    )
    return {
        "experiment": "exp04_sparse_transitions",
        "seed": seed,
        "num_states": num_states,
        "num_inputs": num_inputs,
        "state_dim": state_dim,
        "input_dim": input_dim,
        "register_type": register_type,
        "write_fraction": write_fraction,
        "mode": mode,
        "sparse_transition_accuracy": sparse_transition_accuracy(
            fsm, transition, register, state_codebook, input_codebook
        ),
        "mean_abs_overlap": mean_abs_offdiag_overlap(state_codebook),
    }


def run_experiment(quick: bool = False, jobs: int | None = 1) -> pd.DataFrame:
    num_states_list: list[int] = [8] if quick else [8, 16, 32]
    num_inputs: int = 4
    state_dim_list: list[int] = [128, 256] if quick else [128, 256, 512]
    input_dim: int = 16
    write_fractions: list[float] = (
        [0.1, 0.3, 1.0]
        if quick
        else [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]
    )
    seeds: range = range(2) if quick else range(10)
    register_types: list[str] = ["nearest", "hopfield"]
    modes: list[str] = ["keep_current", "random_noise"]

    conditions: list[Condition] = [
        (
            seed,
            num_states,
            state_dim,
            register_type,
            mode,
            write_fraction,
            num_inputs,
            input_dim,
        )
        for seed, num_states, state_dim, register_type, mode, write_fraction in product(
            seeds,
            num_states_list,
            state_dim_list,
            register_types,
            modes,
            write_fractions,
        )
    ]
    print(
        f"Running experiment 4: sparse transitions ({len(conditions)} conditions, jobs={jobs})..."
    )
    rows: list[Row] = parallel_map(
        _run_condition,
        conditions,
        jobs=jobs,
        progress_label="Exp04 sparse transitions",
    )
    df: pd.DataFrame = pd.DataFrame(rows)
    csv_path = save_csv(df, "exp04_sparse_transitions.csv")
    figure_paths: list[Path] = plot_sparse_transitions(df)
    print(f"Saved CSV to {csv_path}")
    print(f"Saved figures to results/figures ({len(figure_paths)} files)")
    print(
        f"Summary:\n  mean sparse accuracy = {df['sparse_transition_accuracy'].mean():.3f}"
    )
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
