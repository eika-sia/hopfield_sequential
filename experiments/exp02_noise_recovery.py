"""Experiment 2: state-register recovery after noise."""

from __future__ import annotations

import argparse
from itertools import product

import numpy as np
import pandas as pd

from biologic.encodings import mean_abs_offdiag_overlap, random_state_codebook
from biologic.metrics import recovery_accuracy
from experiments.common import make_register, parallel_map, save_csv
from scripts.plot_results import plot_noise_recovery


def _run_condition(args: tuple[int, int, int, str, list[float], int]) -> list[dict[str, object]]:
    seed, num_states, state_dim, register_type, flip_fractions, trials_per_state = args
    rng = np.random.default_rng(seed)
    state_codebook = random_state_codebook(num_states, state_dim, rng)
    register = make_register(register_type, state_codebook)
    overlap = mean_abs_offdiag_overlap(state_codebook)
    rows = []
    for flip_fraction in flip_fractions:
        rows.append(
            {
                "experiment": "exp02_noise_recovery",
                "seed": seed,
                "num_states": num_states,
                "state_dim": state_dim,
                "register_type": register_type,
                "flip_fraction": flip_fraction,
                "trials_per_state": trials_per_state,
                "recovery_accuracy": recovery_accuracy(
                    register, state_codebook, flip_fraction, trials_per_state, rng
                ),
                "mean_abs_overlap": overlap,
            }
        )
    return rows


def run_experiment(quick: bool = False, jobs: int | None = 1) -> pd.DataFrame:
    num_states_list = [8, 16] if quick else [8, 16, 32, 64]
    state_dim_list = [64, 128] if quick else [64, 128, 256, 512]
    flip_fractions = [0.0, 0.1, 0.2] if quick else [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    trials_per_state = 5 if quick else 20
    seeds = range(2) if quick else range(10)
    register_types = ["nearest", "hopfield"]

    conditions = [
        (seed, num_states, state_dim, register_type, flip_fractions, trials_per_state)
        for seed, num_states, state_dim, register_type in product(
            seeds, num_states_list, state_dim_list, register_types
        )
    ]
    print(f"Running experiment 2: noise recovery ({len(conditions)} conditions, jobs={jobs})...")
    rows = [row for group in parallel_map(_run_condition, conditions, jobs=jobs) for row in group]
    df = pd.DataFrame(rows)
    csv_path = save_csv(df, "exp02_noise_recovery.csv")
    figure_paths = plot_noise_recovery(df)
    print(f"Saved CSV to {csv_path}")
    print(f"Saved figures to results/figures ({len(figure_paths)} files)")
    print(f"Summary:\n  mean recovery = {df['recovery_accuracy'].mean():.3f}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--jobs", type=int, default=1, help="Worker threads; 0 uses all CPUs.")
    args = parser.parse_args()
    run_experiment(quick=args.quick, jobs=args.jobs)


if __name__ == "__main__":
    main()
