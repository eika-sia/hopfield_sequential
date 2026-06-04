"""Experiment 5: descriptor/payload separation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from biologic.descriptors import (
    COMPARE,
    COMPARE_FALSE,
    COMPARE_TRUE,
    ERROR,
    STORE,
    STORED,
    DescriptorPayloadMachine,
)
from biologic.encodings import random_bipolar
from experiments.common import parallel_map, save_csv
from scripts.plot_results import plot_descriptor_payload

Sequence = list[tuple[str, object]]
Row = dict[str, object]
Condition = tuple[int, list[float], int]


def _sequence_label(sequence: Sequence) -> str:
    labels: list[str] = []
    for kind, value in sequence:
        if kind == "descriptor":
            labels.append(str(value))
        else:
            labels.append("payload")
    return " ".join(labels)


def _run_case(
    seed: int,
    case_name: str,
    sequence: Sequence,
    expected_state: str,
    payload_id: str,
    descriptor: str,
) -> Row:
    machine: DescriptorPayloadMachine = DescriptorPayloadMachine()
    states: list[str] = machine.run_sequence(sequence)
    final_state: str = states[-1]
    return {
        "experiment": "exp05_descriptor_payload",
        "seed": seed,
        "case_name": case_name,
        "sequence": _sequence_label(sequence),
        "final_state": final_state,
        "expected_state": expected_state,
        "success": final_state == expected_state,
        "payload_id": payload_id,
        "descriptor": descriptor,
        "corruption_rate": np.nan,
        "num_trials": np.nan,
        "operation_success_rate": np.nan,
        "content_success_rate": np.nan,
    }


def _corruption_row(
    seed: int, corruption_rate: float, num_trials: int, p1: np.ndarray
) -> Row:
    rng: np.random.Generator = np.random.default_rng(
        seed + int(round(corruption_rate * 10_000))
    )
    operation_success: int = 0
    content_success: int = 0
    for _ in range(num_trials):
        descriptor: str = COMPARE
        if rng.random() < corruption_rate:
            descriptor = STORE
        machine: DescriptorPayloadMachine = DescriptorPayloadMachine()
        machine.run_sequence([("descriptor", STORE), ("payload", p1)])
        final_state: str = machine.run_sequence(
            [("descriptor", descriptor), ("payload", p1)]
        )[-1]
        operation_success += int(final_state == COMPARE_TRUE)
        content_success += int(final_state == COMPARE_TRUE)
    return {
        "experiment": "exp05_descriptor_payload",
        "seed": seed,
        "case_name": "descriptor_corruption",
        "sequence": "STORE payload then corrupted COMPARE payload",
        "final_state": "",
        "expected_state": COMPARE_TRUE,
        "success": np.nan,
        "payload_id": "p1",
        "descriptor": COMPARE,
        "corruption_rate": corruption_rate,
        "num_trials": num_trials,
        "operation_success_rate": operation_success / num_trials,
        "content_success_rate": content_success / num_trials,
    }


def _run_seed(args: Condition) -> list[Row]:
    seed, corruption_rates, num_trials = args
    rng: np.random.Generator = np.random.default_rng(seed)
    p1, p2, _p3 = random_bipolar((3, 32), rng)
    rows: list[Row] = [
        _run_case(
            seed,
            "same_payload_store",
            [("descriptor", STORE), ("payload", p1)],
            STORED,
            "p1",
            STORE,
        ),
        _run_case(
            seed,
            "same_payload_compare",
            [
                ("descriptor", STORE),
                ("payload", p1),
                ("descriptor", COMPARE),
                ("payload", p1),
            ],
            COMPARE_TRUE,
            "p1",
            COMPARE,
        ),
        _run_case(
            seed,
            "same_descriptor_store_p1",
            [("descriptor", STORE), ("payload", p1)],
            STORED,
            "p1",
            STORE,
        ),
        _run_case(
            seed,
            "same_descriptor_store_p2",
            [("descriptor", STORE), ("payload", p2)],
            STORED,
            "p2",
            STORE,
        ),
        _run_case(seed, "missing_descriptor", [("payload", p1)], ERROR, "p1", ""),
        _run_case(
            seed,
            "wrong_content_compare",
            [
                ("descriptor", STORE),
                ("payload", p1),
                ("descriptor", COMPARE),
                ("payload", p2),
            ],
            COMPARE_FALSE,
            "p2",
            COMPARE,
        ),
    ]
    for corruption_rate in corruption_rates:
        rows.append(_corruption_row(seed, corruption_rate, num_trials, p1))
    return rows


def run_experiment(quick: bool = False, jobs: int | None = 1) -> pd.DataFrame:
    seeds: range = range(2) if quick else range(10)
    corruption_rates: list[float] = (
        [0.0, 0.25, 0.5, 1.0]
        if quick
        else [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
    )
    num_trials: int = 50 if quick else 500
    conditions: list[Condition] = [
        (seed, corruption_rates, num_trials) for seed in seeds
    ]
    print(
        f"Running experiment 5: descriptor/payload separation ({len(conditions)} seeds, jobs={jobs})..."
    )
    rows: list[Row] = [
        row
        for group in parallel_map(
            _run_seed,
            conditions,
            jobs=jobs,
            progress_label="Exp05 descriptor/payload",
        )
        for row in group
    ]

    df: pd.DataFrame = pd.DataFrame(rows)
    csv_path = save_csv(df, "exp05_descriptor_payload.csv")
    figure_paths: list[Path] = plot_descriptor_payload(df)
    case_success: float = df[df["case_name"] != "descriptor_corruption"][
        "success"
    ].mean()
    print(f"Saved CSV to {csv_path}")
    print(f"Saved figures to results/figures ({len(figure_paths)} files)")
    print(f"Summary:\n  mean case success = {case_success:.3f}")
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
