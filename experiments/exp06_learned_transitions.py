"""Experiment 6: learned FSM transitions from demonstrations."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from biologic.encodings import (
    mean_abs_offdiag_overlap,
    random_input_codebook,
    random_state_codebook,
)
from biologic.fsm import FiniteStateMachine
from biologic.learning import (
    EligibilityTraceTransitionLearner,
    FeatureMode,
    LearnedAssociativeTransition,
    OutputMode,
    StateInputPairEncoder,
    UnwrittenMode,
    UpdateRule,
)
from biologic.metrics import normalized_overlap, transition_accuracy
from biologic.transition import ExactCoordinateTransition
from experiments.common import make_register, parallel_map, save_csv
from scripts.plot_results import plot_learning_results

Row = dict[str, object]
Pair = tuple[int, int]


@dataclass(frozen=True)
class TransitionDemonstration:
    """Observed transition example used by the learner."""

    state_idx: int
    input_idx: int
    next_state_idx: int


@dataclass(frozen=True)
class TrainingItem:
    """Precomputed feature-target example."""

    phi: np.ndarray
    target_state_vec: np.ndarray
    state_idx: int
    input_idx: int
    next_state_idx: int


@dataclass(frozen=True)
class Condition:
    seed: int
    num_states: int
    num_inputs: int
    state_dim: int
    input_dim: int
    feature_mode: FeatureMode
    hidden_dim: int | None
    training_mode: str
    coverage_fraction: float
    update_rule: UpdateRule
    output_mode: OutputMode
    write_fraction: float
    unwritten_mode: UnwrittenMode
    epochs: tuple[int, ...]
    notes: str


def generate_transition_demonstrations(
    fsm: FiniteStateMachine,
    state_codebook: np.ndarray,
    input_codebook: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    num_steps: int | None = None,
    coverage_fraction: float = 1.0,
) -> list[TransitionDemonstration]:
    """Generate observed transition examples without exposing them to learners."""
    del state_codebook, input_codebook
    if not 0.0 <= coverage_fraction <= 1.0:
        raise ValueError("coverage_fraction must be in [0, 1]")

    if mode == "full_table":
        return [
            TransitionDemonstration(state_idx, input_idx, next_state_idx)
            for state_idx, input_idx, next_state_idx in fsm.all_transitions()
        ]

    if mode == "coverage_sweep":
        all_examples = [
            TransitionDemonstration(state_idx, input_idx, next_state_idx)
            for state_idx, input_idx, next_state_idx in fsm.all_transitions()
        ]
        rng.shuffle(all_examples)
        count = int(round(coverage_fraction * len(all_examples)))
        if coverage_fraction > 0.0:
            count = max(1, count)
        return all_examples[:count]

    if mode == "trajectory":
        steps = num_steps if num_steps is not None else fsm.num_states * fsm.num_inputs
        if steps <= 0:
            raise ValueError("num_steps must be positive")
        state_idx = int(rng.integers(0, fsm.num_states))
        examples: list[TransitionDemonstration] = []
        for _ in range(steps):
            input_idx = int(rng.integers(0, fsm.num_inputs))
            next_state_idx = fsm.next_state(state_idx, input_idx)
            examples.append(
                TransitionDemonstration(state_idx, input_idx, next_state_idx)
            )
            state_idx = next_state_idx
        return examples

    raise ValueError(f"unsupported training mode: {mode}")


def _make_encoder(
    condition: Condition,
    rng: np.random.Generator,
) -> StateInputPairEncoder:
    hidden_dim = condition.hidden_dim
    if condition.feature_mode in {"exact_pair", "noisy_exact_pair"}:
        hidden_dim = None
    return StateInputPairEncoder(
        num_states=condition.num_states,
        num_inputs=condition.num_inputs,
        mode=condition.feature_mode,
        rng=rng,
        hidden_dim=hidden_dim,
        state_dim=condition.state_dim,
        input_dim=condition.input_dim,
        threshold=0.0,
        num_active_features=1,
    )


def _encode_demonstration(
    demonstration: TransitionDemonstration,
    encoder: StateInputPairEncoder,
    state_codebook: np.ndarray,
    input_codebook: np.ndarray,
) -> TrainingItem:
    state_vec = state_codebook[demonstration.state_idx]
    input_vec = input_codebook[demonstration.input_idx]
    phi = encoder.encode(
        state_vec,
        input_vec,
        demonstration.state_idx,
        demonstration.input_idx,
    )
    return TrainingItem(
        phi=phi,
        target_state_vec=state_codebook[demonstration.next_state_idx],
        state_idx=demonstration.state_idx,
        input_idx=demonstration.input_idx,
        next_state_idx=demonstration.next_state_idx,
    )


def _training_examples(
    demonstrations: Sequence[TransitionDemonstration],
    encoder: StateInputPairEncoder,
    state_codebook: np.ndarray,
    input_codebook: np.ndarray,
) -> list[TrainingItem]:
    return [
        _encode_demonstration(demo, encoder, state_codebook, input_codebook)
        for demo in demonstrations
    ]


def _learner_examples(items: Sequence[TrainingItem]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(item.phi, item.target_state_vec) for item in items]


def _seen_pairs(items: Sequence[TrainingItem]) -> set[Pair]:
    return {(item.state_idx, item.input_idx) for item in items}


def _basin_margin(proposal: np.ndarray, target_idx: int, codebook: np.ndarray) -> float:
    overlaps = (codebook @ proposal) / codebook.shape[1]
    target_overlap = float(overlaps[target_idx])
    if len(overlaps) <= 1:
        return target_overlap
    non_target = np.delete(overlaps, target_idx)
    return target_overlap - float(np.max(non_target))


def _evaluate_writer(
    fsm: FiniteStateMachine,
    writer: LearnedAssociativeTransition,
    encoder: StateInputPairEncoder,
    register: object,
    state_codebook: np.ndarray,
    input_codebook: np.ndarray,
    seen: set[Pair],
) -> dict[str, float]:
    all_correct = 0
    seen_correct = 0
    unseen_correct = 0
    total = 0
    seen_total = 0
    unseen_total = 0
    bit_accuracies: list[float] = []
    target_overlaps: list[float] = []
    basin_margins: list[float] = []

    for state_idx, input_idx, target_idx in fsm.all_transitions():
        state_vec = state_codebook[state_idx]
        input_vec = input_codebook[input_idx]
        target_vec = state_codebook[target_idx]
        proposal = writer.propose(state_vec, input_vec, encoder, state_idx, input_idx)
        recovered_idx, _ = register.cleanup(proposal)  # type: ignore[attr-defined]
        correct = int(recovered_idx == target_idx)
        pair = (state_idx, input_idx)

        all_correct += correct
        total += 1
        if pair in seen:
            seen_correct += correct
            seen_total += 1
        else:
            unseen_correct += correct
            unseen_total += 1
        bit_accuracies.append(float(np.mean(proposal == target_vec)))
        target_overlaps.append(normalized_overlap(proposal, target_vec))
        basin_margins.append(_basin_margin(proposal, target_idx, state_codebook))

    return {
        "seen_transition_accuracy": seen_correct / seen_total if seen_total else math.nan,
        "unseen_transition_accuracy": (
            unseen_correct / unseen_total if unseen_total else math.nan
        ),
        "all_transition_accuracy": all_correct / total if total else 0.0,
        "raw_bit_accuracy": float(np.mean(bit_accuracies)) if bit_accuracies else math.nan,
        "raw_overlap_with_target": (
            float(np.mean(target_overlaps)) if target_overlaps else math.nan
        ),
        "basin_margin_mean": (
            float(np.mean(basin_margins)) if basin_margins else math.nan
        ),
    }


def _threshold_epochs(rows: list[Row], register_type: str, threshold: float) -> float:
    register_rows = [
        row for row in rows if row["register_type"] == register_type
    ]
    register_rows.sort(key=lambda row: int(row["epochs"]))
    for row in register_rows:
        value = float(row["all_transition_accuracy"])
        if value >= threshold:
            return float(row["epochs"])
    return math.nan


def _run_condition(condition: Condition) -> list[Row]:
    rng = np.random.default_rng(condition.seed)
    fsm = FiniteStateMachine.random(
        condition.num_states,
        condition.num_inputs,
        rng,
    )
    state_codebook = random_state_codebook(condition.num_states, condition.state_dim, rng)
    input_codebook = random_input_codebook(condition.num_inputs, condition.input_dim, rng)
    encoder = _make_encoder(condition, rng)
    writer = LearnedAssociativeTransition(
        state_dim=condition.state_dim,
        feature_dim=encoder.feature_dim,
        rng=rng,
        learning_rate=1.0,
        update_rule=condition.update_rule,
        output_mode=condition.output_mode,
        write_fraction=condition.write_fraction,
        unwritten_mode=condition.unwritten_mode,
    )
    learner = EligibilityTraceTransitionLearner(
        writer,
        trace_decay=0.0,
        normalize_features=False,
    )
    demonstrations = generate_transition_demonstrations(
        fsm,
        state_codebook,
        input_codebook,
        mode=condition.training_mode,
        rng=rng,
        coverage_fraction=condition.coverage_fraction,
    )
    training_items = _training_examples(
        demonstrations,
        encoder,
        state_codebook,
        input_codebook,
    )
    examples = _learner_examples(training_items)
    seen = _seen_pairs(training_items)
    registers = {
        "nearest": make_register("nearest", state_codebook),
        "hopfield": make_register("hopfield", state_codebook),
    }
    exact_transition = ExactCoordinateTransition(fsm, state_codebook, input_codebook)
    exact_upper = {
        register_type: transition_accuracy(
            fsm,
            exact_transition,
            register,
            state_codebook,
            input_codebook,
        )
        for register_type, register in registers.items()
    }
    untrained = writer.copy()
    untrained_baseline = {
        register_type: _evaluate_writer(
            fsm,
            untrained,
            encoder,
            register,
            state_codebook,
            input_codebook,
            seen,
        )["all_transition_accuracy"]
        for register_type, register in registers.items()
    }

    rows: list[Row] = []
    current_epoch = 0
    epoch_values = sorted(set(condition.epochs))
    for epoch in epoch_values:
        while current_epoch < epoch:
            learner.train_epoch(
                examples,
                rng=rng,
                shuffle=condition.training_mode != "trajectory",
            )
            current_epoch += 1
        for register_type, register in registers.items():
            metrics = _evaluate_writer(
                fsm,
                writer,
                encoder,
                register,
                state_codebook,
                input_codebook,
                seen,
            )
            row: Row = {
                "experiment": "exp06_learned_transitions",
                "seed": condition.seed,
                "num_states": condition.num_states,
                "num_inputs": condition.num_inputs,
                "state_dim": condition.state_dim,
                "input_dim": condition.input_dim,
                "feature_mode": condition.feature_mode,
                "feature_dim": encoder.feature_dim,
                "hidden_dim": (
                    encoder.feature_dim
                    if condition.hidden_dim is None
                    else condition.hidden_dim
                ),
                "training_mode": condition.training_mode,
                "coverage_fraction": condition.coverage_fraction,
                "epochs": epoch,
                "update_rule": condition.update_rule,
                "output_mode": condition.output_mode,
                "write_fraction": condition.write_fraction,
                "unwritten_mode": condition.unwritten_mode,
                "register_type": register_type,
                "cleanup_type": register_type,
                "num_training_examples": len(training_items),
                "seen_transition_accuracy": metrics["seen_transition_accuracy"],
                "unseen_transition_accuracy": metrics["unseen_transition_accuracy"],
                "all_transition_accuracy": metrics["all_transition_accuracy"],
                "raw_bit_accuracy": metrics["raw_bit_accuracy"],
                "raw_overlap_with_target": metrics["raw_overlap_with_target"],
                "basin_margin_mean": metrics["basin_margin_mean"],
                "exact_upper_bound_accuracy": exact_upper[register_type],
                "untrained_baseline_accuracy": untrained_baseline[register_type],
                "mean_abs_overlap": mean_abs_offdiag_overlap(state_codebook),
                "notes": condition.notes,
            }
            rows.append(row)

    for register_type in registers:
        epochs_to_95 = _threshold_epochs(rows, register_type, 0.95)
        epochs_to_99 = _threshold_epochs(rows, register_type, 0.99)
        for row in rows:
            if row["register_type"] == register_type:
                row["epochs_to_95"] = epochs_to_95
                row["epochs_to_99"] = epochs_to_99
    return rows


def _dense_feature_configs(
    quick: bool,
) -> list[tuple[FeatureMode, int | None]]:
    if quick:
        return [
            ("exact_pair", None),
            ("hashed_pair", 32),
            ("hashed_pair", 64),
            ("hashed_pair", 128),
        ]
    configs: list[tuple[FeatureMode, int | None]] = [("exact_pair", None)]
    configs.extend(("hashed_pair", value) for value in [32, 64, 128, 256, 512, 1024])
    configs.extend(
        ("random_conjunctive", value)
        for value in [128, 256, 512, 1024, 2048]
    )
    return configs


def _conditions(quick: bool) -> list[Condition]:
    seeds = range(3) if quick else range(10)
    num_states_list = [8, 16] if quick else [8, 16, 32, 64]
    num_inputs_list = [2, 4] if quick else [2, 4, 8]
    state_dim_list = [128, 256] if quick else [128, 256, 512]
    input_dim = 16
    dense_epochs = (0, 1, 2, 5, 10) if quick else (0, 1, 2, 5, 10, 20, 50)
    update_rules: list[UpdateRule] = ["delta"] if quick else ["hebbian", "delta"]

    conditions: list[Condition] = []
    for seed, num_states, num_inputs, state_dim, update_rule in product(
        seeds,
        num_states_list,
        num_inputs_list,
        state_dim_list,
        update_rules,
    ):
        for feature_mode, hidden_dim in _dense_feature_configs(quick):
            conditions.append(
                Condition(
                    seed=seed,
                    num_states=num_states,
                    num_inputs=num_inputs,
                    state_dim=state_dim,
                    input_dim=input_dim,
                    feature_mode=feature_mode,
                    hidden_dim=hidden_dim,
                    training_mode="full_table",
                    coverage_fraction=1.0,
                    update_rule=update_rule,
                    output_mode="dense",
                    write_fraction=1.0,
                    unwritten_mode="random_noise",
                    epochs=dense_epochs,
                    notes="learning_curve",
                )
            )

    coverage_values = [0.5] if quick else [0.25, 0.50, 0.75]
    coverage_epochs = (10,) if quick else (50,)
    for seed, num_states, num_inputs, state_dim, coverage_fraction in product(
        seeds,
        num_states_list,
        num_inputs_list,
        state_dim_list,
        coverage_values,
    ):
        conditions.append(
            Condition(
                seed=seed,
                num_states=num_states,
                num_inputs=num_inputs,
                state_dim=state_dim,
                input_dim=input_dim,
                feature_mode="exact_pair",
                hidden_dim=None,
                training_mode="coverage_sweep",
                coverage_fraction=coverage_fraction,
                update_rule="delta",
                output_mode="dense",
                write_fraction=1.0,
                unwritten_mode="random_noise",
                epochs=coverage_epochs,
                notes="coverage_sanity_check",
            )
        )

    sparse_state_values = num_states_list if quick else [16, 32]
    sparse_input_values = num_inputs_list if quick else [4]
    sparse_dim_values = state_dim_list if quick else [256, 512]
    sparse_write_fractions = (
        [0.30, 0.50]
        if quick
        else [0.20, 0.30, 0.40, 0.50, 0.75, 1.00]
    )
    sparse_epochs = (10,) if quick else (50,)
    sparse_unwritten_modes: list[UnwrittenMode] = ["random_noise", "keep_current"]
    for seed, num_states, num_inputs, state_dim, write_fraction, unwritten_mode in product(
        seeds,
        sparse_state_values,
        sparse_input_values,
        sparse_dim_values,
        sparse_write_fractions,
        sparse_unwritten_modes,
    ):
        conditions.append(
            Condition(
                seed=seed,
                num_states=num_states,
                num_inputs=num_inputs,
                state_dim=state_dim,
                input_dim=input_dim,
                feature_mode="exact_pair",
                hidden_dim=None,
                training_mode="full_table",
                coverage_fraction=1.0,
                update_rule="hebbian",
                output_mode="sparse_topk",
                write_fraction=write_fraction,
                unwritten_mode=unwritten_mode,
                epochs=sparse_epochs,
                notes="sparse_learned_writer",
            )
        )
    return conditions


def run_experiment(quick: bool = False, jobs: int | None = 1) -> pd.DataFrame:
    conditions = _conditions(quick)
    print(
        f"Running experiment 6: learned transitions ({len(conditions)} training conditions, jobs={jobs})..."
    )
    nested_rows: list[list[Row]] = parallel_map(
        _run_condition,
        conditions,
        jobs=jobs,
        progress_label="Exp06 learned transitions",
    )
    rows: list[Row] = [row for condition_rows in nested_rows for row in condition_rows]
    df = pd.DataFrame(rows)
    csv_path = save_csv(df, "exp06_learned_transitions.csv")
    figure_paths: list[Path] = plot_learning_results(df)
    print(f"Saved CSV to {csv_path}")
    print(f"Saved figures to results/figures ({len(figure_paths)} files)")
    print(
        "Summary:\n"
        f"  mean learned transition accuracy = {df['all_transition_accuracy'].mean():.3f}"
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Worker processes; 0 uses all CPUs.",
    )
    args = parser.parse_args()
    run_experiment(quick=not args.full, jobs=args.jobs)


if __name__ == "__main__":
    main()
