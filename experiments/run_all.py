"""Run all BioLogic simulation experiments."""

from __future__ import annotations

import argparse

from experiments import (
    exp01_transition_accuracy,
    exp02_noise_recovery,
    exp03_capacity,
    exp04_sparse_transitions,
    exp05_descriptor_payload,
    exp06_learned_transitions,
    exp07_structured_grammar_learning,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick", action="store_true", help="Run reduced settings in under a minute."
    )
    mode.add_argument("--full", action="store_true", help="Run publication settings.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Worker processes per experiment; 0 uses all CPUs.",
    )
    args = parser.parse_args()
    quick = not args.full

    exp01_transition_accuracy.run_experiment(quick=quick, jobs=args.jobs)
    exp02_noise_recovery.run_experiment(quick=quick, jobs=args.jobs)
    exp03_capacity.run_experiment(quick=quick, jobs=args.jobs)
    exp04_sparse_transitions.run_experiment(quick=quick, jobs=args.jobs)
    exp05_descriptor_payload.run_experiment(quick=quick, jobs=args.jobs)
    exp06_learned_transitions.run_experiment(quick=quick, jobs=args.jobs)
    exp07_structured_grammar_learning.run_experiment(quick=quick, jobs=args.jobs)


if __name__ == "__main__":
    main()
