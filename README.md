# BioLogic State Machine Simulations

This repository contains reproducible simulations for BioLogic-style state machines built from high-dimensional bipolar state vectors. The experiments test exact finite-state transition construction, ideal nearest-attractor cleanup, Hopfield-style recurrent cleanup, sparse partial writes, and a small descriptor/payload protocol witness.

The code is meant to support paper figures and Results-section numbers. Running the plotting and statistics scripts never reruns experiments; they only read the CSV files already present in `results/csv/`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Runtime dependencies are `numpy`, `pandas`, `matplotlib`, and `tqdm`. The dev extra adds `pytest` and `pandas-stubs`.

## Repository Layout

```text
biologic/      Core state-vector, register, transition, FSM, and descriptor code
experiments/   Reproducible experiment entry points and multiprocessing helpers
scripts/       Figure generation and statistics/report generation scripts
tests/         Unit tests for the core mechanics
results/csv/   Experiment CSV outputs consumed by plotting/statistics scripts
```

The most important core modules are:

- `biologic/encodings.py`: bipolar vector generation and overlap utilities.
- `biologic/fsm.py`: a small typed finite-state machine abstraction.
- `biologic/register.py`: nearest-attractor and Hopfield register cleanup.
- `biologic/transition.py`: exact and sparse transition mechanisms.
- `biologic/descriptors.py`: descriptor/payload protocol helpers.
- `biologic/metrics.py`: shared accuracy and recovery metrics.

The experiment modules are intentionally separate from the core logic. That keeps the paper sweeps, parameter grids, and CSV writing code out of the implementation being tested.

## Experiments

| Experiment | File | Purpose |
| --- | --- | --- |
| Exp01 | `experiments/exp01_transition_accuracy.py` | Tests exact transition realization under nearest and Hopfield cleanup. |
| Exp02 | `experiments/exp02_noise_recovery.py` | Measures recovery from bit-flip corruption. |
| Exp03 | `experiments/exp03_capacity.py` | Sweeps capacity ratio and compares Hopfield behavior to the classical capacity reference. |
| Exp04 | `experiments/exp04_sparse_transitions.py` | Tests sparse transition writes under `keep_current` and `random_noise` modes. |
| Exp05 | `experiments/exp05_descriptor_payload.py` | Demonstrates descriptor/payload separation as an executable protocol witness. |

For Exp04, the two modes should not be averaged together. `keep_current` measures sparse writes against old-state inertia. `random_noise` measures whether partial target writes bias the system into the target basin.

## Running Tests

```bash
pytest
```

## Running Experiments

Quick smoke run:

```bash
python -m experiments.run_all --quick --jobs 4
```

Full run using all available CPU cores:

```bash
python -m experiments.run_all --full --jobs 0
```

`--jobs 0` uses all CPUs. Each individual experiment also accepts `--jobs`. The parallelism is process-based because the experiment conditions are independent.

CSV outputs are written to:

```text
results/csv/
```

## Generating Figures

The plotting script reads existing CSV files from `results/csv/` and writes paper-quality Matplotlib figures. It does not rerun simulations.

```bash
python scripts/plot_results.py
python scripts/plot_results.py --format pdf
python scripts/plot_results.py --format png
python scripts/plot_results.py --format both
```

The default is `--format both`. Figures are written to:

```text
results/figures/
```

The plotting script also writes:

```text
results/tables/simulation_summary.tex
```

Current paper figure stems are `fig01_*` through `fig07_*`. Older unnumbered `fig_transition_*`, `fig_noise_*`, `fig_capacity_*`, `fig_sparse_*`, and `fig_descriptor_*` files are legacy outputs from an earlier plotting pass.

## Generating Statistics

The statistics script reads existing CSV files and produces summaries for the paper Results section. It does not rerun simulations.

```bash
python scripts/statistics.py
python scripts/statistics.py --csv-dir results/csv --out-dir results/statistics
python scripts/statistics.py --paper-only
```

`--paper-only` prints the extracted paper-ready values to stdout while still writing the report files.

Statistics outputs:

```text
results/statistics/statistics_report.md
results/statistics/statistics_report.txt
results/statistics/exp01_transition_accuracy_summary.csv
results/statistics/exp02_noise_recovery_summary.csv
results/statistics/exp03_capacity_summary.csv
results/statistics/exp04_sparse_transitions_summary.csv
results/statistics/exp05_descriptor_payload_summary.csv
results/tables/simulation_statistics_table.tex
```

## Reproducibility Notes

The simulations use deterministic `numpy.random.default_rng(seed)` generators. Experiment scripts write CSVs; plotting and statistics scripts treat those CSVs as fixed inputs. This separation makes it possible to regenerate paper figures and tables without accidentally changing the underlying simulation data.

If a CSV is missing, the plotting and statistics scripts print a warning and continue with the files that are available.

## Typical Paper Workflow

```bash
pytest
python -m experiments.run_all --full --jobs 0
python scripts/plot_results.py --format both
python scripts/statistics.py
python scripts/statistics.py --paper-only
```

Use the generated figures in `results/figures/`, the LaTeX tables in `results/tables/`, and the narrative/statistical details in `results/statistics/statistics_report.md`.
