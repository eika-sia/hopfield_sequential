# BioLogic State Machine Simulations

This repository contains reproducible simulations for BioLogic-style finite-state machines built from high-dimensional bipolar state vectors. The code separates core mechanisms from experiment sweeps:

- core modules in `biologic/` implement state encodings, finite-state machines, cleanup registers, transition writers, descriptor/payload logic, associative learning, and DFA-backed grammar tasks;
- experiment modules in `experiments/` generate CSV outputs under `results/csv/`;
- scripts in `scripts/` read existing CSVs to generate figures, tables, and statistics reports.

Plotting and statistics scripts do not rerun simulations. They treat `results/csv/` as fixed input.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Runtime dependencies are `numpy`, `pandas`, `matplotlib`, and `tqdm`. The dev extra adds `pytest` and `pandas-stubs`.

## Repository Layout

```text
biologic/      Core implementation
experiments/   Reproducible experiment entry points
scripts/       Plotting and statistics scripts
tests/         Unit tests
results/csv/   Experiment CSV outputs
results/figures/ Generated PDF/PNG figures
results/tables/  Generated LaTeX tables
results/statistics/ Generated reports and summary CSVs
```

Important core modules:

| Module | Role |
| --- | --- |
| `biologic/encodings.py` | Bipolar codebook generation and overlap utilities. |
| `biologic/fsm.py` | Typed finite-state machine abstraction. |
| `biologic/register.py` | Nearest-attractor and Hopfield cleanup. |
| `biologic/transition.py` | Constructed exact and sparse transition mechanisms. |
| `biologic/learning.py` | Local associative transition learning from state-input demonstrations. |
| `biologic/grammars.py` | DFA-backed structured grammar tasks: custom/MLReg-style tasks, Tomita grammars, and Reber grammar. |
| `biologic/sequence_eval.py` | Autonomous learned-state rollout, string-level evaluation, and masked top-k cleanup. |
| `biologic/descriptors.py` | Descriptor/payload protocol witness helpers. |
| `biologic/metrics.py` | Shared accuracy and recovery metrics. |

## Experiments

| Experiment | Module | Main output CSV |
| --- | --- | --- |
| Exp01 transition accuracy | `experiments.exp01_transition_accuracy` | `results/csv/exp01_transition_accuracy.csv` |
| Exp02 noise recovery | `experiments.exp02_noise_recovery` | `results/csv/exp02_noise_recovery.csv` |
| Exp03 capacity sweep | `experiments.exp03_capacity` | `results/csv/exp03_capacity.csv` |
| Exp04 sparse transitions | `experiments.exp04_sparse_transitions` | `results/csv/exp04_sparse_transitions.csv` |
| Exp05 descriptor/payload witness | `experiments.exp05_descriptor_payload` | `results/csv/exp05_descriptor_payload.csv` |
| Exp06 learned transitions | `experiments.exp06_learned_transitions` | `results/csv/exp06_learned_transitions.csv` |
| Exp07 structured grammar learning | `experiments.exp07_structured_grammar_learning` | `results/csv/exp07_structured_grammar_learning.csv` |

## Running Tests

```bash
pytest
```

For the grammar and learning code only:

```bash
pytest tests/test_learning.py tests/test_grammars.py tests/test_exp07_sequence_learning.py
```

## Running Experiments

Run all experiments:

```bash
python -m experiments.run_all --quick --jobs 4
python -m experiments.run_all --full --jobs 0
```

`--jobs 0` uses all available CPU cores. Experiment conditions are independent, so parallelism is process-based.

Individual experiment entry points use the descriptive module names shown in the experiment table. Exp01-Exp05 run full settings by default and accept `--quick` for reduced settings:

```bash
python -m experiments.exp01_transition_accuracy --quick --jobs 0
python -m experiments.exp01_transition_accuracy --jobs 0

python -m experiments.exp02_noise_recovery --quick --jobs 0
python -m experiments.exp03_capacity --quick --jobs 0
python -m experiments.exp04_sparse_transitions --quick --jobs 0
python -m experiments.exp05_descriptor_payload --quick --jobs 0
```

The larger learned-transition and grammar-learning modules use explicit quick/full modes:

```bash
python -m experiments.exp06_learned_transitions --quick --jobs 0
python -m experiments.exp06_learned_transitions --full --jobs 0

python -m experiments.exp07_structured_grammar_learning --quick --jobs 0
python -m experiments.exp07_structured_grammar_learning --full --jobs 0
python -m experiments.exp07_structured_grammar_learning --full --jobs 0 --continue
```

All experiment runners print progress with completed conditions, percent, throughput, and ETA.
For Exp07, `--continue` resumes from the existing CSV by skipping completed main rows and appending the missing conditions.

## Background Runs

For long SSH sessions, use the systemd wrapper instead of relying on an open terminal:

```bash
scripts/run_exp07_full_background.sh
systemctl --user status exp07-full.service
tail -f results/logs/exp07_full.log
```

Stop the service:

```bash
systemctl --user stop exp07-full.service
```

The background wrapper writes the same Exp07 CSV as the foreground command:

```text
results/csv/exp07_structured_grammar_learning.csv
```

## Generating Figures

The plotting script reads existing CSVs and writes both paper-quality PDFs and PNGs by default:

```bash
python scripts/plot_results.py
python scripts/plot_results.py --format pdf
python scripts/plot_results.py --format png
python scripts/plot_results.py --format both
```

Outputs:

```text
results/figures/
results/tables/simulation_summary.tex
```

The numbered paper figure stems are `fig01_*` through `fig16_*`. Older unnumbered `fig_transition_*`, `fig_noise_*`, `fig_capacity_*`, `fig_sparse_*`, and `fig_descriptor_*` files are legacy outputs from an earlier plotting pass.

## Generating Statistics

The statistics script reads existing CSVs and writes paper-facing summaries. It does not rerun experiments.

```bash
python scripts/statistics.py
python scripts/statistics.py --csv-dir results/csv --out-dir results/statistics
python scripts/statistics.py --paper-only
```

`--paper-only` prints the extracted paper-ready values to stdout while still writing report files.

Main outputs:

```text
results/statistics/statistics_report.md
results/statistics/statistics_report.txt
results/statistics/exp01_transition_accuracy_summary.csv
results/statistics/exp02_noise_recovery_summary.csv
results/statistics/exp03_capacity_summary.csv
results/statistics/exp04_sparse_transitions_summary.csv
results/statistics/exp05_descriptor_payload_summary.csv
results/statistics/exp06_learned_transitions_summary.csv
results/statistics/exp07_structured_grammar_learning_summary.csv
results/statistics/exp07_structured_grammar_report.md
results/tables/simulation_statistics_table.tex
```

## Reproducibility

Experiments use deterministic `numpy.random.default_rng(seed)` generators. The CSVs are the boundary between simulation and paper artifacts: regenerate CSVs when changing experiments, then regenerate figures/statistics from those CSVs.

Plotting and statistics scripts warn and continue if an expected CSV is missing.

## Typical Paper Workflow

```bash
pytest
python -m experiments.run_all --full --jobs 0
python scripts/plot_results.py --format both
python scripts/statistics.py
python scripts/statistics.py --paper-only
```

Use:

```text
results/figures/      PDF/PNG figures
results/tables/       LaTeX tables
results/statistics/   Markdown/TXT reports and summary CSVs
```
