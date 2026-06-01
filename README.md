# BioLogic State Machine Simulations

This repository contains reproducible simulations for BioLogic-style state machines built from high-dimensional bipolar state vectors. The experiments test exact finite-state transition construction, ideal nearest-attractor cleanup, Hopfield-style recurrent cleanup, sparse partial writes, a descriptor/payload protocol witness, learned transition acquisition from demonstrations, and structured grammar generalization.

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
- `biologic/learning.py`: local associative transition learning from state-input demonstrations.
- `biologic/grammars.py`: DFA-backed regular grammar tasks, including custom languages, Tomita grammars, and Reber grammar.
- `biologic/sequence_eval.py`: autonomous learned-state sequence rollout, string-level evaluation, and masked top-k cleanup.
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
| Exp06 | `experiments/exp06_learned_transitions.py` | Learns transition associations from demonstrated current-state, input, and next-state examples. |
| Exp07 | `experiments/exp07_structured_grammar_learning.py` | Trains on strings from structured regular grammars and tests length generalization. |

For Exp04, the two modes should not be averaged together. `keep_current` measures sparse writes against old-state inertia. `random_noise` measures whether partial target writes bias the system into the target basin.

Exp06 uses Interface-Conditioned Eligibility Transition Learning. The learner sees demonstrations of a current state plus input/descriptor followed by an observed next state, then updates pair-to-state weights using a local associative eligibility rule. The learner module does not accept an FSM or transition table; the experiment code uses the FSM only to generate demonstrations and evaluate the learned behavior.

Exp07 replaces random transition tables with structured finite-state languages. It trains from DFA traces of strings, then evaluates autonomous accept/reject behavior, internal state tracking, transition coverage, length generalization, and masked top-k sparse cleanup. This is the experiment to use when the question is rule reuse on unseen strings rather than memorization of arbitrary transition triples.

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
Exp07 caps its own worker pool at four processes to avoid excessive RAM use from sequence datasets when `--jobs 0` is used.
For harder Exp07-only runs, use `--advanced --jobs 0 --max-workers 0 --total-memory-gb <budget>` to use all cores while dividing a total memory budget across workers.

`run_all.py` runs Exp01 through Exp06. Exp06 can also be run directly when iterating on the learned-transition sweep:

```bash
python -m experiments.run_all --quick --jobs 0
python -m experiments.exp06_learned_transitions --quick --jobs 0
python -m experiments.exp06_learned_transitions --full --jobs 0
python -m experiments.exp07_structured_grammar_learning --quick --jobs 0
python -m experiments.exp07_structured_grammar_learning --task tomita --quick
python -m experiments.exp07_structured_grammar_learning --task custom --full --jobs 0
python -m experiments.exp07_structured_grammar_learning --advanced --jobs 0 --max-workers 0 --total-memory-gb 48
```

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

Current paper figure stems are `fig01_*` through `fig16_*`. Figures `fig08_*` through `fig11_*` come from Exp06:

```text
fig08_learned_transition_accuracy
fig09_pair_layer_capacity
fig10_learned_sparse_transitions
fig11_coverage_seen_unseen
```

Figures `fig12_*` through `fig16_*` come from Exp07:

```text
fig12_tomita_generalization
fig13_length_generalization
fig14_transition_coverage_vs_accuracy
fig15_topk_bound
fig16_seen_unseen_structured
```

Older unnumbered `fig_transition_*`, `fig_noise_*`, `fig_capacity_*`, `fig_sparse_*`, and `fig_descriptor_*` files are legacy outputs from an earlier plotting pass.

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
results/statistics/exp06_learned_transitions_summary.csv
results/statistics/exp07_structured_grammar_learning_summary.csv
results/statistics/exp07_structured_grammar_report.md
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
