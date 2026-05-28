# BioLogic State Machine Simulations

Reproducible Python simulations for BioLogic-style finite-state transition mechanisms, state-register cleanup, sparse basin-based transitions, and descriptor/payload separation.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run quick tests

```bash
pytest
python -m experiments.run_all --quick
```

Use multiple worker threads for the independent simulation conditions:

```bash
python -m experiments.run_all --quick --jobs 4
python -m experiments.run_all --full --jobs 0
```

`--jobs 0` uses all available CPUs. Each experiment also accepts `--jobs`.

## Run full experiments

```bash
python -m experiments.run_all --full
```

## Outputs

CSV files are written to:

```text
results/csv/
```

Figures are written to:

```text
results/figures/
```

The simulations use deterministic `numpy.random.default_rng(seed)` generators and standard scientific Python dependencies only.
