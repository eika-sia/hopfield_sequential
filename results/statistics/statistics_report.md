# Simulation Statistics Report

## Data files

Loaded CSVs:
- `results/csv/exp01_transition_accuracy.csv`
- `results/csv/exp02_noise_recovery.csv`
- `results/csv/exp03_capacity.csv`
- `results/csv/exp04_sparse_transitions.csv`
- `results/csv/exp05_descriptor_payload.csv`
- `results/csv/exp06_learned_transitions.csv`
- `results/csv/exp07_structured_grammar_learning.csv`

## Exp01: Transition realization

- Nearest-attractor cleanup tests the exact transition construction under ideal basin decoding.
- Hopfield cleanup tests a concrete recurrent implementation.
- Hopfield failures concentrated in low-dimensional, high-load regimes.
- Nearest correctness check: passed (0 failing rows).
- Nearest transition accuracy: mean=1.000, std=0.000, min=1.000, max=1.000.
- Hopfield transition accuracy: mean=0.874, std=0.222, median=1.000, min=0.141, max=1.000.
- Hopfield mean accuracy at state_dim=64,num_states=64: 0.319.
- Hopfield mean accuracy at state_dim=512,num_states=64: 0.999.
- Hopfield mean accuracy at state_dim=64,num_states=4: 1.000.

### Overall summary by register type

| register_type | count | transition_accuracy_mean | transition_accuracy_std | transition_accuracy_sem | transition_accuracy_min | transition_accuracy_median | transition_accuracy_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hopfield | 600.000 | 0.874 | 0.222 | 0.009 | 0.141 | 1.000 | 1.000 |
| nearest | 600.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 |

### Hopfield degradation pivot

| state_dim | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- |
| 64.000 | 1.000 | 0.990 | 0.810 | 0.455 | 0.319 |
| 128.000 | 1.000 | 1.000 | 0.988 | 0.743 | 0.493 |
| 256.000 | 1.000 | 1.000 | 1.000 | 0.998 | 0.690 |
| 512.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 |

### Worst Hopfield rows

| seed | state_dim | num_states | num_inputs | transition_accuracy | mean_abs_overlap |
| --- | --- | --- | --- | --- | --- |
| 9.000 | 64.000 | 32.000 | 8.000 | 0.141 | 0.099 |
| 9.000 | 64.000 | 64.000 | 2.000 | 0.156 | 0.098 |
| 4.000 | 64.000 | 64.000 | 4.000 | 0.168 | 0.098 |
| 9.000 | 64.000 | 32.000 | 4.000 | 0.219 | 0.099 |
| 4.000 | 64.000 | 64.000 | 2.000 | 0.234 | 0.099 |
| 5.000 | 64.000 | 64.000 | 2.000 | 0.250 | 0.099 |
| 7.000 | 64.000 | 64.000 | 4.000 | 0.254 | 0.103 |
| 0.000 | 64.000 | 64.000 | 8.000 | 0.266 | 0.101 |
| 6.000 | 64.000 | 64.000 | 8.000 | 0.277 | 0.104 |
| 0.000 | 64.000 | 64.000 | 4.000 | 0.281 | 0.101 |
| 1.000 | 64.000 | 32.000 | 2.000 | 0.281 | 0.103 |
| 2.000 | 64.000 | 64.000 | 2.000 | 0.281 | 0.101 |
| 4.000 | 64.000 | 64.000 | 8.000 | 0.281 | 0.098 |
| 6.000 | 64.000 | 64.000 | 2.000 | 0.281 | 0.102 |
| 2.000 | 128.000 | 64.000 | 2.000 | 0.289 | 0.071 |
| 1.000 | 64.000 | 64.000 | 8.000 | 0.293 | 0.098 |
| 3.000 | 128.000 | 64.000 | 4.000 | 0.297 | 0.071 |
| 5.000 | 64.000 | 64.000 | 8.000 | 0.303 | 0.099 |
| 9.000 | 64.000 | 64.000 | 4.000 | 0.309 | 0.098 |
| 1.000 | 64.000 | 64.000 | 2.000 | 0.312 | 0.100 |

## Exp02: Noise recovery

- Nearest cleanup shows ideal basin robustness.
- Hopfield cleanup combines baseline attractor stability and robustness to corruption.
- Hopfield recovery below 1.0 at zero noise indicates that some stored patterns are unstable under recurrent Hebbian dynamics in high-load regimes.
- The flip-fraction summaries keep nearest and Hopfield separate to avoid hiding overloaded regimes.
- Nearest recovery at 0.00: 1.000.
- Nearest recovery at 0.20: 1.000.
- Nearest recovery at 0.30: 0.996.
- Nearest recovery at 0.40: 0.752.
- Hopfield recovery at 0.00: 0.844.
- Hopfield recovery at 0.20: 0.753.
- Hopfield recovery at 0.30: 0.639.
- Hopfield recovery at 0.40: 0.389.
- Hopfield recovery for state_dim=64,num_states=64: 0.158.
- Hopfield recovery for state_dim=512,num_states=64: 0.829.

### Overall summary by register type

| register_type | count | recovery_accuracy_mean | recovery_accuracy_std | recovery_accuracy_sem | recovery_accuracy_min | recovery_accuracy_median | recovery_accuracy_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hopfield | 1440.000 | 0.699 | 0.338 | 0.009 | 0.017 | 0.883 | 1.000 |
| nearest | 1440.000 | 0.967 | 0.136 | 0.004 | 0.010 | 1.000 | 1.000 |

### Noise degradation

| register_type | recovery_at_0.00 | actual_flip_for_0.00 | recovery_at_0.20 | actual_flip_for_0.20 | recovery_at_0.30 | actual_flip_for_0.30 | recovery_at_0.40 | actual_flip_for_0.40 | degradation_0_to_40 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hopfield | 0.844 | 0.000 | 0.753 | 0.200 | 0.639 | 0.300 | 0.389 | 0.400 | 0.455 |
| nearest | 1.000 | 0.000 | 1.000 | 0.200 | 0.996 | 0.300 | 0.752 | 0.400 | 0.248 |

### Hopfield zero-noise recovery by state_dim,num_states

| register_type | state_dim | num_states | count | recovery_accuracy_mean | recovery_accuracy_std | recovery_accuracy_sem | recovery_accuracy_min | recovery_accuracy_median | recovery_accuracy_max | recovery_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hopfield | 64.000 | 8.000 | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hopfield | 64.000 | 16.000 | 10.000 | 0.856 | 0.114 | 0.036 | 0.625 | 0.875 | 1.000 | 0.856 |
| hopfield | 64.000 | 32.000 | 10.000 | 0.484 | 0.196 | 0.062 | 0.156 | 0.516 | 0.781 | 0.484 |
| hopfield | 64.000 | 64.000 | 10.000 | 0.305 | 0.068 | 0.022 | 0.219 | 0.297 | 0.422 | 0.305 |
| hopfield | 128.000 | 8.000 | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hopfield | 128.000 | 16.000 | 10.000 | 0.981 | 0.059 | 0.019 | 0.812 | 1.000 | 1.000 | 0.981 |
| hopfield | 128.000 | 32.000 | 10.000 | 0.706 | 0.157 | 0.050 | 0.406 | 0.719 | 0.969 | 0.706 |
| hopfield | 128.000 | 64.000 | 10.000 | 0.483 | 0.064 | 0.020 | 0.406 | 0.477 | 0.594 | 0.483 |
| hopfield | 256.000 | 8.000 | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hopfield | 256.000 | 16.000 | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hopfield | 256.000 | 32.000 | 10.000 | 0.997 | 0.010 | 0.003 | 0.969 | 1.000 | 1.000 | 0.997 |
| hopfield | 256.000 | 64.000 | 10.000 | 0.697 | 0.141 | 0.044 | 0.500 | 0.711 | 0.922 | 0.697 |
| hopfield | 512.000 | 8.000 | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hopfield | 512.000 | 16.000 | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hopfield | 512.000 | 32.000 | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hopfield | 512.000 | 64.000 | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### Worst recovery rows

| seed | register_type | state_dim | num_states | flip_fraction | recovery_accuracy |
| --- | --- | --- | --- | --- | --- |
| 8.000 | nearest | 64.000 | 64.000 | 0.400 | 0.010 |
| 9.000 | nearest | 64.000 | 64.000 | 0.400 | 0.010 |
| 6.000 | nearest | 64.000 | 64.000 | 0.400 | 0.012 |
| 0.000 | nearest | 64.000 | 64.000 | 0.400 | 0.013 |
| 3.000 | nearest | 64.000 | 64.000 | 0.400 | 0.013 |
| 4.000 | nearest | 64.000 | 64.000 | 0.400 | 0.013 |
| 1.000 | nearest | 64.000 | 64.000 | 0.400 | 0.014 |
| 2.000 | nearest | 64.000 | 64.000 | 0.400 | 0.016 |
| 7.000 | nearest | 64.000 | 64.000 | 0.400 | 0.016 |
| 5.000 | nearest | 64.000 | 64.000 | 0.400 | 0.017 |
| 8.000 | hopfield | 64.000 | 64.000 | 0.400 | 0.017 |
| 9.000 | hopfield | 64.000 | 64.000 | 0.400 | 0.027 |
| 4.000 | hopfield | 64.000 | 64.000 | 0.400 | 0.034 |
| 6.000 | hopfield | 64.000 | 64.000 | 0.400 | 0.034 |
| 7.000 | hopfield | 64.000 | 64.000 | 0.400 | 0.034 |
| 2.000 | hopfield | 64.000 | 64.000 | 0.400 | 0.037 |
| 0.000 | hopfield | 64.000 | 64.000 | 0.400 | 0.040 |
| 1.000 | hopfield | 64.000 | 64.000 | 0.400 | 0.040 |
| 3.000 | hopfield | 64.000 | 64.000 | 0.400 | 0.040 |
| 5.000 | hopfield | 64.000 | 64.000 | 0.400 | 0.044 |

## Exp03: Capacity limits

- This is the clearest capacity-limit experiment.
- Nearest cleanup is an ideal control and remains perfect.
- Hopfield recovery is high below/near classical capacity and degrades above it.
- This motivates modular state machines instead of monolithic registers.
- Nearest mean recovery: 1.000.
- Hopfield mean recovery: 0.777.
- Hopfield recovery near capacity_ratio=0.138: 0.985.
- Hopfield recovery at capacity_ratio=1.000: 0.466.
- First capacity ratio where Hopfield mean drops below 0.90: 0.200.
- First capacity ratio where Hopfield mean drops below 0.75: 0.300.

### Hopfield recovery by capacity ratio

| capacity_ratio | recovery_accuracy |
| --- | --- |
| 0.025 | 1.000 |
| 0.050 | 1.000 |
| 0.100 | 1.000 |
| 0.138 | 0.985 |
| 0.200 | 0.824 |
| 0.300 | 0.656 |
| 0.500 | 0.567 |
| 0.750 | 0.494 |
| 1.000 | 0.466 |

### Nearest recovery by capacity ratio

| capacity_ratio | recovery_accuracy |
| --- | --- |
| 0.025 | 1.000 |
| 0.050 | 1.000 |
| 0.100 | 1.000 |
| 0.138 | 1.000 |
| 0.200 | 1.000 |
| 0.300 | 1.000 |
| 0.500 | 1.000 |
| 0.750 | 1.000 |
| 1.000 | 1.000 |

### Classical capacity comparison

| target_capacity_ratio | actual_capacity_ratio | hopfield_recovery_accuracy | used_nearest_available |
| --- | --- | --- | --- |
| 0.025 | 0.025 | 1.000 | 0.000 |
| 0.050 | 0.050 | 1.000 | 0.000 |
| 0.100 | 0.100 | 1.000 | 0.000 |
| 0.138 | 0.138 | 0.985 | 0.000 |
| 0.200 | 0.200 | 0.824 | 0.000 |
| 0.300 | 0.300 | 0.656 | 0.000 |
| 0.500 | 0.500 | 0.567 | 0.000 |
| 0.750 | 0.750 | 0.494 | 0.000 |
| 1.000 | 1.000 | 0.466 | 0.000 |

### Threshold estimates

| threshold | first_capacity_ratio_below_threshold |
| --- | --- |
| 0.990 | 0.138 |
| 0.950 | 0.200 |
| 0.900 | 0.200 |
| 0.750 | 0.300 |
| 0.500 | 0.750 |

### Worst Hopfield capacity rows

| seed | state_dim | num_states | capacity_ratio | recovery_accuracy | mean_abs_overlap |
| --- | --- | --- | --- | --- | --- |
| 9.000 | 64.000 | 48.000 | 0.750 | 0.139 | 0.100 |
| 9.000 | 64.000 | 64.000 | 1.000 | 0.145 | 0.099 |
| 9.000 | 64.000 | 32.000 | 0.500 | 0.163 | 0.102 |
| 4.000 | 64.000 | 64.000 | 1.000 | 0.165 | 0.099 |
| 8.000 | 128.000 | 128.000 | 1.000 | 0.173 | 0.071 |
| 1.000 | 64.000 | 64.000 | 1.000 | 0.196 | 0.100 |
| 2.000 | 64.000 | 64.000 | 1.000 | 0.209 | 0.101 |
| 3.000 | 128.000 | 96.000 | 0.750 | 0.209 | 0.071 |
| 1.000 | 64.000 | 32.000 | 0.500 | 0.211 | 0.104 |
| 6.000 | 64.000 | 32.000 | 0.500 | 0.214 | 0.101 |
| 2.000 | 64.000 | 48.000 | 0.750 | 0.215 | 0.105 |
| 6.000 | 64.000 | 64.000 | 1.000 | 0.221 | 0.102 |
| 9.000 | 128.000 | 128.000 | 1.000 | 0.222 | 0.070 |
| 7.000 | 64.000 | 64.000 | 1.000 | 0.230 | 0.104 |
| 3.000 | 128.000 | 128.000 | 1.000 | 0.231 | 0.070 |
| 5.000 | 64.000 | 64.000 | 1.000 | 0.236 | 0.099 |
| 9.000 | 128.000 | 96.000 | 0.750 | 0.240 | 0.070 |
| 8.000 | 64.000 | 64.000 | 1.000 | 0.252 | 0.099 |
| 1.000 | 64.000 | 48.000 | 0.750 | 0.256 | 0.100 |
| 0.000 | 64.000 | 64.000 | 1.000 | 0.259 | 0.101 |

## Exp04: Sparse transitions

- Do not interpret aggregate mean alone.
- keep_current and random_noise answer different questions.
- keep_current measures sparse transition against old-state inertia.
- random_noise measures whether partial target writes can bias the system into the target basin.
- Sparse transitions work well in random_noise mode once enough target coordinates are written.
- keep_current shows that reset/gating/inhibition may be needed before sparse writes.

### Summary by register type and mode

| register_type | mode | count | sparse_transition_accuracy_mean | sparse_transition_accuracy_std | sparse_transition_accuracy_sem | sparse_transition_accuracy_min | sparse_transition_accuracy_median | sparse_transition_accuracy_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hopfield | keep_current | 810.000 | 0.315 | 0.367 | 0.013 | 0.008 | 0.125 | 1.000 |
| hopfield | random_noise | 810.000 | 0.654 | 0.355 | 0.012 | 0.016 | 0.781 | 1.000 |
| nearest | keep_current | 810.000 | 0.334 | 0.386 | 0.014 | 0.008 | 0.125 | 1.000 |
| nearest | random_noise | 810.000 | 0.810 | 0.287 | 0.010 | 0.055 | 1.000 | 1.000 |

### random_noise sparse accuracy by write fraction

| register_type | mode | target_write_fraction | actual_write_fraction | sparse_transition_accuracy |
| --- | --- | --- | --- | --- |
| nearest | random_noise | 0.100 | 0.100 | 0.488 |
| nearest | random_noise | 0.200 | 0.200 | 0.877 |
| nearest | random_noise | 0.300 | 0.300 | 0.982 |
| nearest | random_noise | 0.400 | 0.400 | 1.000 |
| nearest | random_noise | 0.500 | 0.500 | 1.000 |
| hopfield | random_noise | 0.100 | 0.100 | 0.277 |
| hopfield | random_noise | 0.200 | 0.200 | 0.575 |
| hopfield | random_noise | 0.300 | 0.300 | 0.762 |
| hopfield | random_noise | 0.400 | 0.400 | 0.850 |
| hopfield | random_noise | 0.500 | 0.500 | 0.906 |

### keep_current sparse accuracy at selected write fractions

| register_type | mode | target_write_fraction | actual_write_fraction | sparse_transition_accuracy |
| --- | --- | --- | --- | --- |
| nearest | keep_current | 0.500 | 0.500 | 0.540 |
| nearest | keep_current | 0.750 | 0.750 | 1.000 |
| nearest | keep_current | 1.000 | 1.000 | 1.000 |
| hopfield | keep_current | 0.500 | 0.500 | 0.408 |
| hopfield | keep_current | 0.750 | 0.750 | 0.949 |
| hopfield | keep_current | 1.000 | 1.000 | 0.970 |

### Threshold estimates

| register_type | mode | threshold | first_write_fraction_reaching_threshold |
| --- | --- | --- | --- |
| hopfield | keep_current | 0.500 | 0.750 |
| hopfield | keep_current | 0.750 | 0.750 |
| hopfield | keep_current | 0.900 | 0.750 |
| hopfield | keep_current | 0.950 | 1.000 |
| hopfield | keep_current | 0.990 | not reached |
| hopfield | random_noise | 0.500 | 0.200 |
| hopfield | random_noise | 0.750 | 0.300 |
| hopfield | random_noise | 0.900 | 0.500 |
| hopfield | random_noise | 0.950 | 0.750 |
| hopfield | random_noise | 0.990 | not reached |
| nearest | keep_current | 0.500 | 0.500 |
| nearest | keep_current | 0.750 | 0.750 |
| nearest | keep_current | 0.900 | 0.750 |
| nearest | keep_current | 0.950 | 0.750 |
| nearest | keep_current | 0.990 | 0.750 |
| nearest | random_noise | 0.500 | 0.150 |
| nearest | random_noise | 0.750 | 0.200 |
| nearest | random_noise | 0.900 | 0.300 |
| nearest | random_noise | 0.950 | 0.300 |
| nearest | random_noise | 0.990 | 0.400 |

### random_noise minus keep_current

| register_type | write_fraction | random_noise_mean | keep_current_mean | random_noise_minus_keep_current |
| --- | --- | --- | --- | --- |
| hopfield | 0.050 | 0.152 | 0.077 | 0.075 |
| hopfield | 0.100 | 0.277 | 0.078 | 0.199 |
| hopfield | 0.150 | 0.442 | 0.078 | 0.365 |
| hopfield | 0.200 | 0.575 | 0.079 | 0.496 |
| hopfield | 0.300 | 0.762 | 0.083 | 0.679 |
| hopfield | 0.400 | 0.850 | 0.114 | 0.736 |
| hopfield | 0.500 | 0.906 | 0.408 | 0.498 |
| hopfield | 0.750 | 0.955 | 0.949 | 0.006 |
| hopfield | 1.000 | 0.970 | 0.970 | 0.000 |
| nearest | 0.050 | 0.235 | 0.078 | 0.157 |
| nearest | 0.100 | 0.488 | 0.078 | 0.410 |
| nearest | 0.150 | 0.706 | 0.078 | 0.629 |
| nearest | 0.200 | 0.877 | 0.078 | 0.799 |
| nearest | 0.300 | 0.982 | 0.078 | 0.904 |
| nearest | 0.400 | 1.000 | 0.081 | 0.918 |
| nearest | 0.500 | 1.000 | 0.540 | 0.460 |
| nearest | 0.750 | 1.000 | 1.000 | 0.000 |
| nearest | 1.000 | 1.000 | 1.000 | 0.000 |

### Worst sparse-transition rows

| seed | register_type | mode | state_dim | num_states | write_fraction | sparse_transition_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 9.000 | nearest | keep_current | 128.000 | 32.000 | 0.050 | 0.008 |
| 9.000 | nearest | keep_current | 128.000 | 32.000 | 0.100 | 0.008 |
| 9.000 | nearest | keep_current | 128.000 | 32.000 | 0.150 | 0.008 |
| 9.000 | nearest | keep_current | 128.000 | 32.000 | 0.200 | 0.008 |
| 9.000 | nearest | keep_current | 128.000 | 32.000 | 0.300 | 0.008 |
| 9.000 | nearest | keep_current | 128.000 | 32.000 | 0.400 | 0.008 |
| 9.000 | hopfield | keep_current | 128.000 | 32.000 | 0.050 | 0.008 |
| 9.000 | hopfield | keep_current | 128.000 | 32.000 | 0.150 | 0.008 |
| 9.000 | nearest | keep_current | 256.000 | 32.000 | 0.050 | 0.008 |
| 9.000 | nearest | keep_current | 256.000 | 32.000 | 0.100 | 0.008 |
| 9.000 | nearest | keep_current | 256.000 | 32.000 | 0.150 | 0.008 |
| 9.000 | nearest | keep_current | 256.000 | 32.000 | 0.200 | 0.008 |
| 9.000 | nearest | keep_current | 256.000 | 32.000 | 0.300 | 0.008 |
| 9.000 | nearest | keep_current | 256.000 | 32.000 | 0.400 | 0.008 |
| 9.000 | hopfield | keep_current | 256.000 | 32.000 | 0.050 | 0.008 |
| 9.000 | hopfield | keep_current | 256.000 | 32.000 | 0.100 | 0.008 |
| 9.000 | hopfield | keep_current | 256.000 | 32.000 | 0.150 | 0.008 |
| 9.000 | hopfield | keep_current | 256.000 | 32.000 | 0.200 | 0.008 |
| 9.000 | hopfield | keep_current | 256.000 | 32.000 | 0.300 | 0.008 |
| 9.000 | nearest | keep_current | 512.000 | 32.000 | 0.050 | 0.008 |

### Best rows with write_fraction < 1.0

| seed | register_type | mode | state_dim | num_states | write_fraction | sparse_transition_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 0.000 | nearest | keep_current | 128.000 | 8.000 | 0.750 | 1.000 |
| 0.000 | nearest | random_noise | 128.000 | 8.000 | 0.400 | 1.000 |
| 0.000 | nearest | random_noise | 128.000 | 8.000 | 0.500 | 1.000 |
| 0.000 | nearest | random_noise | 128.000 | 8.000 | 0.750 | 1.000 |
| 0.000 | hopfield | keep_current | 128.000 | 8.000 | 0.750 | 1.000 |
| 0.000 | hopfield | random_noise | 128.000 | 8.000 | 0.750 | 1.000 |
| 0.000 | nearest | keep_current | 256.000 | 8.000 | 0.750 | 1.000 |
| 0.000 | nearest | random_noise | 256.000 | 8.000 | 0.300 | 1.000 |
| 0.000 | nearest | random_noise | 256.000 | 8.000 | 0.400 | 1.000 |
| 0.000 | nearest | random_noise | 256.000 | 8.000 | 0.500 | 1.000 |
| 0.000 | nearest | random_noise | 256.000 | 8.000 | 0.750 | 1.000 |
| 0.000 | hopfield | keep_current | 256.000 | 8.000 | 0.750 | 1.000 |
| 0.000 | hopfield | random_noise | 256.000 | 8.000 | 0.400 | 1.000 |
| 0.000 | hopfield | random_noise | 256.000 | 8.000 | 0.500 | 1.000 |
| 0.000 | hopfield | random_noise | 256.000 | 8.000 | 0.750 | 1.000 |
| 0.000 | nearest | keep_current | 512.000 | 8.000 | 0.750 | 1.000 |
| 0.000 | nearest | random_noise | 512.000 | 8.000 | 0.150 | 1.000 |
| 0.000 | nearest | random_noise | 512.000 | 8.000 | 0.200 | 1.000 |
| 0.000 | nearest | random_noise | 512.000 | 8.000 | 0.300 | 1.000 |
| 0.000 | nearest | random_noise | 512.000 | 8.000 | 0.400 | 1.000 |

## Exp05: Descriptor/payload separation

- This is an executable witness, not a performance benchmark.
- It demonstrates that the same payload can produce different effects under different descriptors.
- It demonstrates that payload without a descriptor is invalid/error-state producing.
- Descriptor corruption affects operation-level behavior.
- Protocol success across non-NaN success rows: 1.000.
- All non-NaN success values true: yes.

### Protocol case success

| case_name | count | success_mean |
| --- | --- | --- |
| missing_descriptor | 10.000 | 1.000 |
| same_descriptor_store_p1 | 10.000 | 1.000 |
| same_descriptor_store_p2 | 10.000 | 1.000 |
| same_payload_compare | 10.000 | 1.000 |
| same_payload_store | 10.000 | 1.000 |
| wrong_content_compare | 10.000 | 1.000 |

### Descriptor corruption by rate

| corruption_rate | operation_success_rate_count | operation_success_rate_mean | operation_success_rate_std | operation_success_rate_sem | content_success_rate_count | content_success_rate_mean | content_success_rate_std | content_success_rate_sem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000 | 10.000 | 1.000 | 0.000 | 0.000 | 10.000 | 1.000 | 0.000 | 0.000 |
| 0.050 | 10.000 | 0.950 | 0.014 | 0.004 | 10.000 | 0.950 | 0.014 | 0.004 |
| 0.100 | 10.000 | 0.902 | 0.009 | 0.003 | 10.000 | 0.902 | 0.009 | 0.003 |
| 0.200 | 10.000 | 0.805 | 0.025 | 0.008 | 10.000 | 0.805 | 0.025 | 0.008 |
| 0.300 | 10.000 | 0.697 | 0.027 | 0.008 | 10.000 | 0.697 | 0.027 | 0.008 |
| 0.500 | 10.000 | 0.498 | 0.018 | 0.006 | 10.000 | 0.498 | 0.018 | 0.006 |
| 0.750 | 10.000 | 0.250 | 0.016 | 0.005 | 10.000 | 0.250 | 0.016 | 0.005 |
| 1.000 | 10.000 | 0.000 | 0.000 | 0.000 | 10.000 | 0.000 | 0.000 | 0.000 |

### Descriptor corruption by descriptor and rate

| descriptor | corruption_rate | operation_success_rate_count | operation_success_rate_mean | operation_success_rate_std | operation_success_rate_sem | content_success_rate_count | content_success_rate_mean | content_success_rate_std | content_success_rate_sem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COMPARE | 0.000 | 10.000 | 1.000 | 0.000 | 0.000 | 10.000 | 1.000 | 0.000 | 0.000 |
| COMPARE | 0.050 | 10.000 | 0.950 | 0.014 | 0.004 | 10.000 | 0.950 | 0.014 | 0.004 |
| COMPARE | 0.100 | 10.000 | 0.902 | 0.009 | 0.003 | 10.000 | 0.902 | 0.009 | 0.003 |
| COMPARE | 0.200 | 10.000 | 0.805 | 0.025 | 0.008 | 10.000 | 0.805 | 0.025 | 0.008 |
| COMPARE | 0.300 | 10.000 | 0.697 | 0.027 | 0.008 | 10.000 | 0.697 | 0.027 | 0.008 |
| COMPARE | 0.500 | 10.000 | 0.498 | 0.018 | 0.006 | 10.000 | 0.498 | 0.018 | 0.006 |
| COMPARE | 0.750 | 10.000 | 0.250 | 0.016 | 0.005 | 10.000 | 0.250 | 0.016 | 0.005 |
| COMPARE | 1.000 | 10.000 | 0.000 | 0.000 | 0.000 | 10.000 | 0.000 | 0.000 | 0.000 |

## Exp06: Learned transitions

- exact_pair/full_table shows that transition associations can be acquired from demonstrations.
- hashed_pair exposes capacity limits in the state-input conjunctive interface layer.
- sparse learned writer rows test whether learned transitions can target basins partially.
- coverage split rows show that arbitrary FSMs do not generalize without structural regularity.
- exact_pair nearest accuracy after one epoch/full table: 1.000.
- exact_pair Hopfield accuracy after one epoch/full table: 0.909.
- First hashed hidden_dim reaching 0.95 nearest accuracy: not reached.
- nearest sparse random-noise accuracy at write_fraction=0.30: 0.999.
- Hopfield sparse random-noise accuracy at write_fraction=0.30: 0.945.
- nearest seen/unseen at coverage_fraction=0.50: seen=1.000, unseen=0.058.

### Overall summary by feature mode and register type

| feature_mode | register_type | count | all_transition_accuracy_mean | all_transition_accuracy_std | all_transition_accuracy_sem | all_transition_accuracy_min | all_transition_accuracy_median | all_transition_accuracy_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact_pair | hopfield | 6600.000 | 0.737 | 0.341 | 0.004 | 0.000 | 1.000 | 1.000 |
| exact_pair | nearest | 6600.000 | 0.799 | 0.343 | 0.004 | 0.000 | 1.000 | 1.000 |
| hashed_pair | hopfield | 30240.000 | 0.562 | 0.342 | 0.002 | 0.000 | 0.625 | 1.000 |
| hashed_pair | nearest | 30240.000 | 0.621 | 0.333 | 0.002 | 0.000 | 0.750 | 1.000 |
| random_conjunctive | hopfield | 25200.000 | 0.308 | 0.342 | 0.002 | 0.000 | 0.156 | 1.000 |
| random_conjunctive | nearest | 25200.000 | 0.396 | 0.371 | 0.002 | 0.000 | 0.234 | 1.000 |

### Hashed-pair capacity at max epoch

| register_type | hidden_dim | count | all_transition_accuracy_mean | all_transition_accuracy_std | all_transition_accuracy_sem | all_transition_accuracy_min | all_transition_accuracy_median | all_transition_accuracy_max | all_transition_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hopfield | 32.000 | 360.000 | 0.384 | 0.250 | 0.013 | 0.033 | 0.309 | 1.000 | 0.384 |
| hopfield | 64.000 | 360.000 | 0.518 | 0.267 | 0.014 | 0.041 | 0.496 | 1.000 | 0.518 |
| hopfield | 128.000 | 360.000 | 0.647 | 0.252 | 0.013 | 0.090 | 0.680 | 1.000 | 0.647 |
| hopfield | 256.000 | 360.000 | 0.745 | 0.224 | 0.012 | 0.143 | 0.828 | 1.000 | 0.745 |
| hopfield | 512.000 | 360.000 | 0.818 | 0.200 | 0.011 | 0.215 | 0.898 | 1.000 | 0.818 |
| hopfield | 1024.000 | 360.000 | 0.858 | 0.184 | 0.010 | 0.258 | 0.949 | 1.000 | 0.858 |
| nearest | 32.000 | 360.000 | 0.400 | 0.238 | 0.013 | 0.068 | 0.352 | 1.000 | 0.400 |
| nearest | 64.000 | 360.000 | 0.546 | 0.245 | 0.013 | 0.131 | 0.551 | 1.000 | 0.546 |
| nearest | 128.000 | 360.000 | 0.690 | 0.211 | 0.011 | 0.248 | 0.703 | 1.000 | 0.690 |
| nearest | 256.000 | 360.000 | 0.804 | 0.156 | 0.008 | 0.420 | 0.836 | 1.000 | 0.804 |
| nearest | 512.000 | 360.000 | 0.890 | 0.100 | 0.005 | 0.621 | 0.922 | 1.000 | 0.890 |
| nearest | 1024.000 | 360.000 | 0.939 | 0.059 | 0.003 | 0.758 | 0.953 | 1.000 | 0.939 |

### Sparse learned transitions at max epoch

| register_type | unwritten_mode | write_fraction | count | all_transition_accuracy_mean | all_transition_accuracy_std | all_transition_accuracy_sem | all_transition_accuracy_min | all_transition_accuracy_median | all_transition_accuracy_max | all_transition_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hopfield | keep_current | 0.200 | 40.000 | 0.049 | 0.026 | 0.004 | 0.008 | 0.047 | 0.109 | 0.049 |
| hopfield | keep_current | 0.300 | 40.000 | 0.085 | 0.041 | 0.007 | 0.016 | 0.078 | 0.211 | 0.085 |
| hopfield | keep_current | 0.400 | 40.000 | 0.571 | 0.100 | 0.016 | 0.336 | 0.578 | 0.750 | 0.571 |
| hopfield | keep_current | 0.500 | 40.000 | 0.969 | 0.057 | 0.009 | 0.758 | 1.000 | 1.000 | 0.969 |
| hopfield | keep_current | 0.750 | 40.000 | 1.000 | 0.002 | 0.000 | 0.992 | 1.000 | 1.000 | 1.000 |
| hopfield | keep_current | 1.000 | 40.000 | 0.999 | 0.006 | 0.001 | 0.961 | 1.000 | 1.000 | 0.999 |
| hopfield | random_noise | 0.200 | 40.000 | 0.770 | 0.215 | 0.034 | 0.297 | 0.855 | 1.000 | 0.770 |
| hopfield | random_noise | 0.300 | 40.000 | 0.945 | 0.093 | 0.015 | 0.711 | 1.000 | 1.000 | 0.945 |
| hopfield | random_noise | 0.400 | 40.000 | 0.989 | 0.024 | 0.004 | 0.922 | 1.000 | 1.000 | 0.989 |
| hopfield | random_noise | 0.500 | 40.000 | 0.996 | 0.011 | 0.002 | 0.953 | 1.000 | 1.000 | 0.996 |
| hopfield | random_noise | 0.750 | 40.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hopfield | random_noise | 1.000 | 40.000 | 0.999 | 0.006 | 0.001 | 0.961 | 1.000 | 1.000 | 0.999 |
| nearest | keep_current | 0.200 | 40.000 | 0.048 | 0.026 | 0.004 | 0.008 | 0.043 | 0.109 | 0.048 |
| nearest | keep_current | 0.300 | 40.000 | 0.048 | 0.026 | 0.004 | 0.008 | 0.043 | 0.109 | 0.048 |
| nearest | keep_current | 0.400 | 40.000 | 0.048 | 0.026 | 0.004 | 0.008 | 0.043 | 0.109 | 0.048 |
| nearest | keep_current | 0.500 | 40.000 | 0.521 | 0.070 | 0.011 | 0.352 | 0.516 | 0.641 | 0.521 |
| nearest | keep_current | 0.750 | 40.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| nearest | keep_current | 1.000 | 40.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| nearest | random_noise | 0.200 | 40.000 | 0.941 | 0.058 | 0.009 | 0.812 | 0.965 | 1.000 | 0.941 |
| nearest | random_noise | 0.300 | 40.000 | 0.999 | 0.004 | 0.001 | 0.984 | 1.000 | 1.000 | 0.999 |
| nearest | random_noise | 0.400 | 40.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| nearest | random_noise | 0.500 | 40.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| nearest | random_noise | 0.750 | 40.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| nearest | random_noise | 1.000 | 40.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### Coverage all accuracy

| register_type | coverage_fraction | count | all_transition_accuracy_mean | all_transition_accuracy_std | all_transition_accuracy_sem | all_transition_accuracy_min | all_transition_accuracy_median | all_transition_accuracy_max | all_transition_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hopfield | 0.250 | 360.000 | 0.268 | 0.069 | 0.004 | 0.074 | 0.273 | 0.469 | 0.268 |
| hopfield | 0.500 | 360.000 | 0.481 | 0.099 | 0.005 | 0.125 | 0.516 | 0.688 | 0.481 |
| hopfield | 0.750 | 360.000 | 0.695 | 0.135 | 0.007 | 0.227 | 0.750 | 0.875 | 0.695 |
| nearest | 0.250 | 360.000 | 0.293 | 0.038 | 0.002 | 0.250 | 0.281 | 0.438 | 0.293 |
| nearest | 0.500 | 360.000 | 0.529 | 0.032 | 0.002 | 0.500 | 0.516 | 0.688 | 0.529 |
| nearest | 0.750 | 360.000 | 0.766 | 0.021 | 0.001 | 0.750 | 0.758 | 0.875 | 0.766 |

### Coverage seen accuracy

| register_type | coverage_fraction | count | seen_transition_accuracy_mean | seen_transition_accuracy_std | seen_transition_accuracy_sem | seen_transition_accuracy_min | seen_transition_accuracy_median | seen_transition_accuracy_max | seen_transition_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hopfield | 0.250 | 360.000 | 0.907 | 0.179 | 0.009 | 0.219 | 1.000 | 1.000 | 0.907 |
| hopfield | 0.500 | 360.000 | 0.909 | 0.173 | 0.009 | 0.219 | 1.000 | 1.000 | 0.909 |
| hopfield | 0.750 | 360.000 | 0.909 | 0.172 | 0.009 | 0.292 | 1.000 | 1.000 | 0.909 |
| nearest | 0.250 | 360.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| nearest | 0.500 | 360.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| nearest | 0.750 | 360.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### Coverage unseen accuracy

| register_type | coverage_fraction | count | unseen_transition_accuracy_mean | unseen_transition_accuracy_std | unseen_transition_accuracy_sem | unseen_transition_accuracy_min | unseen_transition_accuracy_median | unseen_transition_accuracy_max | unseen_transition_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hopfield | 0.250 | 360.000 | 0.055 | 0.053 | 0.003 | 0.000 | 0.042 | 0.292 | 0.055 |
| hopfield | 0.500 | 360.000 | 0.054 | 0.058 | 0.003 | 0.000 | 0.031 | 0.375 | 0.054 |
| hopfield | 0.750 | 360.000 | 0.056 | 0.075 | 0.004 | 0.000 | 0.031 | 0.500 | 0.056 |
| nearest | 0.250 | 360.000 | 0.057 | 0.051 | 0.003 | 0.000 | 0.042 | 0.250 | 0.057 |
| nearest | 0.500 | 360.000 | 0.058 | 0.063 | 0.003 | 0.000 | 0.031 | 0.375 | 0.058 |
| nearest | 0.750 | 360.000 | 0.062 | 0.082 | 0.004 | 0.000 | 0.031 | 0.500 | 0.062 |

## Exp07: Structured grammar learning

- Random transition tables test acquisition and capacity, but structured grammars test reusable transition structure.
- Exact-pair nearest rows are the cleanest test of whether learned DFA transitions support string-level generalization.
- Exact-pair Hopfield rows test whether the same learned transitions inherit recurrent cleanup limits.
- Hashed-pair rows test capacity limits in the state-input context layer.
- Sparse-top-k rows report autonomous rollout accuracy, not only one-step masked cleanup.
- Full-transition and limited-exposure regimes are reported separately to avoid mixing complete-coverage success with coverage-failure rows.
- Transition coverage explains most failures: missing DFA edges lead to systematic autonomous rollout errors.
- Masked top-k rows test whether sparse visible coordinates can identify the correct next-state basin.
- Exact-pair nearest test string accuracy: 0.968.
- Exact-pair nearest length-generalization accuracy: 0.967.
- Exact-pair nearest state-tracking accuracy: 0.940.
- Exact-pair Hopfield dense test string accuracy: 0.969.
- Exact-pair Hopfield dense length-generalization accuracy: 0.969.
- Exact-pair Hopfield dense state-tracking accuracy: 0.943.
- Hashed-pair nearest dense test accuracy by hidden_dim: 32.000=0.805, 64.000=0.875, 128.000=0.918, 256.000=0.940.
- Hashed-pair nearest dense mean test accuracy: 0.885.
- Exact-pair nearest sparse-top-k autonomous transition accuracy at write_fraction=0.30 random_noise: 0.941.
- Exact-pair nearest sparse-top-k autonomous transition accuracy at write_fraction=0.50 random_noise: 0.942.
- Exact-pair nearest sparse-top-k autonomous transition accuracy at write_fraction=0.50 keep_current: 0.427.
- Mean training transition coverage: 0.912.
- Exact-pair nearest dense incomplete-coverage rows: 280.
- Incomplete-coverage mean transition coverage: 0.253.
- Incomplete-coverage mean test string accuracy: 0.518.
- Seen/unseen transition accuracy: seen=1.000, unseen=0.075.
- Limited-exposure seen/unseen transition accuracy: seen=1.000, unseen=0.299.
- First masked top-k reaching 0.95: not reached.
- First masked top-k reaching 0.99: not reached.

### Overall summary by model

| regime | feature_mode | register_type | output_mode | count | test_string_accuracy_mean | test_string_accuracy_std | test_string_accuracy_sem | test_string_accuracy_min | test_string_accuracy_median | test_string_accuracy_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_transition_exposure | exact_pair | hopfield | dense | 280.000 | 0.969 | 0.120 | 0.007 | 0.500 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | hopfield | sparse_topk | 840.000 | 0.830 | 0.230 | 0.008 | 0.485 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | nearest | dense | 280.000 | 0.968 | 0.123 | 0.007 | 0.500 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | nearest | sparse_topk | 840.000 | 0.832 | 0.231 | 0.008 | 0.494 | 1.000 | 1.000 |
| full_transition_exposure | hashed_pair | hopfield | dense | 1120.000 | 0.886 | 0.193 | 0.006 | 0.435 | 1.000 | 1.000 |
| full_transition_exposure | hashed_pair | nearest | dense | 1120.000 | 0.885 | 0.194 | 0.006 | 0.435 | 1.000 | 1.000 |

### Variant test accuracy

| regime | feature_mode | hidden_dim | register_type | output_mode | write_fraction | unwritten_mode | count | test_string_accuracy_mean | test_string_accuracy_std | test_string_accuracy_sem | test_string_accuracy_min | test_string_accuracy_median | test_string_accuracy_max | test_string_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_transition_exposure | exact_pair | 4.000 | hopfield | dense | 1.000 | random_noise | 40.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 4.000 | hopfield | sparse_topk | 0.300 | random_noise | 40.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 4.000 | hopfield | sparse_topk | 0.500 | keep_current | 40.000 | 0.688 | 0.245 | 0.039 | 0.500 | 0.500 | 1.000 | 0.688 |
| full_transition_exposure | exact_pair | 4.000 | hopfield | sparse_topk | 0.500 | random_noise | 40.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 4.000 | nearest | dense | 1.000 | random_noise | 40.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 4.000 | nearest | sparse_topk | 0.300 | random_noise | 40.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 4.000 | nearest | sparse_topk | 0.500 | keep_current | 40.000 | 0.663 | 0.237 | 0.037 | 0.500 | 0.500 | 1.000 | 0.663 |
| full_transition_exposure | exact_pair | 4.000 | nearest | sparse_topk | 0.500 | random_noise | 40.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 6.000 | hopfield | dense | 1.000 | random_noise | 100.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 6.000 | hopfield | sparse_topk | 0.300 | random_noise | 100.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 6.000 | hopfield | sparse_topk | 0.500 | keep_current | 100.000 | 0.551 | 0.136 | 0.014 | 0.485 | 0.500 | 1.000 | 0.551 |
| full_transition_exposure | exact_pair | 6.000 | hopfield | sparse_topk | 0.500 | random_noise | 100.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 6.000 | nearest | dense | 1.000 | random_noise | 100.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 6.000 | nearest | sparse_topk | 0.300 | random_noise | 100.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 6.000 | nearest | sparse_topk | 0.500 | keep_current | 100.000 | 0.581 | 0.174 | 0.017 | 0.494 | 0.500 | 1.000 | 0.581 |
| full_transition_exposure | exact_pair | 6.000 | nearest | sparse_topk | 0.500 | random_noise | 100.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 8.000 | hopfield | dense | 1.000 | random_noise | 60.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 8.000 | hopfield | sparse_topk | 0.300 | random_noise | 60.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 8.000 | hopfield | sparse_topk | 0.500 | keep_current | 60.000 | 0.539 | 0.120 | 0.015 | 0.500 | 0.500 | 1.000 | 0.539 |
| full_transition_exposure | exact_pair | 8.000 | hopfield | sparse_topk | 0.500 | random_noise | 60.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 8.000 | nearest | dense | 1.000 | random_noise | 60.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 8.000 | nearest | sparse_topk | 0.300 | random_noise | 60.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 8.000 | nearest | sparse_topk | 0.500 | keep_current | 60.000 | 0.546 | 0.139 | 0.018 | 0.500 | 0.500 | 1.000 | 0.546 |
| full_transition_exposure | exact_pair | 8.000 | nearest | sparse_topk | 0.500 | random_noise | 60.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 10.000 | hopfield | dense | 1.000 | random_noise | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 10.000 | hopfield | sparse_topk | 0.300 | random_noise | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 10.000 | hopfield | sparse_topk | 0.500 | keep_current | 20.000 | 0.528 | 0.112 | 0.025 | 0.500 | 0.500 | 1.000 | 0.528 |
| full_transition_exposure | exact_pair | 10.000 | hopfield | sparse_topk | 0.500 | random_noise | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 10.000 | nearest | dense | 1.000 | random_noise | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 10.000 | nearest | sparse_topk | 0.300 | random_noise | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 10.000 | nearest | sparse_topk | 0.500 | keep_current | 20.000 | 0.525 | 0.112 | 0.025 | 0.500 | 0.500 | 1.000 | 0.525 |
| full_transition_exposure | exact_pair | 10.000 | nearest | sparse_topk | 0.500 | random_noise | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 12.000 | hopfield | dense | 1.000 | random_noise | 40.000 | 0.787 | 0.250 | 0.040 | 0.500 | 1.000 | 1.000 | 0.787 |
| full_transition_exposure | exact_pair | 12.000 | hopfield | sparse_topk | 0.300 | random_noise | 40.000 | 0.779 | 0.227 | 0.036 | 0.500 | 0.833 | 1.000 | 0.779 |
| full_transition_exposure | exact_pair | 12.000 | hopfield | sparse_topk | 0.500 | keep_current | 40.000 | 0.521 | 0.063 | 0.010 | 0.500 | 0.500 | 0.731 | 0.521 |
| full_transition_exposure | exact_pair | 12.000 | hopfield | sparse_topk | 0.500 | random_noise | 40.000 | 0.785 | 0.229 | 0.036 | 0.500 | 0.915 | 1.000 | 0.785 |
| full_transition_exposure | exact_pair | 12.000 | nearest | dense | 1.000 | random_noise | 40.000 | 0.775 | 0.252 | 0.040 | 0.500 | 1.000 | 1.000 | 0.775 |
| full_transition_exposure | exact_pair | 12.000 | nearest | sparse_topk | 0.300 | random_noise | 40.000 | 0.772 | 0.236 | 0.037 | 0.500 | 0.847 | 1.000 | 0.772 |
| full_transition_exposure | exact_pair | 12.000 | nearest | sparse_topk | 0.500 | keep_current | 40.000 | 0.532 | 0.096 | 0.015 | 0.500 | 0.500 | 1.000 | 0.532 |
| full_transition_exposure | exact_pair | 12.000 | nearest | sparse_topk | 0.500 | random_noise | 40.000 | 0.781 | 0.231 | 0.037 | 0.500 | 0.893 | 1.000 | 0.781 |
| full_transition_exposure | exact_pair | 63.000 | hopfield | dense | 1.000 | random_noise | 20.000 | 0.997 | 0.003 | 0.001 | 0.992 | 0.999 | 1.000 | 0.997 |
| full_transition_exposure | exact_pair | 63.000 | hopfield | sparse_topk | 0.300 | random_noise | 20.000 | 0.930 | 0.075 | 0.017 | 0.714 | 0.951 | 0.993 | 0.930 |
| full_transition_exposure | exact_pair | 63.000 | hopfield | sparse_topk | 0.500 | keep_current | 20.000 | 0.517 | 0.054 | 0.012 | 0.500 | 0.500 | 0.718 | 0.517 |
| full_transition_exposure | exact_pair | 63.000 | hopfield | sparse_topk | 0.500 | random_noise | 20.000 | 0.957 | 0.074 | 0.017 | 0.693 | 0.985 | 0.999 | 0.957 |
| full_transition_exposure | exact_pair | 63.000 | nearest | dense | 1.000 | random_noise | 20.000 | 0.995 | 0.012 | 0.003 | 0.946 | 0.999 | 1.000 | 0.995 |
| full_transition_exposure | exact_pair | 63.000 | nearest | sparse_topk | 0.300 | random_noise | 20.000 | 0.948 | 0.031 | 0.007 | 0.892 | 0.957 | 0.990 | 0.948 |
| full_transition_exposure | exact_pair | 63.000 | nearest | sparse_topk | 0.500 | keep_current | 20.000 | 0.500 | 0.000 | 0.000 | 0.500 | 0.500 | 0.500 | 0.500 |
| full_transition_exposure | exact_pair | 63.000 | nearest | sparse_topk | 0.500 | random_noise | 20.000 | 0.955 | 0.069 | 0.015 | 0.716 | 0.978 | 0.999 | 0.955 |
| full_transition_exposure | hashed_pair | 32.000 | hopfield | dense | 1.000 | random_noise | 280.000 | 0.805 | 0.222 | 0.013 | 0.436 | 0.993 | 1.000 | 0.805 |
| full_transition_exposure | hashed_pair | 32.000 | nearest | dense | 1.000 | random_noise | 280.000 | 0.805 | 0.222 | 0.013 | 0.436 | 0.992 | 1.000 | 0.805 |
| full_transition_exposure | hashed_pair | 64.000 | hopfield | dense | 1.000 | random_noise | 280.000 | 0.876 | 0.195 | 0.012 | 0.500 | 1.000 | 1.000 | 0.876 |
| full_transition_exposure | hashed_pair | 64.000 | nearest | dense | 1.000 | random_noise | 280.000 | 0.875 | 0.196 | 0.012 | 0.500 | 1.000 | 1.000 | 0.875 |
| full_transition_exposure | hashed_pair | 128.000 | hopfield | dense | 1.000 | random_noise | 280.000 | 0.921 | 0.168 | 0.010 | 0.435 | 1.000 | 1.000 | 0.921 |
| full_transition_exposure | hashed_pair | 128.000 | nearest | dense | 1.000 | random_noise | 280.000 | 0.918 | 0.170 | 0.010 | 0.435 | 1.000 | 1.000 | 0.918 |
| full_transition_exposure | hashed_pair | 256.000 | hopfield | dense | 1.000 | random_noise | 280.000 | 0.942 | 0.151 | 0.009 | 0.500 | 1.000 | 1.000 | 0.942 |
| full_transition_exposure | hashed_pair | 256.000 | nearest | dense | 1.000 | random_noise | 280.000 | 0.940 | 0.153 | 0.009 | 0.500 | 1.000 | 1.000 | 0.940 |

### Variant autonomous transition accuracy

| regime | feature_mode | hidden_dim | register_type | output_mode | write_fraction | unwritten_mode | count | transition_accuracy_autonomous_mean | transition_accuracy_autonomous_std | transition_accuracy_autonomous_sem | transition_accuracy_autonomous_min | transition_accuracy_autonomous_median | transition_accuracy_autonomous_max | transition_accuracy_autonomous |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_transition_exposure | exact_pair | 4.000 | hopfield | dense | 1.000 | random_noise | 40.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 4.000 | hopfield | sparse_topk | 0.300 | random_noise | 40.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 4.000 | hopfield | sparse_topk | 0.500 | keep_current | 40.000 | 0.672 | 0.264 | 0.042 | 0.404 | 0.547 | 1.000 | 0.672 |
| full_transition_exposure | exact_pair | 4.000 | hopfield | sparse_topk | 0.500 | random_noise | 40.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 4.000 | nearest | dense | 1.000 | random_noise | 40.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 4.000 | nearest | sparse_topk | 0.300 | random_noise | 40.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 4.000 | nearest | sparse_topk | 0.500 | keep_current | 40.000 | 0.642 | 0.258 | 0.041 | 0.404 | 0.503 | 1.000 | 0.642 |
| full_transition_exposure | exact_pair | 4.000 | nearest | sparse_topk | 0.500 | random_noise | 40.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 6.000 | hopfield | dense | 1.000 | random_noise | 100.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 6.000 | hopfield | sparse_topk | 0.300 | random_noise | 100.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 6.000 | hopfield | sparse_topk | 0.500 | keep_current | 100.000 | 0.437 | 0.204 | 0.020 | 0.136 | 0.377 | 1.000 | 0.437 |
| full_transition_exposure | exact_pair | 6.000 | hopfield | sparse_topk | 0.500 | random_noise | 100.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 6.000 | nearest | dense | 1.000 | random_noise | 100.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 6.000 | nearest | sparse_topk | 0.300 | random_noise | 100.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 6.000 | nearest | sparse_topk | 0.500 | keep_current | 100.000 | 0.481 | 0.244 | 0.024 | 0.192 | 0.432 | 1.000 | 0.481 |
| full_transition_exposure | exact_pair | 6.000 | nearest | sparse_topk | 0.500 | random_noise | 100.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 8.000 | hopfield | dense | 1.000 | random_noise | 60.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 8.000 | hopfield | sparse_topk | 0.300 | random_noise | 60.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 8.000 | hopfield | sparse_topk | 0.500 | keep_current | 60.000 | 0.384 | 0.182 | 0.023 | 0.194 | 0.324 | 1.000 | 0.384 |
| full_transition_exposure | exact_pair | 8.000 | hopfield | sparse_topk | 0.500 | random_noise | 60.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 8.000 | nearest | dense | 1.000 | random_noise | 60.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 8.000 | nearest | sparse_topk | 0.300 | random_noise | 60.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 8.000 | nearest | sparse_topk | 0.500 | keep_current | 60.000 | 0.417 | 0.209 | 0.027 | 0.253 | 0.393 | 1.000 | 0.417 |
| full_transition_exposure | exact_pair | 8.000 | nearest | sparse_topk | 0.500 | random_noise | 60.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 10.000 | hopfield | dense | 1.000 | random_noise | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 10.000 | hopfield | sparse_topk | 0.300 | random_noise | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 10.000 | hopfield | sparse_topk | 0.500 | keep_current | 20.000 | 0.352 | 0.219 | 0.049 | 0.140 | 0.371 | 1.000 | 0.352 |
| full_transition_exposure | exact_pair | 10.000 | hopfield | sparse_topk | 0.500 | random_noise | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 10.000 | nearest | dense | 1.000 | random_noise | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 10.000 | nearest | sparse_topk | 0.300 | random_noise | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 10.000 | nearest | sparse_topk | 0.500 | keep_current | 20.000 | 0.320 | 0.248 | 0.055 | 0.140 | 0.145 | 1.000 | 0.320 |
| full_transition_exposure | exact_pair | 10.000 | nearest | sparse_topk | 0.500 | random_noise | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | exact_pair | 12.000 | hopfield | dense | 1.000 | random_noise | 40.000 | 0.788 | 0.249 | 0.039 | 0.494 | 1.000 | 1.000 | 0.788 |
| full_transition_exposure | exact_pair | 12.000 | hopfield | sparse_topk | 0.300 | random_noise | 40.000 | 0.784 | 0.223 | 0.035 | 0.494 | 0.848 | 1.000 | 0.784 |
| full_transition_exposure | exact_pair | 12.000 | hopfield | sparse_topk | 0.500 | keep_current | 40.000 | 0.282 | 0.133 | 0.021 | 0.087 | 0.293 | 0.545 | 0.282 |
| full_transition_exposure | exact_pair | 12.000 | hopfield | sparse_topk | 0.500 | random_noise | 40.000 | 0.788 | 0.225 | 0.036 | 0.494 | 0.905 | 1.000 | 0.788 |
| full_transition_exposure | exact_pair | 12.000 | nearest | dense | 1.000 | random_noise | 40.000 | 0.776 | 0.251 | 0.040 | 0.494 | 1.000 | 1.000 | 0.776 |
| full_transition_exposure | exact_pair | 12.000 | nearest | sparse_topk | 0.300 | random_noise | 40.000 | 0.776 | 0.233 | 0.037 | 0.494 | 0.849 | 1.000 | 0.776 |
| full_transition_exposure | exact_pair | 12.000 | nearest | sparse_topk | 0.500 | keep_current | 40.000 | 0.287 | 0.175 | 0.028 | 0.087 | 0.223 | 0.877 | 0.287 |
| full_transition_exposure | exact_pair | 12.000 | nearest | sparse_topk | 0.500 | random_noise | 40.000 | 0.785 | 0.228 | 0.036 | 0.494 | 0.910 | 1.000 | 0.785 |
| full_transition_exposure | exact_pair | 63.000 | hopfield | dense | 1.000 | random_noise | 20.000 | 0.629 | 0.127 | 0.028 | 0.580 | 0.588 | 1.000 | 0.629 |
| full_transition_exposure | exact_pair | 63.000 | hopfield | sparse_topk | 0.300 | random_noise | 20.000 | 0.605 | 0.055 | 0.012 | 0.445 | 0.607 | 0.698 | 0.605 |
| full_transition_exposure | exact_pair | 63.000 | hopfield | sparse_topk | 0.500 | keep_current | 20.000 | 0.166 | 0.126 | 0.028 | 0.000 | 0.169 | 0.356 | 0.166 |
| full_transition_exposure | exact_pair | 63.000 | hopfield | sparse_topk | 0.500 | random_noise | 20.000 | 0.610 | 0.026 | 0.006 | 0.580 | 0.605 | 0.679 | 0.610 |
| full_transition_exposure | exact_pair | 63.000 | nearest | dense | 1.000 | random_noise | 20.000 | 0.608 | 0.092 | 0.021 | 0.580 | 0.587 | 1.000 | 0.608 |
| full_transition_exposure | exact_pair | 63.000 | nearest | sparse_topk | 0.300 | random_noise | 20.000 | 0.621 | 0.026 | 0.006 | 0.582 | 0.620 | 0.680 | 0.621 |
| full_transition_exposure | exact_pair | 63.000 | nearest | sparse_topk | 0.500 | keep_current | 20.000 | 0.137 | 0.129 | 0.029 | 0.000 | 0.082 | 0.355 | 0.137 |
| full_transition_exposure | exact_pair | 63.000 | nearest | sparse_topk | 0.500 | random_noise | 20.000 | 0.616 | 0.040 | 0.009 | 0.582 | 0.597 | 0.708 | 0.616 |
| full_transition_exposure | hashed_pair | 32.000 | hopfield | dense | 1.000 | random_noise | 280.000 | 0.771 | 0.259 | 0.015 | 0.081 | 0.939 | 1.000 | 0.771 |
| full_transition_exposure | hashed_pair | 32.000 | nearest | dense | 1.000 | random_noise | 280.000 | 0.770 | 0.260 | 0.016 | 0.152 | 0.939 | 1.000 | 0.770 |
| full_transition_exposure | hashed_pair | 64.000 | hopfield | dense | 1.000 | random_noise | 280.000 | 0.847 | 0.223 | 0.013 | 0.263 | 1.000 | 1.000 | 0.847 |
| full_transition_exposure | hashed_pair | 64.000 | nearest | dense | 1.000 | random_noise | 280.000 | 0.845 | 0.227 | 0.014 | 0.224 | 1.000 | 1.000 | 0.845 |
| full_transition_exposure | hashed_pair | 128.000 | hopfield | dense | 1.000 | random_noise | 280.000 | 0.891 | 0.204 | 0.012 | 0.225 | 1.000 | 1.000 | 0.891 |
| full_transition_exposure | hashed_pair | 128.000 | nearest | dense | 1.000 | random_noise | 280.000 | 0.888 | 0.209 | 0.013 | 0.224 | 1.000 | 1.000 | 0.888 |
| full_transition_exposure | hashed_pair | 256.000 | hopfield | dense | 1.000 | random_noise | 280.000 | 0.918 | 0.175 | 0.010 | 0.338 | 1.000 | 1.000 | 0.918 |
| full_transition_exposure | hashed_pair | 256.000 | nearest | dense | 1.000 | random_noise | 280.000 | 0.915 | 0.180 | 0.011 | 0.338 | 1.000 | 1.000 | 0.915 |

### Grammar-level accuracy

| regime | grammar_name | feature_mode | register_type | count | test_string_accuracy_mean | test_string_accuracy_std | test_string_accuracy_sem | test_string_accuracy_min | test_string_accuracy_median | test_string_accuracy_max | test_string_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_transition_exposure | contains_101 | exact_pair | hopfield | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | contains_101 | exact_pair | nearest | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | contains_101 | hashed_pair | hopfield | 80.000 | 0.945 | 0.152 | 0.017 | 0.500 | 1.000 | 1.000 | 0.945 |
| full_transition_exposure | contains_101 | hashed_pair | nearest | 80.000 | 0.945 | 0.152 | 0.017 | 0.500 | 1.000 | 1.000 | 0.945 |
| full_transition_exposure | ends_with_01 | exact_pair | hopfield | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | ends_with_01 | exact_pair | nearest | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | ends_with_01 | hashed_pair | hopfield | 80.000 | 0.939 | 0.146 | 0.016 | 0.500 | 1.000 | 1.000 | 0.939 |
| full_transition_exposure | ends_with_01 | hashed_pair | nearest | 80.000 | 0.939 | 0.146 | 0.016 | 0.500 | 1.000 | 1.000 | 0.939 |
| full_transition_exposure | even_ones | exact_pair | hopfield | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | even_ones | exact_pair | nearest | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | even_ones | hashed_pair | hopfield | 80.000 | 0.981 | 0.095 | 0.011 | 0.500 | 1.000 | 1.000 | 0.981 |
| full_transition_exposure | even_ones | hashed_pair | nearest | 80.000 | 0.981 | 0.095 | 0.011 | 0.500 | 1.000 | 1.000 | 0.981 |
| full_transition_exposure | no_substring_11 | exact_pair | hopfield | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | no_substring_11 | exact_pair | nearest | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | no_substring_11 | hashed_pair | hopfield | 80.000 | 0.877 | 0.209 | 0.023 | 0.500 | 1.000 | 1.000 | 0.877 |
| full_transition_exposure | no_substring_11 | hashed_pair | nearest | 80.000 | 0.877 | 0.209 | 0.023 | 0.500 | 1.000 | 1.000 | 0.877 |
| full_transition_exposure | ones_mod3_zero | exact_pair | hopfield | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | ones_mod3_zero | exact_pair | nearest | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | ones_mod3_zero | hashed_pair | hopfield | 80.000 | 0.952 | 0.145 | 0.016 | 0.436 | 1.000 | 1.000 | 0.952 |
| full_transition_exposure | ones_mod3_zero | hashed_pair | nearest | 80.000 | 0.952 | 0.145 | 0.016 | 0.436 | 1.000 | 1.000 | 0.952 |
| full_transition_exposure | reber | exact_pair | hopfield | 20.000 | 0.997 | 0.003 | 0.001 | 0.992 | 0.999 | 1.000 | 0.997 |
| full_transition_exposure | reber | exact_pair | nearest | 20.000 | 0.995 | 0.012 | 0.003 | 0.946 | 0.999 | 1.000 | 0.995 |
| full_transition_exposure | reber | hashed_pair | hopfield | 80.000 | 0.860 | 0.167 | 0.019 | 0.493 | 0.988 | 1.000 | 0.860 |
| full_transition_exposure | reber | hashed_pair | nearest | 80.000 | 0.859 | 0.162 | 0.018 | 0.500 | 0.965 | 1.000 | 0.859 |
| full_transition_exposure | tier_alternating_12 | exact_pair | hopfield | 20.000 | 0.575 | 0.183 | 0.041 | 0.500 | 0.500 | 1.000 | 0.575 |
| full_transition_exposure | tier_alternating_12 | exact_pair | nearest | 20.000 | 0.550 | 0.154 | 0.034 | 0.500 | 0.500 | 1.000 | 0.550 |
| full_transition_exposure | tier_alternating_12 | hashed_pair | hopfield | 80.000 | 0.554 | 0.144 | 0.016 | 0.500 | 0.500 | 1.000 | 0.554 |
| full_transition_exposure | tier_alternating_12 | hashed_pair | nearest | 80.000 | 0.538 | 0.124 | 0.014 | 0.500 | 0.500 | 1.000 | 0.538 |
| full_transition_exposure | tomita_1 | exact_pair | hopfield | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | tomita_1 | exact_pair | nearest | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | tomita_1 | hashed_pair | hopfield | 80.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | tomita_1 | hashed_pair | nearest | 80.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | tomita_2 | exact_pair | hopfield | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | tomita_2 | exact_pair | nearest | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | tomita_2 | hashed_pair | hopfield | 80.000 | 0.986 | 0.070 | 0.008 | 0.577 | 1.000 | 1.000 | 0.986 |
| full_transition_exposure | tomita_2 | hashed_pair | nearest | 80.000 | 0.986 | 0.070 | 0.008 | 0.577 | 1.000 | 1.000 | 0.986 |
| full_transition_exposure | tomita_3 | exact_pair | hopfield | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | tomita_3 | exact_pair | nearest | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | tomita_3 | hashed_pair | hopfield | 80.000 | 0.810 | 0.206 | 0.023 | 0.435 | 0.912 | 1.000 | 0.810 |
| full_transition_exposure | tomita_3 | hashed_pair | nearest | 80.000 | 0.810 | 0.206 | 0.023 | 0.435 | 0.912 | 1.000 | 0.810 |
| full_transition_exposure | tomita_4 | exact_pair | hopfield | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | tomita_4 | exact_pair | nearest | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | tomita_4 | hashed_pair | hopfield | 80.000 | 0.813 | 0.226 | 0.025 | 0.500 | 1.000 | 1.000 | 0.813 |
| full_transition_exposure | tomita_4 | hashed_pair | nearest | 80.000 | 0.813 | 0.226 | 0.025 | 0.500 | 1.000 | 1.000 | 0.813 |
| full_transition_exposure | tomita_5 | exact_pair | hopfield | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | tomita_5 | exact_pair | nearest | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | tomita_5 | hashed_pair | hopfield | 80.000 | 0.904 | 0.162 | 0.018 | 0.501 | 1.000 | 1.000 | 0.904 |
| full_transition_exposure | tomita_5 | hashed_pair | nearest | 80.000 | 0.904 | 0.162 | 0.018 | 0.501 | 1.000 | 1.000 | 0.904 |
| full_transition_exposure | tomita_6 | exact_pair | hopfield | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | tomita_6 | exact_pair | nearest | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | tomita_6 | hashed_pair | hopfield | 80.000 | 0.944 | 0.159 | 0.018 | 0.461 | 1.000 | 1.000 | 0.944 |
| full_transition_exposure | tomita_6 | hashed_pair | nearest | 80.000 | 0.944 | 0.159 | 0.018 | 0.461 | 1.000 | 1.000 | 0.944 |
| full_transition_exposure | tomita_7 | exact_pair | hopfield | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | tomita_7 | exact_pair | nearest | 20.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_transition_exposure | tomita_7 | hashed_pair | hopfield | 80.000 | 0.835 | 0.194 | 0.022 | 0.500 | 1.000 | 1.000 | 0.835 |
| full_transition_exposure | tomita_7 | hashed_pair | nearest | 80.000 | 0.835 | 0.194 | 0.022 | 0.500 | 1.000 | 1.000 | 0.835 |

### Hashed nearest dense by hidden dimension

| hidden_dim | count | test_string_accuracy_mean | test_string_accuracy_std | test_string_accuracy_sem | test_string_accuracy_min | test_string_accuracy_median | test_string_accuracy_max | test_string_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32.000 | 280.000 | 0.805 | 0.222 | 0.013 | 0.436 | 0.992 | 1.000 | 0.805 |
| 64.000 | 280.000 | 0.875 | 0.196 | 0.012 | 0.500 | 1.000 | 1.000 | 0.875 |
| 128.000 | 280.000 | 0.918 | 0.170 | 0.010 | 0.435 | 1.000 | 1.000 | 0.918 |
| 256.000 | 280.000 | 0.940 | 0.153 | 0.009 | 0.500 | 1.000 | 1.000 | 0.940 |

### Sparse exact-pair nearest autonomous rollout

| write_fraction | unwritten_mode | count | transition_accuracy_autonomous_mean | transition_accuracy_autonomous_std | transition_accuracy_autonomous_sem | transition_accuracy_autonomous_min | transition_accuracy_autonomous_median | transition_accuracy_autonomous_max | transition_accuracy_autonomous |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.300 | random_noise | 280.000 | 0.941 | 0.147 | 0.009 | 0.494 | 1.000 | 1.000 | 0.941 |
| 0.500 | keep_current | 280.000 | 0.427 | 0.258 | 0.015 | 0.000 | 0.382 | 1.000 | 0.427 |
| 0.500 | random_noise | 280.000 | 0.942 | 0.146 | 0.009 | 0.494 | 1.000 | 1.000 | 0.942 |

### Incomplete-coverage exact-pair nearest rows

| grammar_name | transition_coverage_train | count | test_string_accuracy_mean | test_string_accuracy_std | test_string_accuracy_sem | test_string_accuracy_min | test_string_accuracy_median | test_string_accuracy_max | test_string_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contains_101 | 0.125 | 20.000 | 0.500 | 0.001 | 0.000 | 0.500 | 0.500 | 0.502 | 0.500 |
| ends_with_01 | 0.333 | 20.000 | 0.500 | 0.001 | 0.000 | 0.500 | 0.500 | 0.501 | 0.500 |
| even_ones | 0.250 | 20.000 | 0.501 | 0.001 | 0.000 | 0.500 | 0.500 | 0.503 | 0.501 |
| no_substring_11 | 0.167 | 20.000 | 0.502 | 0.003 | 0.001 | 0.500 | 0.500 | 0.509 | 0.502 |
| ones_mod3_zero | 0.167 | 20.000 | 0.502 | 0.001 | 0.000 | 0.500 | 0.502 | 0.503 | 0.502 |
| reber | 0.127 | 4.000 | 0.500 | 0.000 | 0.000 | 0.500 | 0.500 | 0.500 | 0.500 |
| reber | 0.143 | 6.000 | 0.511 | 0.026 | 0.011 | 0.500 | 0.500 | 0.564 | 0.511 |
| reber | 0.159 | 4.000 | 0.532 | 0.037 | 0.019 | 0.500 | 0.532 | 0.565 | 0.532 |
| reber | 0.175 | 6.000 | 0.500 | 0.000 | 0.000 | 0.500 | 0.500 | 0.500 | 0.500 |
| tier_alternating_12 | 0.083 | 20.000 | 0.533 | 0.068 | 0.015 | 0.500 | 0.500 | 0.670 | 0.533 |
| tomita_1 | 0.500 | 20.000 | 0.633 | 0.109 | 0.024 | 0.530 | 0.542 | 0.757 | 0.633 |
| tomita_2 | 0.333 | 20.000 | 0.494 | 0.117 | 0.026 | 0.308 | 0.567 | 0.573 | 0.494 |
| tomita_3 | 0.083 | 20.000 | 0.501 | 0.001 | 0.000 | 0.500 | 0.501 | 0.505 | 0.501 |
| tomita_4 | 0.500 | 20.000 | 0.565 | 0.047 | 0.011 | 0.500 | 0.577 | 0.619 | 0.565 |
| tomita_5 | 0.250 | 20.000 | 0.499 | 0.006 | 0.001 | 0.490 | 0.500 | 0.512 | 0.499 |
| tomita_6 | 0.500 | 20.000 | 0.503 | 0.005 | 0.001 | 0.494 | 0.504 | 0.511 | 0.503 |
| tomita_7 | 0.100 | 20.000 | 0.501 | 0.003 | 0.001 | 0.500 | 0.500 | 0.508 | 0.501 |

### Accuracy by length

| length | count | test_string_accuracy_mean | test_string_accuracy_std | test_string_accuracy_sem | test_string_accuracy_min | test_string_accuracy_median | test_string_accuracy_max | test_string_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000 | 2464.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1.000 | 2752.000 | 0.979 | 0.131 | 0.002 | 0.000 | 1.000 | 1.000 | 0.979 |
| 2.000 | 3520.000 | 0.952 | 0.167 | 0.003 | 0.000 | 1.000 | 1.000 | 0.952 |
| 3.000 | 4320.000 | 0.935 | 0.191 | 0.003 | 0.000 | 1.000 | 1.000 | 0.935 |
| 4.000 | 4416.000 | 0.917 | 0.197 | 0.003 | 0.000 | 1.000 | 1.000 | 0.917 |
| 5.000 | 4480.000 | 0.890 | 0.226 | 0.003 | 0.000 | 1.000 | 1.000 | 0.890 |
| 6.000 | 4480.000 | 0.871 | 0.245 | 0.004 | 0.019 | 1.000 | 1.000 | 0.871 |
| 7.000 | 4480.000 | 0.867 | 0.256 | 0.004 | 0.000 | 1.000 | 1.000 | 0.867 |
| 8.000 | 4480.000 | 0.856 | 0.260 | 0.004 | 0.132 | 1.000 | 1.000 | 0.856 |
| 9.000 | 4480.000 | 0.853 | 0.273 | 0.004 | 0.000 | 1.000 | 1.000 | 0.853 |
| 10.000 | 4480.000 | 0.845 | 0.275 | 0.004 | 0.034 | 1.000 | 1.000 | 0.845 |
| 11.000 | 4480.000 | 0.845 | 0.286 | 0.004 | 0.000 | 1.000 | 1.000 | 0.845 |
| 12.000 | 4480.000 | 0.839 | 0.288 | 0.004 | 0.015 | 1.000 | 1.000 | 0.839 |
| 13.000 | 4480.000 | 0.913 | 0.209 | 0.003 | 0.000 | 1.000 | 1.000 | 0.913 |
| 14.000 | 4480.000 | 0.905 | 0.207 | 0.003 | 0.070 | 1.000 | 1.000 | 0.905 |
| 15.000 | 4480.000 | 0.909 | 0.205 | 0.003 | 0.000 | 1.000 | 1.000 | 0.909 |
| 16.000 | 4480.000 | 0.900 | 0.203 | 0.003 | 0.034 | 1.000 | 1.000 | 0.900 |
| 17.000 | 4480.000 | 0.898 | 0.205 | 0.003 | 0.000 | 1.000 | 1.000 | 0.898 |
| 18.000 | 4480.000 | 0.892 | 0.204 | 0.003 | 0.000 | 1.000 | 1.000 | 0.892 |
| 19.000 | 4480.000 | 0.890 | 0.207 | 0.003 | 0.000 | 1.000 | 1.000 | 0.890 |
| 20.000 | 4480.000 | 0.877 | 0.221 | 0.003 | 0.000 | 1.000 | 1.000 | 0.877 |
| 21.000 | 4480.000 | 0.875 | 0.237 | 0.004 | 0.000 | 1.000 | 1.000 | 0.875 |
| 22.000 | 4480.000 | 0.867 | 0.247 | 0.004 | 0.000 | 1.000 | 1.000 | 0.867 |
| 23.000 | 4480.000 | 0.865 | 0.256 | 0.004 | 0.000 | 1.000 | 1.000 | 0.865 |
| 24.000 | 4480.000 | 0.856 | 0.273 | 0.004 | 0.000 | 1.000 | 1.000 | 0.856 |
| 25.000 | 4480.000 | 0.854 | 0.286 | 0.004 | 0.000 | 1.000 | 1.000 | 0.854 |
| 26.000 | 4480.000 | 0.849 | 0.292 | 0.004 | 0.000 | 1.000 | 1.000 | 0.849 |
| 27.000 | 4480.000 | 0.849 | 0.298 | 0.004 | 0.000 | 1.000 | 1.000 | 0.849 |
| 28.000 | 4480.000 | 0.844 | 0.300 | 0.004 | 0.000 | 1.000 | 1.000 | 0.844 |
| 29.000 | 4480.000 | 0.845 | 0.308 | 0.005 | 0.000 | 1.000 | 1.000 | 0.845 |
| 30.000 | 4480.000 | 0.843 | 0.307 | 0.005 | 0.000 | 1.000 | 1.000 | 0.843 |
| 31.000 | 4480.000 | 0.844 | 0.311 | 0.005 | 0.000 | 1.000 | 1.000 | 0.844 |
| 32.000 | 4480.000 | 0.839 | 0.315 | 0.005 | 0.000 | 1.000 | 1.000 | 0.839 |
| 33.000 | 4480.000 | 0.842 | 0.316 | 0.005 | 0.000 | 1.000 | 1.000 | 0.842 |
| 34.000 | 4480.000 | 0.840 | 0.316 | 0.005 | 0.000 | 1.000 | 1.000 | 0.840 |
| 35.000 | 4480.000 | 0.842 | 0.319 | 0.005 | 0.000 | 1.000 | 1.000 | 0.842 |
| 36.000 | 4480.000 | 0.837 | 0.323 | 0.005 | 0.000 | 1.000 | 1.000 | 0.837 |
| 37.000 | 4480.000 | 0.839 | 0.325 | 0.005 | 0.000 | 1.000 | 1.000 | 0.839 |
| 38.000 | 4480.000 | 0.835 | 0.327 | 0.005 | 0.000 | 1.000 | 1.000 | 0.835 |
| 39.000 | 4480.000 | 0.838 | 0.328 | 0.005 | 0.000 | 1.000 | 1.000 | 0.838 |

_Showing 40 of 65 rows._

### Masked top-k accuracy

| topk_k | count | masked_topk_accuracy_mean | masked_topk_accuracy_std | masked_topk_accuracy_sem | masked_topk_accuracy_min | masked_topk_accuracy_median | masked_topk_accuracy_max | masked_topk_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | 280.000 | 0.470 | 0.275 | 0.016 | 0.000 | 0.417 | 1.000 | 0.470 |
| 2.000 | 280.000 | 0.660 | 0.315 | 0.019 | 0.032 | 0.667 | 1.000 | 0.660 |
| 3.000 | 280.000 | 0.776 | 0.290 | 0.017 | 0.063 | 1.000 | 1.000 | 0.776 |
| 4.000 | 280.000 | 0.854 | 0.260 | 0.016 | 0.079 | 1.000 | 1.000 | 0.854 |
| 5.000 | 280.000 | 0.884 | 0.243 | 0.015 | 0.111 | 1.000 | 1.000 | 0.884 |
| 6.000 | 280.000 | 0.900 | 0.236 | 0.014 | 0.127 | 1.000 | 1.000 | 0.900 |
| 8.000 | 280.000 | 0.912 | 0.224 | 0.013 | 0.143 | 1.000 | 1.000 | 0.912 |
| 10.000 | 280.000 | 0.922 | 0.219 | 0.013 | 0.175 | 1.000 | 1.000 | 0.922 |
| 12.000 | 280.000 | 0.921 | 0.220 | 0.013 | 0.190 | 1.000 | 1.000 | 0.921 |
| 16.000 | 280.000 | 0.920 | 0.223 | 0.013 | 0.190 | 1.000 | 1.000 | 0.920 |
| 24.000 | 280.000 | 0.918 | 0.224 | 0.013 | 0.190 | 1.000 | 1.000 | 0.918 |
| 32.000 | 280.000 | 0.918 | 0.224 | 0.013 | 0.190 | 1.000 | 1.000 | 0.918 |

## Cross-experiment interpretation

| Experiment | Main result | Interpretation | Best figure/table |
| --- | --- | --- | --- |
| Exp01 | Nearest transition accuracy = 1.000; Hopfield is capacity-dependent. | Exact transition construction works; recurrent cleanup imposes capacity limits. | fig01/fig06 and Exp01 summary |
| Exp02 | Nearest robust under corruption; Hopfield lower with monotonic degradation. | State vectors behave as basins; Hopfield implementation has stability/noise limits. | fig02 and Exp02 summary |
| Exp03 | Hopfield near-perfect up to around classical capacity, then degrades. | Capacity limits motivate modularity. | fig03/fig07 and statistics table |
| Exp04 | random_noise sparse transitions improve strongly with write fraction; keep_current is harsher. | Sparse basin targeting is possible but may require reset/gating. | fig04b and Exp04 mode-difference table |
| Exp05 | Protocol cases succeed. | Descriptor/payload distinction is operational. | fig05 and Exp05 summary |
| Exp06 | Learned exact-pair transitions acquire demonstrated FSM associations. | Transition maps can be learned from demonstrations but inherit interface and cleanup capacity limits. | fig08-fig11 and Exp06 summary |
| Exp07 | Structured grammar tasks test length generalization once DFA transitions are covered. | Reusable transition structure can support unseen strings, unlike random FSM tables. | fig12-fig16 and Exp07 report |

## Paper-ready numbers

- Exp01 nearest transition accuracy mean/std/min/max: mean=1.000, std=0.000, min=1.000, max=1.000
- Exp01 Hopfield transition accuracy mean/std/median/min/max: mean=0.874, std=0.222, median=1.000, min=0.141, max=1.000
- Exp01 Hopfield mean accuracy at state_dim=64,num_states=64: 0.319
- Exp01 Hopfield mean accuracy at state_dim=512,num_states=64: 0.999
- Exp02 nearest recovery at flip_fraction=0.00: 1.000
- Exp02 nearest recovery at flip_fraction=0.20: 1.000
- Exp02 nearest recovery at flip_fraction=0.30: 0.996
- Exp02 nearest recovery at flip_fraction=0.40: 0.752
- Exp02 Hopfield recovery at flip_fraction=0.00: 0.844
- Exp02 Hopfield recovery at flip_fraction=0.20: 0.753
- Exp02 Hopfield recovery at flip_fraction=0.30: 0.639
- Exp02 Hopfield recovery at flip_fraction=0.40: 0.389
- Exp02 Hopfield zero-noise recovery by state_dim,num_states: n=64,m=8:1.000; n=64,m=16:0.856; n=64,m=32:0.484; n=64,m=64:0.305; n=128,m=8:1.000; n=128,m=16:0.981; n=128,m=32:0.706; n=128,m=64:0.483; n=256,m=8:1.000; n=256,m=16:1.000; n=256,m=32:0.997; n=256,m=64:0.697; n=512,m=8:1.000; n=512,m=16:1.000; n=512,m=32:1.000; n=512,m=64:1.000
- Exp03 Hopfield recovery by capacity_ratio: 0.025=1.000, 0.050=1.000, 0.100=1.000, 0.138=0.985, 0.200=0.824, 0.300=0.656, 0.500=0.567, 0.750=0.494, 1.000=0.466
- Exp03 nearest recovery by capacity_ratio: 0.025=1.000, 0.050=1.000, 0.100=1.000, 0.138=1.000, 0.200=1.000, 0.300=1.000, 0.500=1.000, 0.750=1.000, 1.000=1.000
- Exp03 first capacity_ratio where Hopfield drops below 0.90: 0.200
- Exp03 first capacity_ratio where Hopfield drops below 0.75: 0.300
- Exp03 Hopfield recovery closest to capacity_ratio=0.138: 0.985
- Exp03 Hopfield recovery closest to capacity_ratio=1.000: 0.466
- Exp04 nearest random_noise sparse accuracy by write_fraction: 0.10=0.488, 0.20=0.877, 0.30=0.982, 0.40=1.000, 0.50=1.000
- Exp04 Hopfield random_noise sparse accuracy by write_fraction: 0.10=0.277, 0.20=0.575, 0.30=0.762, 0.40=0.850, 0.50=0.906
- Exp04 nearest keep_current sparse accuracy at write_fraction 0.50,0.75,1.00: 0.50=0.540, 0.75=1.000, 1.00=1.000
- Exp04 Hopfield keep_current sparse accuracy at write_fraction 0.50,0.75,1.00: 0.50=0.408, 0.75=0.949, 1.00=0.970
- Exp04 nearest random_noise minus keep_current by write_fraction: 0.05=0.157, 0.10=0.410, 0.15=0.629, 0.20=0.799, 0.30=0.904, 0.40=0.918, 0.50=0.460, 0.75=0.000, 1.00=0.000
- Exp04 Hopfield random_noise minus keep_current by write_fraction: 0.05=0.075, 0.10=0.199, 0.15=0.365, 0.20=0.496, 0.30=0.679, 0.40=0.736, 0.50=0.498, 0.75=0.006, 1.00=0.000
- Exp04 nearest random_noise first reaches 0.75 at write_fraction=0.200
- Exp04 nearest random_noise first reaches 0.90 at write_fraction=0.300
- Exp04 nearest random_noise first reaches 0.95 at write_fraction=0.300
- Exp04 hopfield random_noise first reaches 0.75 at write_fraction=0.300
- Exp04 hopfield random_noise first reaches 0.90 at write_fraction=0.500
- Exp04 hopfield random_noise first reaches 0.95 at write_fraction=0.750
- Exp05 protocol success rate across non-NaN success rows: 1.000
- Exp05 success by case_name: missing_descriptor=1.000, same_descriptor_store_p1=1.000, same_descriptor_store_p2=1.000, same_payload_compare=1.000, same_payload_store=1.000, wrong_content_compare=1.000
- Exp05 operation success at corruption_rate=0.0: 1.000
- Exp05 operation success at corruption_rate=0.5: 0.498
- Exp05 operation success at corruption_rate=1.0: 0.000
- Exp05 content success at corruption_rate=0.0: 1.000
- Exp05 content success at corruption_rate=0.5: 0.498
- Exp05 content success at corruption_rate=1.0: 0.000
- Exp06 exact_pair nearest accuracy after one epoch/full table: 1.000
- Exp06 exact_pair Hopfield accuracy after one epoch/full table: 0.909
- Exp06 hashed_pair nearest accuracy by hidden_dim at max epoch: 32.000=0.400, 64.000=0.546, 128.000=0.690, 256.000=0.804, 512.000=0.890, 1024.000=0.939
- Exp06 hashed_pair Hopfield accuracy by hidden_dim at max epoch: 32.000=0.384, 64.000=0.518, 128.000=0.647, 256.000=0.745, 512.000=0.818, 1024.000=0.858
- Exp06 first hidden_dim reaching 0.95 nearest accuracy: not reached
- Exp06 learned sparse nearest accuracy at write_fraction=0.30: 0.999
- Exp06 learned sparse Hopfield accuracy at write_fraction=0.30: 0.945
- Exp06 learned sparse nearest accuracy at write_fraction=0.50: 1.000
- Exp06 learned sparse Hopfield accuracy at write_fraction=0.50: 0.996
- Exp06 nearest seen accuracy at coverage_fraction=0.50: 1.000
- Exp06 nearest unseen accuracy at coverage_fraction=0.50: 0.058
- Exp07 exact_pair nearest test string accuracy: 0.968
- Exp07 exact_pair nearest length-generalization accuracy: 0.967
- Exp07 exact_pair nearest state-tracking accuracy: 0.940
- Exp07 exact_pair Hopfield dense test string accuracy: 0.969
- Exp07 exact_pair Hopfield dense length-generalization accuracy: 0.969
- Exp07 exact_pair Hopfield dense state-tracking accuracy: 0.943
- Exp07 hashed_pair nearest dense test accuracy by hidden_dim: 32.000=0.805, 64.000=0.875, 128.000=0.918, 256.000=0.940
- Exp07 exact_pair nearest sparse_topk autonomous transition accuracy at write_fraction=0.30 random_noise: 0.941
- Exp07 exact_pair nearest sparse_topk autonomous transition accuracy at write_fraction=0.50 random_noise: 0.942
- Exp07 exact_pair nearest sparse_topk autonomous transition accuracy at write_fraction=0.50 keep_current: 0.427
- Exp07 exact_pair nearest transition coverage: 0.912
- Exp07 exact_pair nearest incomplete-coverage rows: 280
- Exp07 exact_pair nearest incomplete-coverage mean test accuracy: 0.518
- Exp07 seen/unseen structured transition accuracy: seen=1.000, unseen=0.075
- Exp07 limited-exposure seen/unseen structured transition accuracy: seen=1.000, unseen=0.299
- Exp07 first masked top-k reaching 0.95: not reached
- Exp07 first masked top-k reaching 0.99: not reached
