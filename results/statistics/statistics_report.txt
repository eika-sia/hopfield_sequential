# Simulation Statistics Report

## Data files

Loaded CSVs:
- `results/csv/exp01_transition_accuracy.csv`
- `results/csv/exp02_noise_recovery.csv`
- `results/csv/exp03_capacity.csv`
- `results/csv/exp04_sparse_transitions.csv`
- `results/csv/exp05_descriptor_payload.csv`

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

## Cross-experiment interpretation

| Experiment | Main result | Interpretation | Best figure/table |
| --- | --- | --- | --- |
| Exp01 | Nearest transition accuracy = 1.000; Hopfield is capacity-dependent. | Exact transition construction works; recurrent cleanup imposes capacity limits. | fig01/fig06 and Exp01 summary |
| Exp02 | Nearest robust under corruption; Hopfield lower with monotonic degradation. | State vectors behave as basins; Hopfield implementation has stability/noise limits. | fig02 and Exp02 summary |
| Exp03 | Hopfield near-perfect up to around classical capacity, then degrades. | Capacity limits motivate modularity. | fig03/fig07 and statistics table |
| Exp04 | random_noise sparse transitions improve strongly with write fraction; keep_current is harsher. | Sparse basin targeting is possible but may require reset/gating. | fig04b and Exp04 mode-difference table |
| Exp05 | Protocol cases succeed. | Descriptor/payload distinction is operational. | fig05 and Exp05 summary |

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
