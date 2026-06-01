# Exp07 Structured Grammar Report

## Exp07: Structured grammar learning

- Random transition tables test acquisition and capacity, but structured grammars test reusable transition structure.
- Exact-pair nearest rows are the cleanest test of whether learned DFA transitions support string-level generalization.
- Exact-pair Hopfield rows test whether the same learned transitions inherit recurrent cleanup limits.
- Hashed-pair rows test capacity limits in the state-input context layer.
- Sparse-top-k rows report autonomous rollout accuracy, not only one-step masked cleanup.
- Transition coverage explains most failures: missing DFA edges lead to systematic autonomous rollout errors.
- Masked top-k rows test whether sparse visible coordinates can identify the correct next-state basin.
- Exact-pair nearest test string accuracy: 0.763.
- Exact-pair nearest length-generalization accuracy: 0.794.
- Exact-pair nearest state-tracking accuracy: 0.688.
- Exact-pair Hopfield dense test string accuracy: 1.000.
- Exact-pair Hopfield dense length-generalization accuracy: 1.000.
- Exact-pair Hopfield dense state-tracking accuracy: 1.000.
- Hashed-pair nearest dense test accuracy by hidden_dim: 32.000=0.825, 128.000=0.959.
- Hashed-pair nearest dense mean test accuracy: 0.892.
- Exact-pair nearest sparse-top-k autonomous transition accuracy at write_fraction=0.30 random_noise: 0.998.
- Exact-pair nearest sparse-top-k autonomous transition accuracy at write_fraction=0.50 random_noise: 1.000.
- Exact-pair nearest sparse-top-k autonomous transition accuracy at write_fraction=0.50 keep_current: 0.432.
- Mean training transition coverage: 0.662.
- Exact-pair nearest dense incomplete-coverage rows: 35.
- Incomplete-coverage mean transition coverage: 0.324.
- Incomplete-coverage mean test string accuracy: 0.525.
- Seen/unseen transition accuracy: seen=1.000, unseen=0.276.
- First masked top-k reaching 0.95: 5.000.
- First masked top-k reaching 0.99: 8.000.

### Overall summary by model

| feature_mode | register_type | output_mode | count | test_string_accuracy_mean | test_string_accuracy_std | test_string_accuracy_sem | test_string_accuracy_min | test_string_accuracy_median | test_string_accuracy_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact_pair | hopfield | dense | 35.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | hopfield | sparse_topk | 105.000 | 0.850 | 0.223 | 0.022 | 0.499 | 0.996 | 1.000 |
| exact_pair | nearest | dense | 70.000 | 0.763 | 0.246 | 0.029 | 0.307 | 0.877 | 1.000 |
| exact_pair | nearest | sparse_topk | 105.000 | 0.856 | 0.224 | 0.022 | 0.500 | 1.000 | 1.000 |
| hashed_pair | hopfield | dense | 70.000 | 0.892 | 0.175 | 0.021 | 0.478 | 1.000 | 1.000 |
| hashed_pair | nearest | dense | 70.000 | 0.892 | 0.175 | 0.021 | 0.478 | 1.000 | 1.000 |

### Variant test accuracy

| feature_mode | hidden_dim | register_type | output_mode | write_fraction | unwritten_mode | count | test_string_accuracy_mean | test_string_accuracy_std | test_string_accuracy_sem | test_string_accuracy_min | test_string_accuracy_median | test_string_accuracy_max | test_string_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact_pair | 4.000 | hopfield | dense | 1.000 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 4.000 | hopfield | sparse_topk | 0.300 | random_noise | 5.000 | 1.000 | 0.001 | 0.000 | 0.998 | 1.000 | 1.000 | 1.000 |
| exact_pair | 4.000 | hopfield | sparse_topk | 0.500 | keep_current | 5.000 | 0.800 | 0.274 | 0.122 | 0.500 | 1.000 | 1.000 | 0.800 |
| exact_pair | 4.000 | hopfield | sparse_topk | 0.500 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 4.000 | nearest | dense | 1.000 | random_noise | 10.000 | 0.812 | 0.212 | 0.067 | 0.534 | 0.877 | 1.000 | 0.812 |
| exact_pair | 4.000 | nearest | sparse_topk | 0.300 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 0.999 | 1.000 | 1.000 | 1.000 |
| exact_pair | 4.000 | nearest | sparse_topk | 0.500 | keep_current | 5.000 | 0.800 | 0.274 | 0.122 | 0.500 | 1.000 | 1.000 | 0.800 |
| exact_pair | 4.000 | nearest | sparse_topk | 0.500 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 6.000 | hopfield | dense | 1.000 | random_noise | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 6.000 | hopfield | sparse_topk | 0.300 | random_noise | 10.000 | 0.992 | 0.012 | 0.004 | 0.969 | 0.998 | 0.999 | 0.992 |
| exact_pair | 6.000 | hopfield | sparse_topk | 0.500 | keep_current | 10.000 | 0.519 | 0.032 | 0.010 | 0.499 | 0.500 | 0.574 | 0.519 |
| exact_pair | 6.000 | hopfield | sparse_topk | 0.500 | random_noise | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 6.000 | nearest | dense | 1.000 | random_noise | 20.000 | 0.744 | 0.271 | 0.061 | 0.307 | 0.790 | 1.000 | 0.744 |
| exact_pair | 6.000 | nearest | sparse_topk | 0.300 | random_noise | 10.000 | 0.998 | 0.002 | 0.001 | 0.994 | 0.999 | 1.000 | 0.998 |
| exact_pair | 6.000 | nearest | sparse_topk | 0.500 | keep_current | 10.000 | 0.514 | 0.029 | 0.009 | 0.500 | 0.500 | 0.574 | 0.514 |
| exact_pair | 6.000 | nearest | sparse_topk | 0.500 | random_noise | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 8.000 | hopfield | dense | 1.000 | random_noise | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 8.000 | hopfield | sparse_topk | 0.300 | random_noise | 10.000 | 0.985 | 0.011 | 0.004 | 0.966 | 0.989 | 0.996 | 0.985 |
| exact_pair | 8.000 | hopfield | sparse_topk | 0.500 | keep_current | 10.000 | 0.520 | 0.064 | 0.020 | 0.500 | 0.500 | 0.701 | 0.520 |
| exact_pair | 8.000 | hopfield | sparse_topk | 0.500 | random_noise | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 8.000 | nearest | dense | 1.000 | random_noise | 20.000 | 0.769 | 0.238 | 0.053 | 0.501 | 0.805 | 1.000 | 0.769 |
| exact_pair | 8.000 | nearest | sparse_topk | 0.300 | random_noise | 10.000 | 0.997 | 0.005 | 0.002 | 0.984 | 0.999 | 1.000 | 0.997 |
| exact_pair | 8.000 | nearest | sparse_topk | 0.500 | keep_current | 10.000 | 0.503 | 0.009 | 0.003 | 0.500 | 0.500 | 0.527 | 0.503 |
| exact_pair | 8.000 | nearest | sparse_topk | 0.500 | random_noise | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 10.000 | hopfield | dense | 1.000 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 10.000 | hopfield | sparse_topk | 0.300 | random_noise | 5.000 | 0.994 | 0.008 | 0.004 | 0.980 | 0.998 | 1.000 | 0.994 |
| exact_pair | 10.000 | hopfield | sparse_topk | 0.500 | keep_current | 5.000 | 0.537 | 0.083 | 0.037 | 0.500 | 0.500 | 0.686 | 0.537 |
| exact_pair | 10.000 | hopfield | sparse_topk | 0.500 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 10.000 | nearest | dense | 1.000 | random_noise | 10.000 | 0.750 | 0.264 | 0.083 | 0.500 | 0.750 | 1.000 | 0.750 |
| exact_pair | 10.000 | nearest | sparse_topk | 0.300 | random_noise | 5.000 | 0.999 | 0.002 | 0.001 | 0.996 | 1.000 | 1.000 | 0.999 |
| exact_pair | 10.000 | nearest | sparse_topk | 0.500 | keep_current | 5.000 | 0.600 | 0.224 | 0.100 | 0.500 | 0.500 | 1.000 | 0.600 |
| exact_pair | 10.000 | nearest | sparse_topk | 0.500 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 12.000 | hopfield | dense | 1.000 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 12.000 | hopfield | sparse_topk | 0.300 | random_noise | 5.000 | 0.986 | 0.010 | 0.004 | 0.972 | 0.987 | 0.996 | 0.986 |
| exact_pair | 12.000 | hopfield | sparse_topk | 0.500 | keep_current | 5.000 | 0.500 | 0.000 | 0.000 | 0.500 | 0.500 | 0.500 | 0.500 |
| exact_pair | 12.000 | hopfield | sparse_topk | 0.500 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 12.000 | nearest | dense | 1.000 | random_noise | 10.000 | 0.750 | 0.264 | 0.083 | 0.500 | 0.750 | 1.000 | 0.750 |
| exact_pair | 12.000 | nearest | sparse_topk | 0.300 | random_noise | 5.000 | 0.996 | 0.005 | 0.002 | 0.987 | 0.998 | 0.999 | 0.996 |
| exact_pair | 12.000 | nearest | sparse_topk | 0.500 | keep_current | 5.000 | 0.548 | 0.107 | 0.048 | 0.500 | 0.500 | 0.739 | 0.548 |
| exact_pair | 12.000 | nearest | sparse_topk | 0.500 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hashed_pair | 32.000 | hopfield | dense | 1.000 | random_noise | 35.000 | 0.825 | 0.197 | 0.033 | 0.478 | 0.990 | 1.000 | 0.825 |
| hashed_pair | 32.000 | nearest | dense | 1.000 | random_noise | 35.000 | 0.825 | 0.197 | 0.033 | 0.478 | 0.990 | 1.000 | 0.825 |
| hashed_pair | 128.000 | hopfield | dense | 1.000 | random_noise | 35.000 | 0.959 | 0.120 | 0.020 | 0.532 | 1.000 | 1.000 | 0.959 |
| hashed_pair | 128.000 | nearest | dense | 1.000 | random_noise | 35.000 | 0.959 | 0.120 | 0.020 | 0.532 | 1.000 | 1.000 | 0.959 |

### Variant autonomous transition accuracy

| feature_mode | hidden_dim | register_type | output_mode | write_fraction | unwritten_mode | count | transition_accuracy_autonomous_mean | transition_accuracy_autonomous_std | transition_accuracy_autonomous_sem | transition_accuracy_autonomous_min | transition_accuracy_autonomous_median | transition_accuracy_autonomous_max | transition_accuracy_autonomous |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact_pair | 4.000 | hopfield | dense | 1.000 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 4.000 | hopfield | sparse_topk | 0.300 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 4.000 | hopfield | sparse_topk | 0.500 | keep_current | 5.000 | 0.766 | 0.321 | 0.144 | 0.409 | 1.000 | 1.000 | 0.766 |
| exact_pair | 4.000 | hopfield | sparse_topk | 0.500 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 4.000 | nearest | dense | 1.000 | random_noise | 10.000 | 0.824 | 0.193 | 0.061 | 0.579 | 0.868 | 1.000 | 0.824 |
| exact_pair | 4.000 | nearest | sparse_topk | 0.300 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 4.000 | nearest | sparse_topk | 0.500 | keep_current | 5.000 | 0.766 | 0.321 | 0.144 | 0.409 | 1.000 | 1.000 | 0.766 |
| exact_pair | 4.000 | nearest | sparse_topk | 0.500 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 6.000 | hopfield | dense | 1.000 | random_noise | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 6.000 | hopfield | sparse_topk | 0.300 | random_noise | 10.000 | 0.994 | 0.008 | 0.003 | 0.979 | 0.998 | 1.000 | 0.994 |
| exact_pair | 6.000 | hopfield | sparse_topk | 0.500 | keep_current | 10.000 | 0.376 | 0.140 | 0.044 | 0.195 | 0.343 | 0.582 | 0.376 |
| exact_pair | 6.000 | hopfield | sparse_topk | 0.500 | random_noise | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 6.000 | nearest | dense | 1.000 | random_noise | 20.000 | 0.698 | 0.319 | 0.071 | 0.268 | 0.800 | 1.000 | 0.698 |
| exact_pair | 6.000 | nearest | sparse_topk | 0.300 | random_noise | 10.000 | 0.998 | 0.003 | 0.001 | 0.992 | 0.999 | 1.000 | 0.998 |
| exact_pair | 6.000 | nearest | sparse_topk | 0.500 | keep_current | 10.000 | 0.349 | 0.130 | 0.041 | 0.195 | 0.324 | 0.546 | 0.349 |
| exact_pair | 6.000 | nearest | sparse_topk | 0.500 | random_noise | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 8.000 | hopfield | dense | 1.000 | random_noise | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 8.000 | hopfield | sparse_topk | 0.300 | random_noise | 10.000 | 0.987 | 0.007 | 0.002 | 0.974 | 0.989 | 0.995 | 0.987 |
| exact_pair | 8.000 | hopfield | sparse_topk | 0.500 | keep_current | 10.000 | 0.355 | 0.082 | 0.026 | 0.254 | 0.361 | 0.515 | 0.355 |
| exact_pair | 8.000 | hopfield | sparse_topk | 0.500 | random_noise | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 8.000 | nearest | dense | 1.000 | random_noise | 20.000 | 0.660 | 0.372 | 0.083 | 0.178 | 0.912 | 1.000 | 0.660 |
| exact_pair | 8.000 | nearest | sparse_topk | 0.300 | random_noise | 10.000 | 0.998 | 0.004 | 0.001 | 0.986 | 0.999 | 1.000 | 0.998 |
| exact_pair | 8.000 | nearest | sparse_topk | 0.500 | keep_current | 10.000 | 0.337 | 0.066 | 0.021 | 0.254 | 0.349 | 0.405 | 0.337 |
| exact_pair | 8.000 | nearest | sparse_topk | 0.500 | random_noise | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 10.000 | hopfield | dense | 1.000 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 10.000 | hopfield | sparse_topk | 0.300 | random_noise | 5.000 | 0.995 | 0.007 | 0.003 | 0.983 | 0.997 | 0.999 | 0.995 |
| exact_pair | 10.000 | hopfield | sparse_topk | 0.500 | keep_current | 5.000 | 0.544 | 0.232 | 0.104 | 0.372 | 0.380 | 0.824 | 0.544 |
| exact_pair | 10.000 | hopfield | sparse_topk | 0.500 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 10.000 | nearest | dense | 1.000 | random_noise | 10.000 | 0.633 | 0.395 | 0.125 | 0.137 | 0.689 | 1.000 | 0.633 |
| exact_pair | 10.000 | nearest | sparse_topk | 0.300 | random_noise | 5.000 | 0.999 | 0.001 | 0.000 | 0.997 | 0.999 | 1.000 | 0.999 |
| exact_pair | 10.000 | nearest | sparse_topk | 0.500 | keep_current | 5.000 | 0.532 | 0.345 | 0.154 | 0.144 | 0.373 | 1.000 | 0.532 |
| exact_pair | 10.000 | nearest | sparse_topk | 0.500 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 12.000 | hopfield | dense | 1.000 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 12.000 | hopfield | sparse_topk | 0.300 | random_noise | 5.000 | 0.989 | 0.007 | 0.003 | 0.978 | 0.988 | 0.996 | 0.989 |
| exact_pair | 12.000 | hopfield | sparse_topk | 0.500 | keep_current | 5.000 | 0.299 | 0.115 | 0.052 | 0.215 | 0.231 | 0.487 | 0.299 |
| exact_pair | 12.000 | hopfield | sparse_topk | 0.500 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_pair | 12.000 | nearest | dense | 1.000 | random_noise | 10.000 | 0.642 | 0.380 | 0.120 | 0.210 | 0.668 | 1.000 | 0.642 |
| exact_pair | 12.000 | nearest | sparse_topk | 0.300 | random_noise | 5.000 | 0.997 | 0.003 | 0.001 | 0.992 | 0.998 | 0.999 | 0.997 |
| exact_pair | 12.000 | nearest | sparse_topk | 0.500 | keep_current | 5.000 | 0.355 | 0.124 | 0.055 | 0.214 | 0.333 | 0.555 | 0.355 |
| exact_pair | 12.000 | nearest | sparse_topk | 0.500 | random_noise | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hashed_pair | 32.000 | hopfield | dense | 1.000 | random_noise | 35.000 | 0.835 | 0.190 | 0.032 | 0.377 | 0.889 | 1.000 | 0.835 |
| hashed_pair | 32.000 | nearest | dense | 1.000 | random_noise | 35.000 | 0.835 | 0.190 | 0.032 | 0.377 | 0.889 | 1.000 | 0.835 |
| hashed_pair | 128.000 | hopfield | dense | 1.000 | random_noise | 35.000 | 0.964 | 0.100 | 0.017 | 0.601 | 1.000 | 1.000 | 0.964 |
| hashed_pair | 128.000 | nearest | dense | 1.000 | random_noise | 35.000 | 0.964 | 0.100 | 0.017 | 0.601 | 1.000 | 1.000 | 0.964 |

### Grammar-level accuracy

| grammar_name | feature_mode | register_type | count | test_string_accuracy_mean | test_string_accuracy_std | test_string_accuracy_sem | test_string_accuracy_min | test_string_accuracy_median | test_string_accuracy_max | test_string_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tomita_1 | exact_pair | hopfield | 20.000 | 0.950 | 0.154 | 0.034 | 0.500 | 1.000 | 1.000 | 0.950 |
| tomita_1 | exact_pair | nearest | 25.000 | 0.885 | 0.197 | 0.039 | 0.500 | 1.000 | 1.000 | 0.885 |
| tomita_1 | hashed_pair | hopfield | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| tomita_1 | hashed_pair | nearest | 10.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| tomita_2 | exact_pair | hopfield | 20.000 | 0.883 | 0.204 | 0.046 | 0.500 | 1.000 | 1.000 | 0.883 |
| tomita_2 | exact_pair | nearest | 25.000 | 0.799 | 0.259 | 0.052 | 0.307 | 0.999 | 1.000 | 0.799 |
| tomita_2 | hashed_pair | hopfield | 10.000 | 0.954 | 0.137 | 0.043 | 0.564 | 1.000 | 1.000 | 0.954 |
| tomita_2 | hashed_pair | nearest | 10.000 | 0.954 | 0.137 | 0.043 | 0.564 | 1.000 | 1.000 | 0.954 |
| tomita_3 | exact_pair | hopfield | 20.000 | 0.872 | 0.220 | 0.049 | 0.500 | 0.998 | 1.000 | 0.872 |
| tomita_3 | exact_pair | nearest | 25.000 | 0.809 | 0.242 | 0.048 | 0.500 | 0.998 | 1.000 | 0.809 |
| tomita_3 | hashed_pair | hopfield | 10.000 | 0.908 | 0.158 | 0.050 | 0.568 | 1.000 | 1.000 | 0.908 |
| tomita_3 | hashed_pair | nearest | 10.000 | 0.908 | 0.158 | 0.050 | 0.568 | 1.000 | 1.000 | 0.908 |
| tomita_4 | exact_pair | hopfield | 20.000 | 0.871 | 0.220 | 0.049 | 0.500 | 0.998 | 1.000 | 0.871 |
| tomita_4 | exact_pair | nearest | 25.000 | 0.814 | 0.233 | 0.047 | 0.500 | 0.999 | 1.000 | 0.814 |
| tomita_4 | hashed_pair | hopfield | 10.000 | 0.827 | 0.226 | 0.072 | 0.500 | 1.000 | 1.000 | 0.827 |
| tomita_4 | hashed_pair | nearest | 10.000 | 0.827 | 0.226 | 0.072 | 0.500 | 1.000 | 1.000 | 0.827 |
| tomita_5 | exact_pair | hopfield | 20.000 | 0.881 | 0.206 | 0.046 | 0.500 | 0.998 | 1.000 | 0.881 |
| tomita_5 | exact_pair | nearest | 25.000 | 0.802 | 0.247 | 0.049 | 0.500 | 1.000 | 1.000 | 0.802 |
| tomita_5 | hashed_pair | hopfield | 10.000 | 0.862 | 0.185 | 0.058 | 0.570 | 1.000 | 1.000 | 0.862 |
| tomita_5 | hashed_pair | nearest | 10.000 | 0.862 | 0.185 | 0.058 | 0.570 | 1.000 | 1.000 | 0.862 |
| tomita_6 | exact_pair | hopfield | 20.000 | 0.873 | 0.221 | 0.049 | 0.499 | 0.999 | 1.000 | 0.873 |
| tomita_6 | exact_pair | nearest | 25.000 | 0.801 | 0.248 | 0.050 | 0.484 | 0.999 | 1.000 | 0.801 |
| tomita_6 | hashed_pair | hopfield | 10.000 | 0.948 | 0.165 | 0.052 | 0.478 | 1.000 | 1.000 | 0.948 |
| tomita_6 | hashed_pair | nearest | 10.000 | 0.948 | 0.165 | 0.052 | 0.478 | 1.000 | 1.000 | 0.948 |
| tomita_7 | exact_pair | hopfield | 20.000 | 0.883 | 0.208 | 0.047 | 0.500 | 1.000 | 1.000 | 0.883 |
| tomita_7 | exact_pair | nearest | 25.000 | 0.820 | 0.245 | 0.049 | 0.500 | 1.000 | 1.000 | 0.820 |
| tomita_7 | hashed_pair | hopfield | 10.000 | 0.748 | 0.171 | 0.054 | 0.509 | 0.746 | 1.000 | 0.748 |
| tomita_7 | hashed_pair | nearest | 10.000 | 0.748 | 0.171 | 0.054 | 0.509 | 0.746 | 1.000 | 0.748 |

### Hashed nearest dense by hidden dimension

| hidden_dim | count | test_string_accuracy_mean | test_string_accuracy_std | test_string_accuracy_sem | test_string_accuracy_min | test_string_accuracy_median | test_string_accuracy_max | test_string_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32.000 | 35.000 | 0.825 | 0.197 | 0.033 | 0.478 | 0.990 | 1.000 | 0.825 |
| 128.000 | 35.000 | 0.959 | 0.120 | 0.020 | 0.532 | 1.000 | 1.000 | 0.959 |

### Sparse exact-pair nearest autonomous rollout

| write_fraction | unwritten_mode | count | transition_accuracy_autonomous_mean | transition_accuracy_autonomous_std | transition_accuracy_autonomous_sem | transition_accuracy_autonomous_min | transition_accuracy_autonomous_median | transition_accuracy_autonomous_max | transition_accuracy_autonomous |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.300 | random_noise | 35.000 | 0.998 | 0.003 | 0.000 | 0.986 | 0.999 | 1.000 | 0.998 |
| 0.500 | keep_current | 35.000 | 0.432 | 0.239 | 0.040 | 0.144 | 0.373 | 1.000 | 0.432 |
| 0.500 | random_noise | 35.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### Incomplete-coverage exact-pair nearest rows

| grammar_name | transition_coverage_train | count | test_string_accuracy_mean | test_string_accuracy_std | test_string_accuracy_sem | test_string_accuracy_min | test_string_accuracy_median | test_string_accuracy_max | test_string_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tomita_1 | 0.500 | 5.000 | 0.624 | 0.116 | 0.052 | 0.534 | 0.548 | 0.755 | 0.624 |
| tomita_2 | 0.333 | 5.000 | 0.467 | 0.140 | 0.063 | 0.307 | 0.557 | 0.580 | 0.467 |
| tomita_3 | 0.083 | 5.000 | 0.500 | 0.000 | 0.000 | 0.500 | 0.500 | 0.500 | 0.500 |
| tomita_4 | 0.500 | 5.000 | 0.573 | 0.029 | 0.013 | 0.537 | 0.582 | 0.611 | 0.573 |
| tomita_5 | 0.250 | 5.000 | 0.505 | 0.004 | 0.002 | 0.501 | 0.503 | 0.509 | 0.505 |
| tomita_6 | 0.500 | 5.000 | 0.509 | 0.017 | 0.008 | 0.484 | 0.514 | 0.528 | 0.509 |
| tomita_7 | 0.100 | 5.000 | 0.500 | 0.000 | 0.000 | 0.500 | 0.500 | 0.500 | 0.500 |

### Accuracy by length

| length | count | test_string_accuracy_mean | test_string_accuracy_std | test_string_accuracy_sem | test_string_accuracy_min | test_string_accuracy_median | test_string_accuracy_max | test_string_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000 | 170.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1.000 | 238.000 | 0.954 | 0.210 | 0.014 | 0.000 | 1.000 | 1.000 | 0.954 |
| 2.000 | 355.000 | 0.917 | 0.266 | 0.014 | 0.000 | 1.000 | 1.000 | 0.917 |
| 3.000 | 266.000 | 0.928 | 0.235 | 0.014 | 0.000 | 1.000 | 1.000 | 0.928 |
| 4.000 | 417.000 | 0.888 | 0.280 | 0.014 | 0.000 | 1.000 | 1.000 | 0.888 |
| 5.000 | 451.000 | 0.911 | 0.230 | 0.011 | 0.000 | 1.000 | 1.000 | 0.911 |
| 6.000 | 455.000 | 0.881 | 0.247 | 0.012 | 0.014 | 1.000 | 1.000 | 0.881 |
| 7.000 | 455.000 | 0.892 | 0.229 | 0.011 | 0.000 | 1.000 | 1.000 | 0.892 |
| 8.000 | 455.000 | 0.866 | 0.236 | 0.011 | 0.110 | 1.000 | 1.000 | 0.866 |
| 9.000 | 455.000 | 0.882 | 0.225 | 0.011 | 0.000 | 1.000 | 1.000 | 0.882 |
| 10.000 | 455.000 | 0.859 | 0.221 | 0.010 | 0.248 | 1.000 | 1.000 | 0.859 |
| 11.000 | 455.000 | 0.873 | 0.233 | 0.011 | 0.000 | 1.000 | 1.000 | 0.873 |
| 12.000 | 455.000 | 0.854 | 0.236 | 0.011 | 0.138 | 1.000 | 1.000 | 0.854 |
| 13.000 | 455.000 | 0.885 | 0.235 | 0.011 | 0.000 | 1.000 | 1.000 | 0.885 |
| 14.000 | 455.000 | 0.867 | 0.233 | 0.011 | 0.153 | 1.000 | 1.000 | 0.867 |
| 15.000 | 455.000 | 0.883 | 0.230 | 0.011 | 0.000 | 1.000 | 1.000 | 0.883 |
| 16.000 | 455.000 | 0.867 | 0.232 | 0.011 | 0.020 | 1.000 | 1.000 | 0.867 |
| 17.000 | 455.000 | 0.873 | 0.241 | 0.011 | 0.000 | 1.000 | 1.000 | 0.873 |
| 18.000 | 455.000 | 0.855 | 0.252 | 0.012 | 0.000 | 1.000 | 1.000 | 0.855 |
| 19.000 | 455.000 | 0.868 | 0.252 | 0.012 | 0.000 | 1.000 | 1.000 | 0.868 |
| 20.000 | 455.000 | 0.856 | 0.262 | 0.012 | 0.000 | 1.000 | 1.000 | 0.856 |
| 21.000 | 455.000 | 0.865 | 0.270 | 0.013 | 0.000 | 1.000 | 1.000 | 0.865 |
| 22.000 | 455.000 | 0.857 | 0.270 | 0.013 | 0.000 | 1.000 | 1.000 | 0.857 |
| 23.000 | 455.000 | 0.861 | 0.282 | 0.013 | 0.000 | 1.000 | 1.000 | 0.861 |
| 24.000 | 455.000 | 0.849 | 0.283 | 0.013 | 0.000 | 1.000 | 1.000 | 0.849 |
| 25.000 | 455.000 | 0.865 | 0.277 | 0.013 | 0.000 | 1.000 | 1.000 | 0.865 |
| 26.000 | 455.000 | 0.845 | 0.291 | 0.014 | 0.000 | 1.000 | 1.000 | 0.845 |
| 27.000 | 455.000 | 0.858 | 0.288 | 0.014 | 0.000 | 1.000 | 1.000 | 0.858 |
| 28.000 | 455.000 | 0.846 | 0.295 | 0.014 | 0.000 | 1.000 | 1.000 | 0.846 |
| 29.000 | 455.000 | 0.860 | 0.290 | 0.014 | 0.000 | 1.000 | 1.000 | 0.860 |
| 30.000 | 455.000 | 0.844 | 0.298 | 0.014 | 0.000 | 1.000 | 1.000 | 0.844 |
| 31.000 | 455.000 | 0.858 | 0.293 | 0.014 | 0.000 | 1.000 | 1.000 | 0.858 |
| 32.000 | 455.000 | 0.840 | 0.307 | 0.014 | 0.000 | 1.000 | 1.000 | 0.840 |

### Masked top-k accuracy

| topk_k | count | masked_topk_accuracy_mean | masked_topk_accuracy_std | masked_topk_accuracy_sem | masked_topk_accuracy_min | masked_topk_accuracy_median | masked_topk_accuracy_max | masked_topk_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | 280.000 | 0.509 | 0.259 | 0.016 | 0.100 | 0.500 | 1.000 | 0.509 |
| 2.000 | 280.000 | 0.656 | 0.261 | 0.016 | 0.250 | 0.625 | 1.000 | 0.656 |
| 3.000 | 280.000 | 0.773 | 0.242 | 0.014 | 0.250 | 0.800 | 1.000 | 0.773 |
| 4.000 | 280.000 | 0.905 | 0.167 | 0.010 | 0.333 | 1.000 | 1.000 | 0.905 |
| 5.000 | 280.000 | 0.954 | 0.140 | 0.008 | 0.333 | 1.000 | 1.000 | 0.954 |
| 6.000 | 280.000 | 0.971 | 0.123 | 0.007 | 0.333 | 1.000 | 1.000 | 0.971 |
| 8.000 | 280.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 10.000 | 280.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 12.000 | 280.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 16.000 | 280.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 24.000 | 280.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 32.000 | 280.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
