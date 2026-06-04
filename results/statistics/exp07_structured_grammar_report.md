# Exp07 Structured Grammar Report

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
