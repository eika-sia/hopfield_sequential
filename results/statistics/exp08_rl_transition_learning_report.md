# Exp08 RL Transition Learning Report

## Exp08: RL transition learning

- Exp08 removes the omniscient transition-vector teacher in RL conditions.
- The transition writer is updated from scalar TD error and the executed next-state basin; hidden DFA states are used only for scalar state-shaped reward and evaluation.
- The final accept/reject readout is supervised from the episode label; the transition writer is not given true next-state vectors in RL conditions.
- supervised exact-pair final test accuracy: 1.000.
- state-shaped RL exact-pair final test accuracy: 0.831.
- terminal-only RL exact-pair final test accuracy: 0.673.
- sparse terminal RL exact-pair final test accuracy: 0.622.
- state-shaped RL exact-pair transition accuracy: 0.835.
- terminal-only RL exact-pair transition accuracy: 0.389.
- seen/unseen transition accuracy across Exp08: seen=0.645, unseen=0.218.

### Final accuracy by condition

| condition | cleanup | feature_type | count | final_test_accuracy_mean | final_test_accuracy_std | final_test_accuracy_sem | final_test_accuracy_min | final_test_accuracy_median | final_test_accuracy_max | final_test_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_terminal_rl | hopfield | exact_pair | 70.000 | 0.624 | 0.200 | 0.024 | 0.364 | 0.500 | 1.000 | 0.624 |
| sparse_terminal_rl | nearest | exact_pair | 70.000 | 0.621 | 0.192 | 0.023 | 0.383 | 0.516 | 1.000 | 0.621 |
| state_shaped_rl | hopfield | exact_pair | 35.000 | 0.830 | 0.210 | 0.036 | 0.373 | 0.988 | 1.000 | 0.830 |
| state_shaped_rl | hopfield | hashed_pair | 35.000 | 0.808 | 0.227 | 0.038 | 0.369 | 0.884 | 1.000 | 0.808 |
| state_shaped_rl | nearest | exact_pair | 35.000 | 0.832 | 0.219 | 0.037 | 0.453 | 0.997 | 1.000 | 0.832 |
| state_shaped_rl | nearest | hashed_pair | 35.000 | 0.759 | 0.218 | 0.037 | 0.373 | 0.832 | 1.000 | 0.759 |
| supervised_baseline | hopfield | exact_pair | 35.000 | 1.000 | 0.002 | 0.000 | 0.988 | 1.000 | 1.000 | 1.000 |
| supervised_baseline | hopfield | hashed_pair | 35.000 | 0.839 | 0.200 | 0.034 | 0.287 | 0.934 | 1.000 | 0.839 |
| supervised_baseline | nearest | exact_pair | 35.000 | 1.000 | 0.000 | 0.000 | 0.999 | 1.000 | 1.000 | 1.000 |
| supervised_baseline | nearest | hashed_pair | 35.000 | 0.861 | 0.189 | 0.032 | 0.493 | 1.000 | 1.000 | 0.861 |
| terminal_rl | hopfield | exact_pair | 35.000 | 0.671 | 0.196 | 0.033 | 0.337 | 0.591 | 1.000 | 0.671 |
| terminal_rl | hopfield | hashed_pair | 35.000 | 0.622 | 0.178 | 0.030 | 0.362 | 0.518 | 1.000 | 0.622 |
| terminal_rl | nearest | exact_pair | 35.000 | 0.674 | 0.203 | 0.034 | 0.487 | 0.540 | 1.000 | 0.674 |
| terminal_rl | nearest | hashed_pair | 35.000 | 0.645 | 0.193 | 0.033 | 0.337 | 0.520 | 1.000 | 0.645 |

### Grammar-level final accuracy

| grammar | condition | cleanup | feature_type | count | final_test_accuracy_mean | final_test_accuracy_std | final_test_accuracy_sem | final_test_accuracy_min | final_test_accuracy_median | final_test_accuracy_max | final_test_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contains_101 | sparse_terminal_rl | hopfield | exact_pair | 10.000 | 0.487 | 0.030 | 0.009 | 0.424 | 0.500 | 0.517 | 0.487 |
| contains_101 | sparse_terminal_rl | nearest | exact_pair | 10.000 | 0.507 | 0.053 | 0.017 | 0.400 | 0.512 | 0.577 | 0.507 |
| contains_101 | state_shaped_rl | hopfield | exact_pair | 5.000 | 0.563 | 0.079 | 0.035 | 0.500 | 0.549 | 0.692 | 0.563 |
| contains_101 | state_shaped_rl | hopfield | hashed_pair | 5.000 | 0.859 | 0.222 | 0.099 | 0.493 | 1.000 | 1.000 | 0.859 |
| contains_101 | state_shaped_rl | nearest | exact_pair | 5.000 | 0.705 | 0.272 | 0.122 | 0.453 | 0.573 | 1.000 | 0.705 |
| contains_101 | state_shaped_rl | nearest | hashed_pair | 5.000 | 0.549 | 0.058 | 0.026 | 0.500 | 0.526 | 0.631 | 0.549 |
| contains_101 | supervised_baseline | hopfield | exact_pair | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| contains_101 | supervised_baseline | hopfield | hashed_pair | 5.000 | 0.579 | 0.198 | 0.088 | 0.287 | 0.602 | 0.796 | 0.579 |
| contains_101 | supervised_baseline | nearest | exact_pair | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| contains_101 | supervised_baseline | nearest | hashed_pair | 5.000 | 0.921 | 0.177 | 0.079 | 0.605 | 1.000 | 1.000 | 0.921 |
| contains_101 | terminal_rl | hopfield | exact_pair | 5.000 | 0.512 | 0.029 | 0.013 | 0.497 | 0.500 | 0.565 | 0.512 |
| contains_101 | terminal_rl | hopfield | hashed_pair | 5.000 | 0.514 | 0.039 | 0.018 | 0.463 | 0.500 | 0.559 | 0.514 |
| contains_101 | terminal_rl | nearest | exact_pair | 5.000 | 0.527 | 0.043 | 0.019 | 0.500 | 0.510 | 0.602 | 0.527 |
| contains_101 | terminal_rl | nearest | hashed_pair | 5.000 | 0.504 | 0.009 | 0.004 | 0.500 | 0.500 | 0.520 | 0.504 |
| even_ones | sparse_terminal_rl | hopfield | exact_pair | 10.000 | 0.501 | 0.022 | 0.007 | 0.466 | 0.500 | 0.541 | 0.501 |
| even_ones | sparse_terminal_rl | nearest | exact_pair | 10.000 | 0.494 | 0.023 | 0.007 | 0.468 | 0.492 | 0.531 | 0.494 |
| even_ones | state_shaped_rl | hopfield | exact_pair | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| even_ones | state_shaped_rl | hopfield | hashed_pair | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| even_ones | state_shaped_rl | nearest | exact_pair | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| even_ones | state_shaped_rl | nearest | hashed_pair | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| even_ones | supervised_baseline | hopfield | exact_pair | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| even_ones | supervised_baseline | hopfield | hashed_pair | 5.000 | 0.904 | 0.214 | 0.096 | 0.521 | 1.000 | 1.000 | 0.904 |
| even_ones | supervised_baseline | nearest | exact_pair | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| even_ones | supervised_baseline | nearest | hashed_pair | 5.000 | 0.814 | 0.254 | 0.114 | 0.532 | 1.000 | 1.000 | 0.814 |
| even_ones | terminal_rl | hopfield | exact_pair | 5.000 | 0.506 | 0.013 | 0.006 | 0.500 | 0.500 | 0.529 | 0.506 |
| even_ones | terminal_rl | hopfield | hashed_pair | 5.000 | 0.507 | 0.009 | 0.004 | 0.500 | 0.500 | 0.518 | 0.507 |
| even_ones | terminal_rl | nearest | exact_pair | 5.000 | 0.608 | 0.220 | 0.098 | 0.500 | 0.500 | 1.000 | 0.608 |
| even_ones | terminal_rl | nearest | hashed_pair | 5.000 | 0.506 | 0.008 | 0.004 | 0.500 | 0.500 | 0.518 | 0.506 |
| no_substring_11 | sparse_terminal_rl | hopfield | exact_pair | 10.000 | 0.469 | 0.055 | 0.017 | 0.410 | 0.471 | 0.579 | 0.469 |
| no_substring_11 | sparse_terminal_rl | nearest | exact_pair | 10.000 | 0.521 | 0.069 | 0.022 | 0.412 | 0.500 | 0.627 | 0.521 |
| no_substring_11 | state_shaped_rl | hopfield | exact_pair | 5.000 | 0.692 | 0.291 | 0.130 | 0.373 | 0.588 | 1.000 | 0.692 |
| no_substring_11 | state_shaped_rl | hopfield | hashed_pair | 5.000 | 0.509 | 0.127 | 0.057 | 0.373 | 0.500 | 0.680 | 0.509 |
| no_substring_11 | state_shaped_rl | nearest | exact_pair | 5.000 | 0.500 | 0.000 | 0.000 | 0.500 | 0.500 | 0.500 | 0.500 |
| no_substring_11 | state_shaped_rl | nearest | hashed_pair | 5.000 | 0.575 | 0.244 | 0.109 | 0.373 | 0.500 | 1.000 | 0.575 |
| no_substring_11 | supervised_baseline | hopfield | exact_pair | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| no_substring_11 | supervised_baseline | hopfield | hashed_pair | 5.000 | 0.936 | 0.144 | 0.064 | 0.678 | 1.000 | 1.000 | 0.936 |
| no_substring_11 | supervised_baseline | nearest | exact_pair | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| no_substring_11 | supervised_baseline | nearest | hashed_pair | 5.000 | 0.938 | 0.139 | 0.062 | 0.690 | 1.000 | 1.000 | 0.938 |
| no_substring_11 | terminal_rl | hopfield | exact_pair | 5.000 | 0.544 | 0.042 | 0.019 | 0.500 | 0.555 | 0.594 | 0.544 |
| no_substring_11 | terminal_rl | hopfield | hashed_pair | 5.000 | 0.457 | 0.055 | 0.024 | 0.362 | 0.474 | 0.500 | 0.457 |
| no_substring_11 | terminal_rl | nearest | exact_pair | 5.000 | 0.512 | 0.035 | 0.016 | 0.487 | 0.500 | 0.575 | 0.512 |
| no_substring_11 | terminal_rl | nearest | hashed_pair | 5.000 | 0.494 | 0.102 | 0.046 | 0.337 | 0.500 | 0.594 | 0.494 |
| reber | sparse_terminal_rl | hopfield | exact_pair | 10.000 | 0.607 | 0.189 | 0.060 | 0.364 | 0.605 | 0.852 | 0.607 |
| reber | sparse_terminal_rl | nearest | exact_pair | 10.000 | 0.541 | 0.114 | 0.036 | 0.383 | 0.503 | 0.734 | 0.541 |
| reber | state_shaped_rl | hopfield | exact_pair | 5.000 | 0.864 | 0.126 | 0.057 | 0.718 | 0.835 | 0.995 | 0.864 |
| reber | state_shaped_rl | hopfield | hashed_pair | 5.000 | 0.496 | 0.080 | 0.036 | 0.369 | 0.516 | 0.590 | 0.496 |
| reber | state_shaped_rl | nearest | exact_pair | 5.000 | 0.999 | 0.002 | 0.001 | 0.997 | 1.000 | 1.000 | 0.999 |
| reber | state_shaped_rl | nearest | hashed_pair | 5.000 | 0.604 | 0.086 | 0.039 | 0.533 | 0.570 | 0.732 | 0.604 |
| reber | supervised_baseline | hopfield | exact_pair | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| reber | supervised_baseline | hopfield | hashed_pair | 5.000 | 0.737 | 0.188 | 0.084 | 0.512 | 0.682 | 0.953 | 0.737 |
| reber | supervised_baseline | nearest | exact_pair | 5.000 | 1.000 | 0.001 | 0.000 | 0.999 | 1.000 | 1.000 | 1.000 |
| reber | supervised_baseline | nearest | hashed_pair | 5.000 | 0.595 | 0.160 | 0.072 | 0.493 | 0.523 | 0.874 | 0.595 |
| reber | terminal_rl | hopfield | exact_pair | 5.000 | 0.760 | 0.085 | 0.038 | 0.642 | 0.795 | 0.849 | 0.760 |
| reber | terminal_rl | hopfield | hashed_pair | 5.000 | 0.727 | 0.118 | 0.053 | 0.623 | 0.710 | 0.922 | 0.727 |
| reber | terminal_rl | nearest | exact_pair | 5.000 | 0.811 | 0.180 | 0.081 | 0.500 | 0.861 | 0.938 | 0.811 |
| reber | terminal_rl | nearest | hashed_pair | 5.000 | 0.619 | 0.152 | 0.068 | 0.417 | 0.676 | 0.764 | 0.619 |
| tier_alternating_12 | sparse_terminal_rl | hopfield | exact_pair | 10.000 | 0.702 | 0.123 | 0.039 | 0.500 | 0.733 | 0.829 | 0.702 |
| tier_alternating_12 | sparse_terminal_rl | nearest | exact_pair | 10.000 | 0.663 | 0.146 | 0.046 | 0.387 | 0.732 | 0.813 | 0.663 |
| tier_alternating_12 | state_shaped_rl | hopfield | exact_pair | 5.000 | 0.816 | 0.188 | 0.084 | 0.500 | 0.880 | 1.000 | 0.816 |
| tier_alternating_12 | state_shaped_rl | hopfield | hashed_pair | 5.000 | 0.839 | 0.071 | 0.032 | 0.755 | 0.825 | 0.934 | 0.839 |
| tier_alternating_12 | state_shaped_rl | nearest | exact_pair | 5.000 | 0.880 | 0.079 | 0.035 | 0.821 | 0.829 | 1.000 | 0.880 |
| tier_alternating_12 | state_shaped_rl | nearest | hashed_pair | 5.000 | 0.835 | 0.024 | 0.011 | 0.796 | 0.839 | 0.857 | 0.835 |
| tier_alternating_12 | supervised_baseline | hopfield | exact_pair | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| tier_alternating_12 | supervised_baseline | hopfield | hashed_pair | 5.000 | 0.879 | 0.066 | 0.029 | 0.786 | 0.900 | 0.940 | 0.879 |
| tier_alternating_12 | supervised_baseline | nearest | exact_pair | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| tier_alternating_12 | supervised_baseline | nearest | hashed_pair | 5.000 | 0.845 | 0.143 | 0.064 | 0.672 | 0.899 | 1.000 | 0.845 |
| tier_alternating_12 | terminal_rl | hopfield | exact_pair | 5.000 | 0.744 | 0.091 | 0.041 | 0.591 | 0.771 | 0.831 | 0.744 |
| tier_alternating_12 | terminal_rl | hopfield | hashed_pair | 5.000 | 0.697 | 0.116 | 0.052 | 0.500 | 0.730 | 0.782 | 0.697 |
| tier_alternating_12 | terminal_rl | nearest | exact_pair | 5.000 | 0.778 | 0.100 | 0.045 | 0.641 | 0.779 | 0.899 | 0.778 |
| tier_alternating_12 | terminal_rl | nearest | hashed_pair | 5.000 | 0.727 | 0.134 | 0.060 | 0.500 | 0.783 | 0.845 | 0.727 |
| tomita_1 | sparse_terminal_rl | hopfield | exact_pair | 10.000 | 0.900 | 0.211 | 0.067 | 0.500 | 1.000 | 1.000 | 0.900 |
| tomita_1 | sparse_terminal_rl | nearest | exact_pair | 10.000 | 0.950 | 0.158 | 0.050 | 0.500 | 1.000 | 1.000 | 0.950 |
| tomita_1 | state_shaped_rl | hopfield | exact_pair | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| tomita_1 | state_shaped_rl | hopfield | hashed_pair | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| tomita_1 | state_shaped_rl | nearest | exact_pair | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| tomita_1 | state_shaped_rl | nearest | hashed_pair | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| tomita_1 | supervised_baseline | hopfield | exact_pair | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| tomita_1 | supervised_baseline | hopfield | hashed_pair | 5.000 | 0.952 | 0.108 | 0.048 | 0.758 | 1.000 | 1.000 | 0.952 |
| tomita_1 | supervised_baseline | nearest | exact_pair | 5.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| tomita_1 | supervised_baseline | nearest | hashed_pair | 5.000 | 0.951 | 0.110 | 0.049 | 0.755 | 1.000 | 1.000 | 0.951 |

_Showing 80 of 98 rows._

### State tracking by condition

| condition | cleanup | feature_type | count | state_tracking_accuracy_mean | state_tracking_accuracy_std | state_tracking_accuracy_sem | state_tracking_accuracy_min | state_tracking_accuracy_median | state_tracking_accuracy_max | state_tracking_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_terminal_rl | hopfield | exact_pair | 70.000 | 0.432 | 0.328 | 0.039 | 0.039 | 0.351 | 1.000 | 0.432 |
| sparse_terminal_rl | nearest | exact_pair | 70.000 | 0.403 | 0.320 | 0.038 | 0.000 | 0.336 | 1.000 | 0.403 |
| state_shaped_rl | hopfield | exact_pair | 35.000 | 0.800 | 0.276 | 0.047 | 0.043 | 0.935 | 1.000 | 0.800 |
| state_shaped_rl | hopfield | hashed_pair | 35.000 | 0.821 | 0.251 | 0.042 | 0.111 | 0.922 | 1.000 | 0.821 |
| state_shaped_rl | nearest | exact_pair | 35.000 | 0.811 | 0.269 | 0.046 | 0.233 | 0.954 | 1.000 | 0.811 |
| state_shaped_rl | nearest | hashed_pair | 35.000 | 0.771 | 0.237 | 0.040 | 0.179 | 0.832 | 1.000 | 0.771 |
| supervised_baseline | hopfield | exact_pair | 35.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| supervised_baseline | hopfield | hashed_pair | 35.000 | 0.816 | 0.196 | 0.033 | 0.484 | 0.877 | 1.000 | 0.816 |
| supervised_baseline | nearest | exact_pair | 35.000 | 1.000 | 0.000 | 0.000 | 0.998 | 1.000 | 1.000 | 1.000 |
| supervised_baseline | nearest | hashed_pair | 35.000 | 0.864 | 0.188 | 0.032 | 0.415 | 1.000 | 1.000 | 0.864 |
| terminal_rl | hopfield | exact_pair | 35.000 | 0.377 | 0.330 | 0.056 | 0.000 | 0.333 | 1.000 | 0.377 |
| terminal_rl | hopfield | hashed_pair | 35.000 | 0.321 | 0.298 | 0.050 | 0.000 | 0.220 | 1.000 | 0.321 |
| terminal_rl | nearest | exact_pair | 35.000 | 0.402 | 0.336 | 0.057 | 0.003 | 0.357 | 1.000 | 0.402 |
| terminal_rl | nearest | hashed_pair | 35.000 | 0.455 | 0.331 | 0.056 | 0.013 | 0.478 | 1.000 | 0.455 |

### Transition accuracy by condition

| condition | cleanup | feature_type | count | transition_accuracy_mean | transition_accuracy_std | transition_accuracy_sem | transition_accuracy_min | transition_accuracy_median | transition_accuracy_max | transition_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_terminal_rl | hopfield | exact_pair | 70.000 | 0.427 | 0.282 | 0.034 | 0.032 | 0.333 | 1.000 | 0.427 |
| sparse_terminal_rl | nearest | exact_pair | 70.000 | 0.402 | 0.290 | 0.035 | 0.000 | 0.333 | 1.000 | 0.402 |
| state_shaped_rl | hopfield | exact_pair | 35.000 | 0.811 | 0.261 | 0.044 | 0.016 | 0.917 | 1.000 | 0.811 |
| state_shaped_rl | hopfield | hashed_pair | 35.000 | 0.810 | 0.246 | 0.042 | 0.079 | 0.917 | 1.000 | 0.810 |
| state_shaped_rl | nearest | exact_pair | 35.000 | 0.858 | 0.145 | 0.024 | 0.587 | 0.875 | 1.000 | 0.858 |
| state_shaped_rl | nearest | hashed_pair | 35.000 | 0.797 | 0.181 | 0.031 | 0.500 | 0.833 | 1.000 | 0.797 |
| supervised_baseline | hopfield | exact_pair | 35.000 | 0.971 | 0.083 | 0.014 | 0.698 | 1.000 | 1.000 | 0.971 |
| supervised_baseline | hopfield | hashed_pair | 35.000 | 0.879 | 0.134 | 0.023 | 0.619 | 0.917 | 1.000 | 0.879 |
| supervised_baseline | nearest | exact_pair | 35.000 | 0.973 | 0.077 | 0.013 | 0.730 | 1.000 | 1.000 | 0.973 |
| supervised_baseline | nearest | hashed_pair | 35.000 | 0.901 | 0.124 | 0.021 | 0.635 | 1.000 | 1.000 | 0.901 |
| terminal_rl | hopfield | exact_pair | 35.000 | 0.385 | 0.306 | 0.052 | 0.000 | 0.250 | 1.000 | 0.385 |
| terminal_rl | hopfield | hashed_pair | 35.000 | 0.313 | 0.270 | 0.046 | 0.000 | 0.250 | 1.000 | 0.313 |
| terminal_rl | nearest | exact_pair | 35.000 | 0.394 | 0.322 | 0.054 | 0.000 | 0.333 | 1.000 | 0.394 |
| terminal_rl | nearest | hashed_pair | 35.000 | 0.400 | 0.316 | 0.053 | 0.000 | 0.333 | 1.000 | 0.400 |

### Seen transition accuracy by condition

| condition | cleanup | feature_type | count | seen_transition_accuracy_mean | seen_transition_accuracy_std | seen_transition_accuracy_sem | seen_transition_accuracy_min | seen_transition_accuracy_median | seen_transition_accuracy_max | seen_transition_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_terminal_rl | hopfield | exact_pair | 70.000 | 0.432 | 0.296 | 0.035 | 0.000 | 0.417 | 1.000 | 0.432 |
| sparse_terminal_rl | nearest | exact_pair | 70.000 | 0.432 | 0.295 | 0.035 | 0.000 | 0.375 | 1.000 | 0.432 |
| state_shaped_rl | hopfield | exact_pair | 35.000 | 0.819 | 0.251 | 0.042 | 0.021 | 0.917 | 1.000 | 0.819 |
| state_shaped_rl | hopfield | hashed_pair | 35.000 | 0.812 | 0.241 | 0.041 | 0.102 | 0.917 | 1.000 | 0.812 |
| state_shaped_rl | nearest | exact_pair | 35.000 | 0.874 | 0.124 | 0.021 | 0.667 | 0.875 | 1.000 | 0.874 |
| state_shaped_rl | nearest | hashed_pair | 35.000 | 0.796 | 0.183 | 0.031 | 0.477 | 0.833 | 1.000 | 0.796 |
| supervised_baseline | hopfield | exact_pair | 35.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| supervised_baseline | hopfield | hashed_pair | 35.000 | 0.888 | 0.119 | 0.020 | 0.625 | 0.917 | 1.000 | 0.888 |
| supervised_baseline | nearest | exact_pair | 35.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| supervised_baseline | nearest | hashed_pair | 35.000 | 0.909 | 0.109 | 0.018 | 0.705 | 1.000 | 1.000 | 0.909 |
| terminal_rl | hopfield | exact_pair | 35.000 | 0.386 | 0.305 | 0.052 | 0.000 | 0.250 | 1.000 | 0.386 |
| terminal_rl | hopfield | hashed_pair | 35.000 | 0.314 | 0.269 | 0.045 | 0.000 | 0.250 | 1.000 | 0.314 |
| terminal_rl | nearest | exact_pair | 35.000 | 0.394 | 0.323 | 0.055 | 0.000 | 0.333 | 1.000 | 0.394 |
| terminal_rl | nearest | hashed_pair | 35.000 | 0.401 | 0.313 | 0.053 | 0.000 | 0.333 | 1.000 | 0.401 |

### Unseen transition accuracy by condition

| condition | cleanup | feature_type | count | unseen_transition_accuracy_mean | unseen_transition_accuracy_std | unseen_transition_accuracy_sem | unseen_transition_accuracy_min | unseen_transition_accuracy_median | unseen_transition_accuracy_max | unseen_transition_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_terminal_rl | hopfield | exact_pair | 10.000 | 0.135 | 0.131 | 0.041 | 0.000 | 0.107 | 0.368 | 0.135 |
| sparse_terminal_rl | nearest | exact_pair | 10.000 | 0.100 | 0.061 | 0.019 | 0.000 | 0.088 | 0.200 | 0.100 |
| state_shaped_rl | hopfield | exact_pair | 5.000 | 0.140 | 0.094 | 0.042 | 0.000 | 0.143 | 0.235 | 0.140 |
| state_shaped_rl | hopfield | hashed_pair | 5.000 | 0.295 | 0.242 | 0.108 | 0.000 | 0.400 | 0.529 | 0.295 |
| state_shaped_rl | nearest | exact_pair | 5.000 | 0.322 | 0.155 | 0.069 | 0.143 | 0.263 | 0.500 | 0.322 |
| state_shaped_rl | nearest | hashed_pair | 5.000 | 0.573 | 0.195 | 0.087 | 0.333 | 0.571 | 0.857 | 0.573 |
| supervised_baseline | hopfield | exact_pair | 5.000 | 0.200 | 0.447 | 0.200 | 0.000 | 0.000 | 1.000 | 0.200 |
| supervised_baseline | hopfield | hashed_pair | 5.000 | 0.472 | 0.032 | 0.014 | 0.421 | 0.471 | 0.500 | 0.472 |
| supervised_baseline | nearest | exact_pair | 5.000 | 0.200 | 0.447 | 0.200 | 0.000 | 0.000 | 1.000 | 0.200 |
| supervised_baseline | nearest | hashed_pair | 5.000 | 0.508 | 0.214 | 0.096 | 0.286 | 0.526 | 0.824 | 0.508 |
| terminal_rl | hopfield | exact_pair | 5.000 | 0.078 | 0.081 | 0.036 | 0.000 | 0.071 | 0.176 | 0.078 |
| terminal_rl | hopfield | hashed_pair | 5.000 | 0.014 | 0.032 | 0.014 | 0.000 | 0.000 | 0.071 | 0.014 |
| terminal_rl | nearest | exact_pair | 5.000 | 0.099 | 0.090 | 0.040 | 0.000 | 0.071 | 0.235 | 0.099 |
| terminal_rl | nearest | hashed_pair | 5.000 | 0.120 | 0.268 | 0.120 | 0.000 | 0.000 | 0.600 | 0.120 |

### Sparse terminal RL by write fraction

| cleanup | write_fraction | count | final_test_accuracy_mean | final_test_accuracy_std | final_test_accuracy_sem | final_test_accuracy_min | final_test_accuracy_median | final_test_accuracy_max | final_test_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hopfield | 0.300 | 35.000 | 0.616 | 0.194 | 0.033 | 0.397 | 0.500 | 1.000 | 0.616 |
| hopfield | 0.500 | 35.000 | 0.631 | 0.208 | 0.035 | 0.364 | 0.500 | 1.000 | 0.631 |
| nearest | 0.300 | 35.000 | 0.608 | 0.185 | 0.031 | 0.383 | 0.506 | 1.000 | 0.608 |
| nearest | 0.500 | 35.000 | 0.634 | 0.201 | 0.034 | 0.387 | 0.519 | 1.000 | 0.634 |

### Learning curve test accuracy

| condition | cleanup | feature_type | episode | count | test_accuracy_mean | test_accuracy_std | test_accuracy_sem | test_accuracy_min | test_accuracy_median | test_accuracy_max | test_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_terminal_rl | hopfield | exact_pair | 1.000 | 70.000 | 0.497 | 0.068 | 0.008 | 0.128 | 0.500 | 0.670 | 0.497 |
| sparse_terminal_rl | hopfield | exact_pair | 250.000 | 70.000 | 0.502 | 0.104 | 0.012 | 0.117 | 0.500 | 0.879 | 0.502 |
| sparse_terminal_rl | hopfield | exact_pair | 500.000 | 70.000 | 0.487 | 0.125 | 0.015 | 0.000 | 0.500 | 0.821 | 0.487 |
| sparse_terminal_rl | hopfield | exact_pair | 750.000 | 70.000 | 0.527 | 0.120 | 0.014 | 0.147 | 0.500 | 0.897 | 0.527 |
| sparse_terminal_rl | hopfield | exact_pair | 1000.000 | 70.000 | 0.572 | 0.168 | 0.020 | 0.178 | 0.500 | 1.000 | 0.572 |
| sparse_terminal_rl | hopfield | exact_pair | 1250.000 | 70.000 | 0.556 | 0.164 | 0.020 | 0.000 | 0.500 | 1.000 | 0.556 |
| sparse_terminal_rl | hopfield | exact_pair | 1500.000 | 70.000 | 0.564 | 0.193 | 0.023 | 0.140 | 0.500 | 1.000 | 0.564 |
| sparse_terminal_rl | hopfield | exact_pair | 1750.000 | 70.000 | 0.560 | 0.182 | 0.022 | 0.121 | 0.500 | 1.000 | 0.560 |
| sparse_terminal_rl | hopfield | exact_pair | 2000.000 | 70.000 | 0.563 | 0.171 | 0.020 | 0.125 | 0.500 | 1.000 | 0.563 |
| sparse_terminal_rl | hopfield | exact_pair | 2250.000 | 70.000 | 0.556 | 0.179 | 0.021 | 0.128 | 0.500 | 1.000 | 0.556 |
| sparse_terminal_rl | hopfield | exact_pair | 2500.000 | 70.000 | 0.536 | 0.190 | 0.023 | 0.125 | 0.500 | 1.000 | 0.536 |
| sparse_terminal_rl | hopfield | exact_pair | 2750.000 | 70.000 | 0.564 | 0.167 | 0.020 | 0.311 | 0.500 | 1.000 | 0.564 |
| sparse_terminal_rl | hopfield | exact_pair | 3000.000 | 70.000 | 0.585 | 0.207 | 0.025 | 0.134 | 0.500 | 1.000 | 0.585 |
| sparse_terminal_rl | hopfield | exact_pair | 3250.000 | 70.000 | 0.572 | 0.197 | 0.024 | 0.142 | 0.500 | 1.000 | 0.572 |
| sparse_terminal_rl | hopfield | exact_pair | 3500.000 | 70.000 | 0.569 | 0.192 | 0.023 | 0.125 | 0.500 | 1.000 | 0.569 |
| sparse_terminal_rl | hopfield | exact_pair | 3750.000 | 70.000 | 0.595 | 0.185 | 0.022 | 0.117 | 0.500 | 1.000 | 0.595 |
| sparse_terminal_rl | hopfield | exact_pair | 4000.000 | 70.000 | 0.613 | 0.205 | 0.025 | 0.266 | 0.507 | 1.000 | 0.613 |
| sparse_terminal_rl | hopfield | exact_pair | 4250.000 | 70.000 | 0.590 | 0.196 | 0.023 | 0.125 | 0.500 | 1.000 | 0.590 |
| sparse_terminal_rl | hopfield | exact_pair | 4500.000 | 70.000 | 0.577 | 0.187 | 0.022 | 0.128 | 0.500 | 1.000 | 0.577 |
| sparse_terminal_rl | hopfield | exact_pair | 4750.000 | 70.000 | 0.606 | 0.201 | 0.024 | 0.278 | 0.500 | 1.000 | 0.606 |
| sparse_terminal_rl | hopfield | exact_pair | 5000.000 | 70.000 | 0.624 | 0.200 | 0.024 | 0.364 | 0.500 | 1.000 | 0.624 |
| sparse_terminal_rl | nearest | exact_pair | 1.000 | 70.000 | 0.489 | 0.067 | 0.008 | 0.128 | 0.500 | 0.690 | 0.489 |
| sparse_terminal_rl | nearest | exact_pair | 250.000 | 70.000 | 0.513 | 0.110 | 0.013 | 0.125 | 0.500 | 0.821 | 0.513 |
| sparse_terminal_rl | nearest | exact_pair | 500.000 | 70.000 | 0.515 | 0.099 | 0.012 | 0.000 | 0.500 | 0.786 | 0.515 |
| sparse_terminal_rl | nearest | exact_pair | 750.000 | 70.000 | 0.523 | 0.104 | 0.012 | 0.350 | 0.500 | 1.000 | 0.523 |
| sparse_terminal_rl | nearest | exact_pair | 1000.000 | 70.000 | 0.537 | 0.164 | 0.020 | 0.000 | 0.500 | 1.000 | 0.537 |
| sparse_terminal_rl | nearest | exact_pair | 1250.000 | 70.000 | 0.526 | 0.140 | 0.017 | 0.244 | 0.500 | 1.000 | 0.526 |
| sparse_terminal_rl | nearest | exact_pair | 1500.000 | 70.000 | 0.549 | 0.133 | 0.016 | 0.383 | 0.500 | 1.000 | 0.549 |
| sparse_terminal_rl | nearest | exact_pair | 1750.000 | 70.000 | 0.550 | 0.132 | 0.016 | 0.358 | 0.500 | 1.000 | 0.550 |
| sparse_terminal_rl | nearest | exact_pair | 2000.000 | 70.000 | 0.531 | 0.148 | 0.018 | 0.213 | 0.500 | 1.000 | 0.531 |
| sparse_terminal_rl | nearest | exact_pair | 2250.000 | 70.000 | 0.541 | 0.185 | 0.022 | 0.129 | 0.500 | 1.000 | 0.541 |
| sparse_terminal_rl | nearest | exact_pair | 2500.000 | 70.000 | 0.565 | 0.176 | 0.021 | 0.140 | 0.500 | 1.000 | 0.565 |
| sparse_terminal_rl | nearest | exact_pair | 2750.000 | 70.000 | 0.549 | 0.185 | 0.022 | 0.117 | 0.500 | 1.000 | 0.549 |
| sparse_terminal_rl | nearest | exact_pair | 3000.000 | 70.000 | 0.579 | 0.185 | 0.022 | 0.273 | 0.500 | 1.000 | 0.579 |
| sparse_terminal_rl | nearest | exact_pair | 3250.000 | 70.000 | 0.547 | 0.193 | 0.023 | 0.125 | 0.500 | 1.000 | 0.547 |
| sparse_terminal_rl | nearest | exact_pair | 3500.000 | 70.000 | 0.586 | 0.170 | 0.020 | 0.341 | 0.500 | 1.000 | 0.586 |
| sparse_terminal_rl | nearest | exact_pair | 3750.000 | 70.000 | 0.570 | 0.193 | 0.023 | 0.213 | 0.500 | 1.000 | 0.570 |
| sparse_terminal_rl | nearest | exact_pair | 4000.000 | 70.000 | 0.609 | 0.184 | 0.022 | 0.267 | 0.528 | 1.000 | 0.609 |
| sparse_terminal_rl | nearest | exact_pair | 4250.000 | 70.000 | 0.588 | 0.183 | 0.022 | 0.125 | 0.514 | 1.000 | 0.588 |
| sparse_terminal_rl | nearest | exact_pair | 4500.000 | 70.000 | 0.597 | 0.186 | 0.022 | 0.125 | 0.500 | 1.000 | 0.597 |
| sparse_terminal_rl | nearest | exact_pair | 4750.000 | 70.000 | 0.612 | 0.185 | 0.022 | 0.335 | 0.519 | 1.000 | 0.612 |
| sparse_terminal_rl | nearest | exact_pair | 5000.000 | 70.000 | 0.621 | 0.192 | 0.023 | 0.383 | 0.516 | 1.000 | 0.621 |
| state_shaped_rl | hopfield | exact_pair | 1.000 | 35.000 | 0.491 | 0.056 | 0.009 | 0.298 | 0.500 | 0.639 | 0.491 |
| state_shaped_rl | hopfield | exact_pair | 250.000 | 35.000 | 0.682 | 0.245 | 0.041 | 0.000 | 0.613 | 1.000 | 0.682 |
| state_shaped_rl | hopfield | exact_pair | 500.000 | 35.000 | 0.749 | 0.236 | 0.040 | 0.198 | 0.743 | 1.000 | 0.749 |
| state_shaped_rl | hopfield | exact_pair | 750.000 | 35.000 | 0.769 | 0.229 | 0.039 | 0.200 | 0.801 | 1.000 | 0.769 |
| state_shaped_rl | hopfield | exact_pair | 1000.000 | 35.000 | 0.795 | 0.228 | 0.039 | 0.374 | 0.872 | 1.000 | 0.795 |
| state_shaped_rl | hopfield | exact_pair | 1250.000 | 35.000 | 0.779 | 0.239 | 0.040 | 0.412 | 0.885 | 1.000 | 0.779 |
| state_shaped_rl | hopfield | exact_pair | 1500.000 | 35.000 | 0.814 | 0.226 | 0.038 | 0.404 | 0.988 | 1.000 | 0.814 |
| state_shaped_rl | hopfield | exact_pair | 1750.000 | 35.000 | 0.790 | 0.224 | 0.038 | 0.500 | 0.875 | 1.000 | 0.790 |
| state_shaped_rl | hopfield | exact_pair | 2000.000 | 35.000 | 0.807 | 0.218 | 0.037 | 0.444 | 0.989 | 1.000 | 0.807 |
| state_shaped_rl | hopfield | exact_pair | 2250.000 | 35.000 | 0.822 | 0.213 | 0.036 | 0.412 | 0.988 | 1.000 | 0.822 |
| state_shaped_rl | hopfield | exact_pair | 2500.000 | 35.000 | 0.824 | 0.206 | 0.035 | 0.451 | 0.924 | 1.000 | 0.824 |
| state_shaped_rl | hopfield | exact_pair | 2750.000 | 35.000 | 0.802 | 0.239 | 0.040 | 0.373 | 0.988 | 1.000 | 0.802 |
| state_shaped_rl | hopfield | exact_pair | 3000.000 | 35.000 | 0.817 | 0.217 | 0.037 | 0.427 | 0.908 | 1.000 | 0.817 |
| state_shaped_rl | hopfield | exact_pair | 3250.000 | 35.000 | 0.798 | 0.239 | 0.040 | 0.290 | 0.988 | 1.000 | 0.798 |
| state_shaped_rl | hopfield | exact_pair | 3500.000 | 35.000 | 0.801 | 0.234 | 0.040 | 0.415 | 0.988 | 1.000 | 0.801 |
| state_shaped_rl | hopfield | exact_pair | 3750.000 | 35.000 | 0.812 | 0.220 | 0.037 | 0.500 | 0.988 | 1.000 | 0.812 |
| state_shaped_rl | hopfield | exact_pair | 4000.000 | 35.000 | 0.809 | 0.228 | 0.038 | 0.332 | 0.926 | 1.000 | 0.809 |
| state_shaped_rl | hopfield | exact_pair | 4250.000 | 35.000 | 0.821 | 0.223 | 0.038 | 0.373 | 0.988 | 1.000 | 0.821 |

_Showing 60 of 294 rows._
