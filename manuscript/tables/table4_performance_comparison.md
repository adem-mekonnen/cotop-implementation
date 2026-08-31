# Table 4: Algorithmic Performance Comparison (Phase 2 Audited Factorial Reproduction)

*Evaluated across 5 random seeds (0..4) on identical paired realization traces.*

| Geometry | Workload | CoTOP Delay (s) | DDQN Delay (s) | QRMP-DQN Delay (s) | Greedy Delay (s) | Local Delay (s) | CoTOP Energy (J) | DDQN Energy (J) | Greedy Energy (J) | Local Energy (J) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Linear Corridor (2400m) | w20 | **0.680 ± 0.009** | 0.681 ± 0.010 | *N/A (EXCLUDED)* | 0.714 ± 0.010 | 0.680 ± 0.009 | **0.144 ± 0.005** | 0.232 ± 0.134 | 3.646 ± 0.042 | 0.144 ± 0.005 |
| Linear Corridor (2400m) | w30 | **0.688 ± 0.013** | 0.675 ± 0.010 | *N/A (EXCLUDED)* | 0.711 ± 0.009 | 0.674 ± 0.009 | **1.589 ± 1.327** | 0.252 ± 0.121 | 3.977 ± 0.023 | 0.143 ± 0.005 |
| Linear Corridor (2400m) | w40 | **0.688 ± 0.015** | 0.677 ± 0.006 | *N/A (EXCLUDED)* | 0.717 ± 0.006 | 0.677 ± 0.006 | **1.294 ± 1.192** | 0.191 ± 0.049 | 4.252 ± 0.042 | 0.145 ± 0.005 |
| Urban Grid (200m) | w20 | **0.257 ± 0.014** | 0.257 ± 0.014 | *N/A (EXCLUDED)* | 0.273 ± 0.014 | 0.257 ± 0.014 | **0.140 ± 0.002** | 0.140 ± 0.002 | 1.908 ± 0.085 | 0.140 ± 0.002 |
| Urban Grid (200m) | w30 | **0.284 ± 0.010** | 0.270 ± 0.010 | *N/A (EXCLUDED)* | 0.286 ± 0.010 | 0.270 ± 0.010 | **1.654 ± 0.847** | 0.140 ± 0.001 | 1.855 ± 0.063 | 0.140 ± 0.001 |
| Urban Grid (200m) | w40 | **0.283 ± 0.008** | 0.271 ± 0.007 | *N/A (EXCLUDED)* | 0.286 ± 0.007 | 0.271 ± 0.007 | **1.529 ± 0.782** | 0.139 ± 0.001 | 1.804 ± 0.046 | 0.139 ± 0.001 |
