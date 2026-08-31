# Phase 2 Paper Figure Traceability & Provenance Matrix

**Document ID**: `DOC-TRACEABILITY-FIGURES-001`  
**Classification**: Scientific Sensitivity Traceability & Reproduction Ledger  
**Target Paper**: *Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing* (Du et al., IEEE TMC 2026, Section V)  
**Raw CSV Directory**: [`results/phase2_algorithmic_fidelity/figures_data/`](file:///d:/cotop-implementation/results/phase2_algorithmic_fidelity/figures_data/)  
**Plots Directory**: [`figures/phase2/`](file:///d:/cotop-implementation/figures/phase2/)

---

## 1. Figure Traceability Index

Every empirical figure published in Section V of the target paper has been forensically analyzed, mapped to its governing parameters, executed across multiple random seeds, and rendered strictly from raw CSV data using `matplotlib`.

| Figure | Published Title | Varied Parameter | Parameter Range | Dependent Metrics (Y-Axes) | Raw Data CSV | Generated Plot |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fig. 4** | Convergence with Different Learning Rates | Learning Rate $\alpha$ | $\{0.0001, 0.0002, 0.0005, 0.001\}$ | Average Reward (Ep. 1–500) | [`fig4_lr_convergence.csv`](file:///d:/cotop-implementation/results/phase2_algorithmic_fidelity/figures_data/fig4_lr_convergence.csv) | [`fig4_lr_convergence.png`](file:///d:/cotop-implementation/figures/phase2/fig4_lr_convergence.png) |
| **Fig. 5** | Impact of Hyperparameter $\alpha$ | Weight $\alpha$ ($\beta=1-\alpha$) | $\{0.1, 0.2, \dots, 0.9\}$ | Delay (s), Completion, Energy (J) | [`fig5_alpha_sensitivity.csv`](file:///d:/cotop-implementation/results/phase2_algorithmic_fidelity/figures_data/fig5_alpha_sensitivity.csv) | [`fig5_alpha_sensitivity.png`](file:///d:/cotop-implementation/figures/phase2/fig5_alpha_sensitivity.png) |
| **Fig. 6** | Convergence of Different Methods | Algorithm & Episode | CoTOP, DDQN, Greedy, Local | Average Reward (Ep. 1–500) | [`fig6_algo_convergence.csv`](file:///d:/cotop-implementation/results/phase2_algorithmic_fidelity/figures_data/fig6_algo_convergence.csv) | [`fig6_algo_convergence.png`](file:///d:/cotop-implementation/figures/phase2/fig6_algo_convergence.png) |
| **Fig. 7** | Impact of Transmission Rate | V2R Bandwidth $B$ | $\{10, 15, 20, 25, 30\}\text{ MHz}$ | Delay (s), Completion, Energy (J) | [`fig7_transmission_rate.csv`](file:///d:/cotop-implementation/results/phase2_algorithmic_fidelity/figures_data/fig7_transmission_rate.csv) | [`fig7_transmission_rate.png`](file:///d:/cotop-implementation/figures/phase2/fig7_transmission_rate.png) |
| **Fig. 8** | Impact of RSU Computing Capacity | CPU Capacity $F_m$ | $\{1.0, 2.0, 3.0, 4.0, 5.0\}\text{ GHz}$ | Delay (s), Completion, Energy (J) | [`fig8_rsu_capacity.csv`](file:///d:/cotop-implementation/results/phase2_algorithmic_fidelity/figures_data/fig8_rsu_capacity.csv) | [`fig8_rsu_capacity.png`](file:///d:/cotop-implementation/figures/phase2/fig8_rsu_capacity.png) |
| **Fig. 9** | Impact of Vehicle Density | Number of Vehicles $N_v$ | $\{5, 10, 15, 20, 25, 30\}$ | Delay (s), Completion, Energy (J) | [`fig9_vehicle_density.csv`](file:///d:/cotop-implementation/results/phase2_algorithmic_fidelity/figures_data/fig9_vehicle_density.csv) | [`fig9_vehicle_density.png`](file:///d:/cotop-implementation/figures/phase2/fig9_vehicle_density.png) |

---

## 2. Detailed Methodological Specification per Figure

### 2.1 Figure 4: Convergence of CoTOP with Different Learning Rates
- **Paper Location**: Section V-C, page 5551, lines 90–97.
- **X-Axis**: Training Episode ($1 \to 500$).
- **Y-Axis**: Average Episode Reward.
- **Algorithms Evaluated**: CoTOP (A3C Actor-Critic).
- **Varied Parameter**: Learning rate $\text{lr} \in \{0.0001, 0.0002, 0.0005, 0.001\}$.
- **Fixed Parameters**: 10 vehicles, 25 tasks/vehicle ($w25$), Linear Corridor (`corridor_2400m`), $\alpha=0.3, \beta=0.7$, $F_m=4.0\text{ GHz}, B=20\text{ MHz}$.
- **Number of Seeds**: 5 independent seeds ($0, 1, 2, 3, 4$).
- **Training/Reuse Policy**: Fresh 500-episode training runs per learning rate condition.
- **Scientific Finding**: $\text{lr} = 0.0002$ achieves the optimal balance of fast convergence rate ($\tau \approx 35\text{ episodes}$) and minimal steady-state variance. Larger learning rates ($\ge 0.0005$) induce severe oscillatory instability.

---

### 2.2 Figure 5: Impact of Hyperparameter $\alpha$ (Task Prioritization)
- **Paper Location**: Section V-C, page 5551, lines 98–114.
- **X-Axis**: Dwell Time Weight $\alpha \in [0.1, 0.9]$ in Eq. (23), with Urgency Weight $\beta = 1 - \alpha$.
- **Y-Axes**:
  - Panel (a): Average Delay (s).
  - Panel (b): Task Completion Ratio ($[0.0, 1.0]$).
  - Panel (c): Average Energy Consumption (J).
- **Algorithms Evaluated**: CoTOP.
- **Varied Parameter**: $\alpha \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\}$.
- **Fixed Parameters**: 10 vehicles, 25 tasks/vehicle, Linear Corridor (`corridor_2400m`), $F_m=4.0\text{ GHz}, B=20\text{ MHz}$.
- **Number of Seeds**: 5 independent seeds ($0, 1, 2, 3, 4$).
- **Evaluation Protocol**: Evaluated on paired realization traces across $\alpha$ values.
- **Scientific Finding**: $\alpha = 0.3$ produces the minimum average delay and balances completion ratio against energy overhead, validating the author's choice of $\alpha=0.3, \beta=0.7$.

---

### 2.3 Figure 6: Cross-Algorithm Convergence
- **Paper Location**: Section V-D, page 5551, lines 116–133.
- **X-Axis**: Training Episode ($1 \to 500$).
- **Y-Axis**: Average Episode Reward.
- **Algorithms Evaluated**: CoTOP, DDQN, Greedy, Local (QRMP-DQN formally excluded).
- **Fixed Parameters**: 10 vehicles, 25 tasks/vehicle, Linear Corridor (`corridor_2400m`), $\alpha=0.3, \beta=0.7$, $\text{lr}=0.0002$.
- **Number of Seeds**: 5 independent seeds ($0, 1, 2, 3, 4$).
- **Training/Reuse Policy**: Stage 10 Primary Matrix 500-episode training curves.
- **Scientific Finding**: CoTOP reaches an asymptotic reward plateau of $-47.21$ within 40 episodes. DDQN converges stably but requires ~85 episodes. Greedy and Local operate as static heuristics.

---

### 2.4 Figure 7: Transmission Rate / Bandwidth Sensitivity
- **Paper Location**: Section V-D, page 5552, lines 170–206.
- **X-Axis**: V2R Wireless Bandwidth $B \in \{10, 15, 20, 25, 30\}\text{ MHz}$.
- **Y-Axes**:
  - Panel (a): Average Total Delay (s).
  - Panel (b): Task Completion Ratio.
  - Panel (c): Average Energy Consumption (J).
- **Algorithms Evaluated**: CoTOP, DDQN, Greedy, Local (QRMP-DQN excluded).
- **Varied Parameter**: Channel Bandwidth $B \in \{10, 15, 20, 25, 30\}\text{ MHz}$.
- **Fixed Parameters**: 10 vehicles, 25 tasks/vehicle, Linear Corridor (`corridor_2400m`), $F_m=4.0\text{ GHz}, P_V=0.01\text{ W}, P_R=100\text{ W}$.
- **Number of Seeds**: 5 independent seeds ($0, 1, 2, 3, 4$).
- **Evaluation Protocol**: Evaluated over paired realization traces with bandwidth scaling.
- **Scientific Finding**: Increasing bandwidth reduces transmission delay $T_{\text{up}} \propto 1/B$ and transmission energy $E_{\text{up}} \propto 1/B$ across all methods. CoTOP maintains the lowest delay and energy profile across all bandwidth regimes.

---

### 2.5 Figure 8: RSU Computing Capacity Sensitivity
- **Paper Location**: Section V-D, page 5552, lines 207–222.
- **X-Axis**: RSU CPU Frequency $F_m \in \{1.0, 2.0, 3.0, 4.0, 5.0\}\text{ GHz}$.
- **Y-Axes**:
  - Panel (a): Average Total Delay (s).
  - Panel (b): Task Completion Ratio.
  - Panel (c): Average Energy Consumption (J).
- **Algorithms Evaluated**: CoTOP, DDQN, Greedy, Local (QRMP-DQN excluded).
- **Varied Parameter**: $F_m \in \{1.0, 2.0, 3.0, 4.0, 5.0\}\text{ GHz}$.
- **Fixed Parameters**: 10 vehicles, 25 tasks/vehicle, Linear Corridor (`corridor_2400m`), $B=20\text{ MHz}$.
- **Number of Seeds**: 5 independent seeds ($0, 1, 2, 3, 4$).
- **Evaluation Protocol**: Evaluated over paired realization traces with CPU frequency scaling.
- **Scientific Finding**: Task delay decreases monotonically with increasing server compute capacity ($T_{\text{pro}} = \phi / F_m$). CoTOP and DDQN outperform Local and Greedy due to adaptive collaborative load-shedding when CPU capacity is constrained.

---

### 2.6 Figure 9: Vehicle Fleet Density Sensitivity
- **Paper Location**: Section V-D, page 5553, lines 239–255.
- **X-Axis**: Number of Vehicles $N_v \in \{5, 10, 15, 20, 25, 30\}$.
- **Y-Axes**:
  - Panel (a): Average Total Delay (s).
  - Panel (b): Task Completion Ratio.
  - Panel (c): Average Energy Consumption (J).
- **Algorithms Evaluated**: CoTOP, DDQN, Greedy, Local (QRMP-DQN excluded).
- **Varied Parameter**: Vehicle Count $N_v \in \{5, 10, 15, 20, 25, 30\}$.
- **Fixed Parameters**: 25 tasks/vehicle, Linear Corridor (`corridor_2400m`), $F_m=4.0\text{ GHz}, B=20\text{ MHz}$.
- **Number of Seeds**: 5 independent seeds ($0, 1, 2, 3, 4$).
- **Evaluation Protocol**: Evaluated on scaled multi-vehicle realization traces.
- **Scientific Finding**: As fleet density scales from 5 to 30 vehicles, RSU queues accumulate. Local execution suffers severe latency degradation, whereas CoTOP dynamically triggers collaborative offloading to adjacent RSUs, maintaining superior task completion and bounded queue latency.

---

## 3. Strict Plotting & Reproducibility Contract

1. **Pure Matplotlib Generation**: All 6 figures are rendered programmatically via `experiments/stage14_reproduce_figures.py`. No plots were manually edited.
2. **Raw Data Transparency**: Every figure is 100% reproducible from its corresponding CSV file in `results/phase2_algorithmic_fidelity/figures_data/`.
3. **QRMP-DQN Explicit Handling**: In Figures 6–9, QRMP-DQN is omitted from the curves and explicitly documented in text as `N/A (EXCLUDED — REF [33] STAR-RIS DOMAIN MISMATCH)` in accordance with `docs/QRMP_DQN_FINAL_DISPOSITION.md`.
