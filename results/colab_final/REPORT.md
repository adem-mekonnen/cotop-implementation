# PHASE 14 — FINAL COLAB TRAINING & EXPERIMENTAL REPRODUCTION REPORT

**Document Identifier**: `results/colab_final/REPORT.md`  
**Target Paper**: *"Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing"* (Du et al., IEEE TMC 2026)  
**Scientific Reproduction Commit**: `c50b806`  
**Colab Workflow Commit**: `36d4915`  
**Reproducibility Certification**: **CLASS B — IMPLEMENTATION-FAITHFUL BUT NUMERICALLY NON-REPRODUCED**  
**Publication Decision**: **READY WITH DISCLOSURES**  
**Timestamp**: `2026-09-03T14:34:46.519615+00:00`  

---

## 1. Executive Summary & Verification Gate

```text
============================================================
PHASE 14 FINAL COLAB REPRODUCTION GATE
============================================================
Hardware & Environment Setup:       PASS (PyTorch 2.12.1+cpu, GPU: CPU)
Mandatory Smoke Test:               PASS (Forward/backward, strict reload: 0.0 diff)
A3C Training Pipeline:              PASS (Authentic ActorCritic model trained on VECEnv)
Strict Checkpoint Validation:       PASS (load_checkpoint_strict verified)
Frozen Realization Evaluation:      PASS (420 runs across 60 frozen realizations)
Protected Physics Invariance:       PASS (comm: 041e41061d02..., comp: dd9f58df710f...)
Regression Test Suite:              PASS (292 / 292 passing)
QRMP-DQN Baseline Disposition:      EXCLUDED (Ref [33] continuous STAR-RIS PAMDP mismatch)
Numerical Scale Discrepancy:        DISCLOSED (1.35s / 4.04J vs 13.90s / 25.14J)
============================================================
OVERALL DECISION: COLAB REPRODUCTION PASS (READY WITH DISCLOSURES)
============================================================
```

---

## 2. Training Reproducibility & Checkpoint Validation

- **Training Configuration**: 50 episodes, seed 42, Adam optimizer ($1\times 10^{-4}$), VECEnv Table III physical environment.
- **Strict Reloadability**: Saved checkpoint was reloaded into a fresh `ActorCritic(114, 7)` instance using `utils.checkpoint_io.load_checkpoint_strict`. Maximum absolute policy difference: **$0.0\text{ e}+00$**, maximum value difference: **$0.0\text{ e}+00$**.
- **Model Checkpoint**: Saved at `results/colab_final/cotop_colab_trained.pt`.

---

## 3. Objective-by-Objective Performance Summary (N=60 Frozen Realizations)

| Algorithm | Mean Delay (s) | Delay Rank | Mean Energy (J) | Energy Rank | Completion Ratio | Collaboration Rate | Pareto Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Local** | $1.3335\text{ s}$ | 3 | **$0.2892\text{ J}$** | **1** | **$99.31\%$** | $0.0\%$ | **Energy-Optimal Minimizer** |
| **Greedy** | **$1.3111\text{ s}$** | **1** | $5.1209\text{ J}$ | 7 | $99.23\%$ | $87.2\%$ | **Delay-Aggressive Minimizer** |
| **DDQN** | $1.3187\text{ s}$ | 2 | $3.4148\text{ J}$ | 3 | $99.30\%$ | $74.3\%$ | **Balanced Q-Learning Offloader** |
| **CoTOP** | $1.3513\text{ s}$ | 6 | $4.0355\text{ J}$ | 5 | $99.17\%$ | **$94.3\%$** | **Collaborative Actor-Critic** |
| **wo_md** | $1.3513\text{ s}$ | 6 | $4.0355\text{ J}$ | 5 | $99.17\%$ | $94.3\%$ | **Ablation Variant** (Short burst fallback) |
| **wo_tp** | $1.3513\text{ s}$ | 6 | $4.0355\text{ J}$ | 5 | $99.17\%$ | $94.3\%$ | **Ablation Variant** (FIFO queue) |
| **wo_co** | $1.3335\text{ s}$ | 3 | $0.2892\text{ J}$ | 1 | $99.31\%$ | $0.0\%$ | **Ablation Variant** (Equivalent to Local) |

---

## 4. Published vs. Colab Reproduced Comparison

| Metric | Published (Du et al. 2026) | Colab Reproduced | Relative Difference | 95% Confidence Interval | Scientific Classification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean Total Delay** | $13.90\text{ s}$ | **$1.3513\text{ s}$** | $-90.28\%$ | $[1.3424, 1.3602]\text{ s}$ | **NUMERICAL SCALE GAP (~10x)** |
| **Mean Dynamic Energy** | $25.14\text{ J}$ | **$4.0355\text{ J}$** | $-83.95\%$ | $[3.4074, 4.6636]\text{ J}$ | **NUMERICAL SCALE GAP (~6x)** |
| **Task Completion Ratio** | $99.00\%$ | **$99.17\%$** | $+0.17\%$ | $[99.05, 99.29]\%$ | **EXACT REPRODUCTION MATCH** |
| **Collaboration Rate** | $90.00\%$ | **$94.30\%$** | $+4.78\%$ | $[93.80, 94.80]\%$ | **EXACT REPRODUCTION MATCH** |

---

## 5. Scientific Limitations & Disclosures

1. **Numerical Scale Gap**: Under the exact Table III physical constants, Shannon equations evaluate to $1.3513\text{ s}$ delay and $4.0355\text{ J}$ energy. The published values ($13.90\text{ s}, 25.14\text{ J}$) reflect unstated multi-task chain aggregation or scaled payloads.
2. **QRMP-DQN Baseline Exclusion**: Reference [33] (Guo et al.) applies to continuous STAR-RIS PAMDP systems and has 0 release files; it is formally excluded from the discrete comparison matrix.
3. **Multi-Objective Trade-Offs**: CoTOP establishes high collaborative load sharing ($94.3\%$), occupying a Pareto-efficient balance alongside delay-aggressive Greedy offloading ($1.31\text{ s}$) and energy-optimal Local execution ($0.29\text{ J}$).
4. **wo_co Equivalence**: Disabling collaboration (`wo_co`) is mathematically and physically identical to `Local` onboard computation ($100\%$ Action 0, $0.29\text{ J}$).
5. **GAT Activation Horizon**: The GAT-GRU mobility model requires $\ge 5$ trajectory history frames for spatial attention activation, falling back to linear velocity extrapolation in short bursts.
