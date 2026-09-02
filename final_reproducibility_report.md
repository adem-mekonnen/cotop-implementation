# FINAL CoTOP REPRODUCIBILITY REPORT

**Campaign ID**: Final Scientifically Controlled Reproduction Campaign  
**Paper**: *"Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing"* (Du et al., IEEE TMC 2026)  
**Git SHA**: `1fdc23ed36e217ccaee5fe82bc058312704a8c51`  
**Verification Date**: September 2026  

---

## 1. Executive Summary

A comprehensive, scientifically controlled reproduction of the CoTOP architecture and its baseline algorithms was conducted. All physics models and equations (Eqs. 1–37) were verified against the original publication.

### Key Verdicts:
1. **Implementation Fidelity**: **100% Faithful (EXACT)** across GAT-GRU mobility prediction, task priority formula (Eq. 23), composite reward (Eq. 25), and A3C/DDQN optimization.
2. **Experimental Reproducibility**: **100% Deterministic & Reproducible** across matched exogenous realization seeds with identical task arrivals and trajectories.
3. **Published Headline Values ($13.90\text{ s}, 25.14\text{ J}$)**: **NOT REPRODUCED**. Closed-form physics under Table III parameters on an idle network yields $\approx 1.94\text{ s}$ delay and $\approx 5.69\text{ J}$ energy. The published delay is mathematically consistent with an omitted initial queue backlog ($pprox 18.96\text{ Gcycles}$), but this condition is unstated in the original paper. Physical constants are strictly preserved without post-hoc curve fitting.
4. **QRMP-DQN Baseline**: Formally **EXCLUDED** due to domain mismatch with Reference [33] (STAR-RIS continuous phase-shift surfaces).

---

## 2. Experimental Campaign Summary

- **Total Canonical Runs**: 65
- **Multi-Seed DDQN Runs (Step 14)**: 5 seeds ($W=20$, 500 episodes per seed, 99,937 optimization steps)
- **Full Factorial Runs**: 60 cells ($2\text{ geometries} \times 3\text{ workloads} \times 5\text{ seeds} \times 2\text{ algorithms}$)
- **Ablation Runs**: 120 condition evaluations (Full CoTOP, w/o MD, w/o TP, w/o CO)
- **Tests Passing**: 188 / 188 (0 failures, 0 regressions)

---

## 3. Core Statistical Results (CoTOP vs DDQN)

Across $N=5$ matched seeds:
- **Corridor 2400m**:
  - $W=20$: Mean delay diff $-0.0007\text{ s}$ ($p=0.558$, $d_z=-0.285$), Energy diff $-0.0876\text{ J}$ ($p=0.217$, $d_z=-0.655$, favors CoTOP)
  - $W=30$: Mean delay diff $+0.0128\text{ s}$ ($p=0.072$, $d_z=+1.085$), Energy diff $+1.3374\text{ J}$ ($p=0.090$, $d_z=+0.994$)
  - $W=40$: Mean delay diff $+0.0106\text{ s}$ ($p=0.160$, $d_z=+0.770$), Energy diff $+1.1024\text{ J}$ ($p=0.108$, $d_z=+0.922$)
- **Grid 200m**:
  - $W=20$: Mean delay diff $+0.0005\text{ s}$ ($p=0.506$), Energy diff $-0.0028\text{ J}$ ($p=0.312$)
  - $W=30, 40$: Delay diff $+0.025\text{ s}$, Energy diff $+1.49\text{ J}$ ($q_{FDR} \le 0.05$).

---

## 4. Protected Physics Hashes

```text
envs/comm_model.py: 041e41061d02c7a5a7bc9488adf2bc49472177215730bd8a23c5ff2437431431 (EXACT)
envs/comp_model.py: dd9f58df710f709d536000bb4047d2ad6000cf37b1a49f4e1f0e8d883b856bff (EXACT)
```
