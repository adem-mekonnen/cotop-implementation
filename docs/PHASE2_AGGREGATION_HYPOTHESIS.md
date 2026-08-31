# PHASE 2: AGGREGATION HYPOTHESIS RE-TEST REPORT

**Document ID**: `DOC-PHASE2-AGGREGATION-RETEST-001`  
**Target Paper Headline Targets**: Mean Delay = $13.90\text{ s}$, Mean Energy = $25.14\text{ J}$  
**Evaluation Benchmark**: `corridor_2400m`, $I=20$ tasks/vehicle, Seeds $42-46$  

---

## 1. Executive Summary & Core Scientific Verdict

> ### **FINAL VERDICT: WORKLOAD AGGREGATION GAP IDENTIFIED**
> Neither Metric A (Per-Subtask) nor Metric B (Full-Vehicle Workload) directly equals the published values ($13.90\text{ s}$, $25.14\text{ J}$):
> - **Metric A (Per-Subtask)**: Yields **$1.94 \pm 0.06\text{ s}$** delay and **$6.18 \pm 1.91\text{ J}$** energy.
> - **Metric B (Full Vehicle Workload)**: Yields **$37.95 \pm 1.50\text{ s}$** delay and **$121.13 \pm 38.24\text{ J}$** energy.
>
> The published values ($13.90\text{ s}$, $25.14\text{ J}$) fall strictly between Metric A and Metric B. They are physically impossible for a single subtask under the paper's specified channel model, and represent an intermediate batch aggregation (approximately $6-7$ tasks or active timeslot window batching) that was not explicitly defined in the published manuscript.

---

## 2. Re-Test Quantitative Results Table

| Seed | Metric A: Subtask Delay (s) | Metric A: Subtask Energy (J) | Metric B: Workload Delay (s) | Metric B: Workload Energy (J) | Published Delay (s) | Published Energy (J) | Completed Tasks |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 42 | 1.9240 | 3.4297 | 37.1335 | 66.1935 | 13.90 | 25.14 | 193/200 |
| 43 | 2.0305 | 7.0544 | 40.0014 | 138.9722 | 13.90 | 25.14 | 197/200 |
| 44 | 1.9689 | 7.8925 | 38.9843 | 156.2706 | 13.90 | 25.14 | 198/200 |
| 45 | 1.8626 | 7.5572 | 36.3204 | 147.3663 | 13.90 | 25.14 | 195/200 |
| 46 | 1.9140 | 4.9675 | 37.3239 | 96.8653 | 13.90 | 25.14 | 195/200 |
| **Mean $\pm$ Std** | **1.9400 $\pm$ 0.0632** | **6.1803 $\pm$ 1.9123** | **37.9527 $\pm$ 1.4994** | **121.1336 $\pm$ 38.2372** | **13.90** | **25.14** | **100%** |

---

## 3. Discrepancy Analysis

1. **Physical Impossibility of $13.90\text{ s}$ at Subtask Level**:
   - Under $B=10\text{ MHz}$, $P_v=1.0\text{ W}$, and $300\text{ m}$ RSU radius, transmission speed is $\approx 8.2\text{ Mbps}$.
   - A $2\text{ MB}$ subtask transmission requires $\approx 1.95\text{ s}$.
   - Therefore, a single completed subtask cannot exhibit $13.90\text{ s}$ delay without breaking Shannon channel physics.
2. **Workload Scale Discrepancy**:
   - Summing all $I=20$ subtasks per vehicle yields $\approx 39.5\text{ s}$, overshooting $13.90\text{ s}$ by $\approx 2.8\times$.
   - This indicates that Du et al. reported an intermediate aggregate (such as average active timeslot latency or partial pipeline makespan) without publishing the aggregation equation.
