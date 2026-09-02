# Phase 8 Metric Definitions & Statistical Formulas

This document defines the mathematical equations and statistical metrics used in the Phase 8 ablation and cross-algorithm comparative audit.

## 1. Paired Statistical Differences
For a set of $N = 60$ identical frozen realization instances evaluated under both policy $A$ (e.g., CoTOP) and policy $B$ (e.g., Local, Greedy, DDQN, ablations):

$$\Delta_i = X_i^{(A)} - X_i^{(B)}, \quad i \in \{1, \dots, N\}$$

### Mean Paired Difference:
$$\overline{\Delta} = \frac{1}{N} \sum_{i=1}^{N} \Delta_i$$

### Standard Deviation of Paired Differences:
$$S_\Delta = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (\Delta_i - \overline{\Delta})^2}$$

### Standard Error of the Mean Difference:
$$SE(\overline{\Delta}) = \frac{S_\Delta}{\sqrt{N}}$$

### Two-Sided 95% Confidence Interval:
$$CI_{95\%} = \overline{\Delta} \pm t_{0.025, N-1} \cdot SE(\overline{\Delta})$$

Where $t_{0.025, 59} \approx 2.000995$.

## 2. Effect Size (Cohen's $d$)
$$d = \frac{\overline{\Delta}}{S_\Delta}$$
- $|d| < 0.2$: Negligible effect
- $0.2 \le |d| < 0.5$: Small effect
- $0.5 \le |d| < 0.8$: Medium effect
- $|d| \ge 0.8$: Large effect

## 3. Paired Significance Tests
1. **Paired Student's $t$-test**:
   $$t = \frac{\overline{\Delta}}{SE(\overline{\Delta})}$$
2. **Wilcoxon Signed-Rank Test**:
   Non-parametric paired rank sum test for paired differences $\Delta_i$, accounting for non-normal or skewed execution delays.

## 4. Operational Metrics
- **Task Delay ($T_k$)**: Total wall-clock time in seconds from offloading decision to completion on vehicle or RSU.
- **Dynamic Energy ($E_k$)**: Cumulative energy in Joules consumed by vehicle computation, V2R transmission, and inter-RSU forwarding.
- **Task Completion Ratio ($R_{\text{comp}}$)**:
  $$R_{\text{comp}} = \frac{N_{\text{completed}}}{N_{\text{total}}}$$
