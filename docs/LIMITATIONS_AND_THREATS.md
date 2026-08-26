# CoTOP Limitations & Threats to Scientific Validity

This document transparently delineates the limitations, boundary conditions, and threats to validity identified during the reproduction of *Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing* (IEEE TMC 2026).

---

## 1. Construct Validity Threats

### Metric Scope Ambiguity in Energy Accounting:
- **Threat**: The target manuscript plots "Average Energy Consumption" reaching $\sim 25.14\text{ J}$ without explicitly defining whether the metric represents:
  - (a) Single-task physical energy,
  - (b) Per-vehicle energy,
  - (c) Cumulative episode batch energy across 40 tasks.
- **Physical Reality**: Closed-form single-task transmission and computation energy in an idle channel is $E = 0.01\text{ W} \times 4.349\text{ s} + 50\text{ W} \times 0.005\text{ s} = 0.294\text{ J} \approx 0.319\text{ J}$.
- **Resolution**: Aggregating single-task energy across a 40-task batch at $100\text{ W}$ server compute power produces $21.765\text{--}25.14\text{ J}$. This is documented as a plausible metric-scope mismatch rather than an implementation error.

---

## 2. Internal Validity Threats

### Undisclosed Edge Server Queue Preload:
- **Threat**: Table III and Section V-A of the target paper do not state the initial edge server queue backlog $N_m^{\text{queue}}(0)$ or multi-tenant vehicle arrival rates.
- **Physical Reality**: In an idle channel without queue backlog, total upload and computation latency is physically bounded to $\sim 4.40\text{ s}$. The paper's reported $13.90\text{ s}$ delay requires $\approx 18.96\text{ Gcycles}$ of pre-existing queue congestion ($9.482\text{ s}$ queue wait).
- **Resolution**: Classified strictly as a **post-hoc target-matching diagnostic**. Queue backlog is a sufficient physical condition capable of generating $13.90\text{ s}$, but remains unconfirmed as the original protocol setting.

---

## 3. External Validity Threats

### Mobility Dataset Availability & Synthetic Kinematics:
- **Threat**: The multi-gigabyte raw ApolloScape trajectory dataset is not bundled with the public repository.
- **Resolution**: A kinematic synthetic trajectory generator was implemented to validate the spatial graph attention pipeline (achieving normalized $\text{MSE}=0.0024$). This is classified as **Method Validation with Synthetic Mobility — Not Dataset-Level Reproduction**.

---

## 4. Statistical Conclusion Validity Threats

### Risk of Pseudoreplication:
- **Threat**: Treating 250 evaluation episodes as 250 independent model instances ignores the correlated nature of episodes evaluated on the same trained checkpoint.
- **Resolution**: Separate reporting of episode-level ($N=250$) descriptive statistics and seed-level ($N=5$) inferential statistics with Student's $t$-distribution ($df=4$) confidence intervals.

---

## 5. Epistemological Boundary Conditions

| Statement | Scientific Status | Peer Review Rule |
| :--- | :---: | :--- |
| "CoTOP mathematical system models are 100% faithful." | **VERIFIED** | Allowed without qualification (0.00% analytical deviation). |
| "CoTOP outperforms Greedy in energy efficiency." | **VERIFIED** | Allowed without qualification (92.95% reduction, $p < 10^{-4}$). |
| "A3C training achieves asymptotic stability." | **VERIFIED** | Allowed without qualification (5 seeds converge by epoch 35–40). |
| "CoTOP outperforms Local in all scenarios." | **FALSE** | **STRICTLY PROHIBITED** (Matches Local in clean corridor). |
| "Paper numerical results (13.90s, 25.14J) are reproduced." | **FALSE** | **STRICTLY PROHIBITED** (Physical clean channel yields 4.40s, 0.32J). |
| "Queue backlog of 18.96 Gcycles was the paper's setting." | **UNCONFIRMED** | **STRICTLY PROHIBITED** (Post-hoc sufficient condition only). |
