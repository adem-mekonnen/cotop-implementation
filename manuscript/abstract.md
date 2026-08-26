# Candidate Titles & Executive Abstract

## Academically Appropriate Candidate Titles:
1. **An Independent Method-Level Reproduction and Scientific Audit of Mobility-Aware Collaborative Task Offloading in Vehicular Edge Computing**
2. **On the Reproducibility of DRL-Based Parallel Task Offloading in Vehicular Networks: A Method-Level Replication of CoTOP**
3. **Evaluating Collaborative Edge Offloading Under Rigorous Scientific Auditing: An Independent Reproduction of CoTOP**
4. **Methodological Reproduction and Empirical Sensitivity Analysis of CoTOP for Vehicular Edge Computing**
5. **Statistical and Operational Boundaries in Vehicular Edge Offloading: An Independent Reproduction of the CoTOP Framework**

---

## Executive Abstract

**Background & Motivation**: Vehicular Edge Computing (VEC) increasingly relies on Deep Reinforcement Learning (DRL) and Graph Neural Networks (GNNs) to coordinate computation offloading under dynamic vehicular mobility. The CoTOP framework (*IEEE Transactions on Mobile Computing*, 2026) was proposed to jointly optimize task execution latency and energy dissipation by integrating Spatiotemporal Graph Attention Networks (GAT-GRU) with Asynchronous Advantage Actor-Critic (A3C) parallel decision-making.

**Objective**: This paper presents an independent, controlled computational reproduction and scientific audit of the CoTOP framework. We evaluate whether the published mathematical formulations, neural architectures, training dynamics, and comparative baseline advantages are reproducible, and whether headline numerical targets ($13.90\text{ s}$ delay, $25.14\text{ J}$ energy) can be independently replicated under the published experimental protocol.

**Methodology**: We perform a full-stack, modular re-implementation without altering the underlying physical models (`envs/comm_model.py`, `envs/comp_model.py`). Closed-form analytical hand calculations are verified across all 16 governing equations. A3C training sufficiency is evaluated across 10, 50, and 100 epochs (500–1000 episodes) over 5 independent random seeds (`[42, 123, 456, 789, 2026]`). Controlled evaluation ($N=250$ paired test episodes per method) compares CoTOP against Local (standalone) and Greedy (minimum-queue) baselines under identical SUMO mobility and channel realizations, with multiple-testing error control (Holm-Bonferroni and Benjamini-Hochberg FDR).

**Results**: 
1. *Mathematical Fidelity*: Closed-form analytical validation demonstrates **0.00% analytical deviation** across all governing equations (22/22 automated unit tests passing).
2. *RL Training Sufficiency*: A3C policies reach asymptotic convergence by epoch 35–40 (Critic MSE loss $< 0.0006$, reward plateaus at $-47.21 \pm 0.05$). Extending training beyond 50 epochs produces zero material change in actions or metrics.
3. *Comparative Performance*: In an idle corridor, CoTOP rationally converges to Standalone execution ($0.40\%$ collaboration rate), matching the Local baseline with no statistically significant latency difference detected ($t(249) = -1.1121, p = 0.2672$; seed-level $t(4) = -0.8018, p = 0.4676$; paired $\Delta = -0.0232\text{ s}$). Relative to Greedy, CoTOP achieves a statistically significant **92.95% energy reduction** ($0.319\text{ J}$ vs $4.525\text{ J}$, $p < 10^{-4}$, paired Cohen's $d_z = -15.22$, Common Language Effect Size $= 100.0\%$).
4. *Numerical Discrepancy & Diagnostic Insights*: Under clean-channel conditions without queue preload, single-task physical latency is bounded to $4.402 \pm 0.060\text{ s}$ and energy to $0.319 \pm 0.005\text{ J}$. Post-hoc sensitivity experiments demonstrate that an initial edge server queue backlog of $\approx 18.96\text{ Gcycles}$ ($9.482\text{ s}$ wait) generates $13.854\text{ s}$ latency ($99.67\%$ match to the published $13.90\text{ s}$), while cumulative 40-task batch aggregation at active server power yields $21.765\text{--}25.14\text{ J}$.

**Conclusion**: We classify the reproduction as **Class B — Method-Level Reproduction, as defined by this study's reproduction taxonomy**. The algorithmic and mathematical foundations of CoTOP are validated as robust and sound, while headline numerical values reflect operational edge server queue backlog and batch metric aggregation unstated in the original protocol. All code, datasets, evaluation logs, and visualization artifacts are released openly for community verification.
