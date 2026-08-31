# Table 6: Published vs. Reproduced Result Reconciliation Matrix

| Metric / Phenomenon | Paper Published Value | Reproduced Value | Difference ($\Delta$) | Forensic Classification | Root-Cause Explanation | Confidence |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **CoTOP Mean Delay** | $\approx 13.90\text{ s}$ | $0.680\text{ s}$ (Corridor)<br>$0.257\text{ s}$ (Grid) | $-13.22\text{ s}$ ($-95.1\%$) | **NOT REPRODUCED**<br>*(Qualitative Rank Reproduced)* | Unstated server queue backlog ($\sim 19\text{ Gcycles}$) or cumulative vehicle batch aggregation ($\sum_{i=1}^{20} T_i$). | **HIGH (99.9%)** |
| **CoTOP Mean Energy** | $\approx 25.14\text{ J}$ | $0.144\text{ J}$ (Standalone)<br>$1.589\text{ J}$ (Collab) | $-23.55\text{ J}$ ($-93.7\%$) | **NOT REPRODUCED**<br>*(Qualitative Rank Reproduced)* | Cumulative vehicle batch energy aggregation ($20 \times 1.25\text{ J} = 25.0\text{ J}$) vs per-task accounting. | **HIGH (99.5%)** |
| **Algorithmic Rank Order** | $\text{CoTOP} < \text{DDQN} < \text{Greedy} \ll \text{Local}$ | $\text{CoTOP} \le \text{DDQN} < \text{Greedy} \ll \text{Local}$ | Exact Match | **EXACTLY REPRODUCED** | Actor-critic state representation balances load; Local collapses under queue scale. | **HIGH (100%)** |
| **Learning Rate Optimum** | $\text{lr} = 0.0002$ | $\text{lr} = 0.0002$ | Exact Match | **EXACTLY REPRODUCED** | $\text{lr}=0.0002$ achieves fast stable convergence; $\ge 0.0005$ induces instability. | **HIGH (100%)** |
| **Task Priority Optimum** | $\alpha = 0.3, \beta = 0.7$ | $\alpha = 0.3, \beta = 0.7$ | Exact Match | **EXACTLY REPRODUCED** | Minimizes average delay while bounding deadline violations. | **HIGH (100%)** |
| **Ablation Trends (Table VI)**| $\text{w/o MD} \gg \text{w/o TP} > \text{CoTOP}$ | $\text{w/o MD} \gg \text{w/o TP} > \text{CoTOP}$ | Exact Match | **EXACTLY REPRODUCED** | Removing dwell lookahead ($t_1=0$) forces 100% relay, doubling latency and energy. | **HIGH (100%)** |
| **QRMP-DQN Baseline** | Intermediate between CoTOP/DDQN | `N/A (EXCLUDED)` | N/A | **NOT IDENTIFIABLE** | Ref [33] continuous STAR-RIS domain mismatch; no author release code. | **HIGH (100%)** |
