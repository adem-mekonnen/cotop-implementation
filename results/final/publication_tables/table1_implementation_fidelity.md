# Table 1: Implementation Fidelity Audit Matrix

| Equation / Module | Paper Target | Code Reference | Mathematical Fidelity | Invariant Status |
| :--- | :--- | :--- | :--- | :--- |
| **Eq. (1–6)** Channel Rates | Shannon V2R/R2R | `envs/comm_model.py` | Exact closed-form | IMMUTABLE |
| **Eq. (7–14)** Computing Models | Standalone & Collab | `envs/comp_model.py` | Exact closed-form | IMMUTABLE |
| **Eq. (15–18)** Spatial GAT | Multi-Head Concatenation & Mean Head (Layer 2) | `models/mobility_gat.py` | Exact Eq. (18) Head-Averaging | VERIFIED |
| **Eq. (19–21)** Temporal GRU | Autoregressive GRU Decoder | `models/mobility_gat.py` | Exact GRU sequence | VERIFIED |
| **Eq. (23)** Priority | $P_n = e^{-\lambda D_n} + \mu(1 - T_{stay}/D_n)$ | `utils/task_priority.py` | Exact formula | VERIFIED |
| **Eq. (25)** Composite Reward | $R = - (\alpha T + \beta E) - Z$ | `envs/vec_env.py` | Exact penalty $-Z$ | VERIFIED |
| **QRMP-DQN** | STAR-RIS Phase Optimization | Reference [33] | Continuous Domain Mismatch | EXCLUDED |
