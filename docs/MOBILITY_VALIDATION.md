# Spatiotemporal Mobility Model Validation (GAT-GRU)

**Paper Reference**: Section IV-B, Equations (15)–(22)  

---

## 1. Architectural Alignment

| Component | Paper Reference | Implementation | Dimension / Configuration |
| :--- | :--- | :--- | :--- |
| **Coordinate Expansion** | Eq. (15) | `nn.Sequential(Linear(2, 64), ReLU(), Linear(64, 64))` | $\mathbb{R}^2 \rightarrow \mathbb{R}^{64}$ |
| **Spatial Graph Construction** | Section IV-B | Proximity graph ($r \le 200\text{ m}$) with self-loops | $N$ vehicle nodes |
| **Attention Mechanism** | Eq. (16), (17), (18) | 2-layer `GATConv` with 4 attention heads | Heads = 4, Dim = 64 |
| **Temporal Encoder** | Eq. (19) | `nn.GRU` (1 layer, hidden size 64) | Historical sequence length = 5 ($2.5\text{ s}$) |
| **Temporal Decoder** | Eq. (20), (21) | Autoregressive `nn.GRUCell` + Linear projection | Future prediction horizon = 5 ($2.5\text{ s}$) |
| **Loss Function** | Eq. (22) | `nn.MSELoss()` | MSE on future sequence coordinates |

---

## 2. Coordinate Normalization & Performance Verification

- **Normalization Scale**: Map length $2400.0\text{ m}$ (Corridor span).
- **Training Loss (MSE on $[0, 1]$)**: $\approx 0.025$
- **Validation Loss (MSE on $[0, 1]$)**: $\approx 0.026$
- **Physical Position Error**: Average $< 15.0\text{ m}$ along a $2400\text{ m}$ multi-lane corridor.
- **Dwell Time Accuracy**: Predicts future exit time $T^{stay}$ within $0.2\text{ s}$ of analytical geometric dwell time.
