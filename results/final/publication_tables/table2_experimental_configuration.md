# Table 2: Experimental Matrix & Environment Configuration

| Parameter | Symbol | Value | Unit |
| :--- | :--- | :--- | :--- |
| Scenarios | - | `corridor_2400m`, `grid_200m` | - |
| Workloads | $I_n$ | 20, 30, 40 | subtasks / vehicle |
| Evaluation Seeds | - | 42, 43, 44, 45, 46, 47, 48, 49, 50, 51 | 10 seeds |
| Factorial Matrix | - | 4 Algorithms × 2 Scenarios × 3 Workloads × 10 Seeds | 240 runs |
| Frozen Realizations | - | 60 Exogenous Trace Files | SHA-256 Verified |
| Vehicle Speed | $v$ | 10 – 20 | m/s |
| RSU Radius | $R$ | 200 | m |
| Subtask Data Size | $\rho_{n,k}$ | 1.0 – 5.0 | Mbits |
| Subtask CPU Demand | $\phi_{n,k}$ | 1.0 – 5.0 | Gcycles |
| Primary RSU Frequency | $f_0$ | 4.0 | GHz |
| Collab RSU Frequency | $f_m$ | 2.0 | GHz |
