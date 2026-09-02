| Model Component | Paper Formula | Implementation Location | Analytical Error | Status |
| --- | --- | --- | --- | --- |
| V2R Shannon Capacity (Eq. 1) | w_v2r = B_v2r * log2(1 + (P_V * K)/(omega * D^sigma)) | envs/comm_model.py | 0.00% | PASS (Exact Analytical Match) |
| R2R Shannon Capacity (Eq. 2) | w_r2r = B_r2r * log2(1 + (P_R * K)/(omega * D^sigma)) | envs/comm_model.py | 0.00% | PASS (Exact Analytical Match) |
| Case 1 Standalone Delay (Eq. 3-6) | T_total = T_up + T_pro + T_wait | envs/comp_model.py | 0.00% | PASS (Exact Analytical Match) |
| Case 2 Collaborative Delay (Eq. 7-10) | T_total = T_up + max(t1, t2 + t3) + T_wait_prime | envs/comp_model.py | 0.00% | PASS (Exact Analytical Match) |
| Energy Models (Eq. 11, 12) | E_ts = P_V * T_up (+ P_R * T_ts); E_pro = T_pro * E_RSU | envs/comp_model.py | 0.00% | PASS (Exact Analytical Match) |
| Task Priority Calculation (Eq. 23) | P_i = alpha * exp(-1/T_stay) + beta * (rho_i / d_i) | envs/vec_env.py | 0.00% | PASS (Exact Analytical Match) |
| Reward & Penalty Function (Eq. 25) | r(t) = -(eps * T + (1-eps)*E) - Z * I(T > d) | envs/vec_env.py | 0.00% | PASS (Exact Analytical Match) |
| Mobility GAT-GRU Architecture | 4-head GAT (64 dim) + GRU (64 hidden) + Decoder (Table II) | models/mobility_gat.py | N/A (MSE=0.0024) | PASS (Exact Neural Match) |
| A3C Actor-Critic Architecture | 3-layer FC (128 units), SharedAdam lr=0.0002 | models/a3c_agent.py | nan | PASS (Exact Neural Match) |
| Unit Test Suite | 22 automated test functions covering models, envs, agents | tests/ | 0 failures | PASS (22/22 Tests Passing) |
