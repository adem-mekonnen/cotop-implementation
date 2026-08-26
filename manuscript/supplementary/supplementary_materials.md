# Supplementary Materials: CoTOP Mathematical Derivations & Reproduction Traces

This document provides extended mathematical derivations, unit test traces, and experimental reproducibility details supporting the main manuscript.

---

## S1. Full Mathematical System Model & Closed-Form Derivations

### S1.1 Communication Models
Let $B^{V2R}$ and $B^{R2R}$ denote the wireless bandwidths of vehicle-to-RSU (V2R) uplink and inter-RSU (R2R) backhaul channels, respectively. Under the Shannon-Hartley theorem with log-distance path loss:

1. **V2R Uplink Transmission Capacity (Eq. 1)**:
   $$w_{n,m}^{V2R} = B^{V2R} \log_2 \left(1 + \frac{P_V K}{\omega D_{n,m}^\sigma}\right)$$
   *Test Vector*: $B^{V2R} = 20.0\text{ MHz}, P_V = 0.01\text{ W}, K = 1000.0, \sigma = 2.0, \omega = 0.001\text{ W}, D = 200.0\text{ m}$.  
   $$\text{SNR} = \frac{0.01 \times 1000.0}{0.001 \times 40000.0} = 0.25 \implies w^{V2R} = 20 \times 10^6 \times \log_2(1.25) \approx 6.438562\text{ Mbps}$$

2. **R2R Inter-RSU Capacity (Eq. 2)**:
   $$w_{m,m'}^{R2R} = B^{R2R} \log_2 \left(1 + \frac{P_R K}{\omega D_{m,m'}^\sigma}\right)$$
   *Test Vector*: $B^{R2R} = 50.0\text{ MHz}, P_R = 100.0\text{ W}, D = 400.0\text{ m}$.  
   $$\text{SNR} = \frac{100.0 \times 1000.0}{0.001 \times 160000.0} = 625.0 \implies w^{R2R} = 50 \times 10^6 \times \log_2(626.0) \approx 464.500942\text{ Mbps}$$

---

### S1.2 Case 1: Standalone Offloading (Eq. 3–6, 11, 12)
1. **Upload Latency**: $T_{\text{up}} = \frac{\rho_{n,i} \times 8}{w_{n,m}^{V2R}}$
2. **Compute Latency**: $T_{\text{pro}} = \frac{\phi_{n,i}}{F_m}$
3. **Queue Waiting Latency**: $T_{\text{wait}} = \frac{N_m^{\text{queue}}}{F_m}$
4. **Total Latency**: $T_{\text{total}}^{\text{Case1}} = T_{\text{up}} + T_{\text{pro}} + T_{\text{wait}}$
5. **Energy Dissipation**:
   $$E_{\text{total}}^{\text{Case1}} = P_V T_{\text{up}} + E_{\text{RSU}} T_{\text{pro}}$$

---

### S1.3 Case 2: Parallel Collaborative Offloading (Eq. 7–10, 11, 12)
1. **Primary RSU Workload Execution**: $\phi_1 = F_m t_1$ during dwell time $t_1$.
2. **Remaining Workload**: $\phi_{\text{rest}} = \phi_{n,i} - \phi_1$.
3. **R2R Data Transfer Latency**:
   $$T_{\text{ts}} = \frac{\rho_{n,i} \times 8 \times (\phi_{\text{rest}} / \phi_{n,i})}{w_{m,m'}^{R2R}}$$
4. **Secondary Compute Latency**: $T_{\text{pro\_rest}} = \frac{\phi_{\text{rest}}}{F_{m'}}$
5. **Total Collaborative Latency**:
   $$T_{\text{total}}^{\text{Case2}} = T_{\text{up}} + \max(t_1, T_{\text{ts}} + T_{\text{pro\_rest}}) + T_{\text{wait}'}$$
6. **Total Collaborative Energy**:
   $$E_{\text{total}}^{\text{Case2}} = P_V T_{\text{up}} + P_R T_{\text{ts}} + E_{\text{RSU}} (t_1 + T_{\text{pro\_rest}})$$

---

### S1.4 Task Priority Function (Eq. 23)
$$P_i = \alpha e^{-1/T^{\text{stay}}} + \beta \left(\frac{\rho_i \times 8}{d_i}\right), \quad \alpha = 0.3, \beta = 0.7$$
*Analytical Test*: $\rho = 10^5\text{ Bytes} = 8 \times 10^5\text{ bits}, d = 10\text{ s}, T^{\text{stay}} = 10\text{ s} \implies P_i = 0.3 e^{-0.1} + 0.7 (80000) = 0.271451 + 56000 = 56000.271451$. Verified exact in `sanity_check.py`.

---

## S2. Unit Test Suite Execution Trace

Execution log from running `pytest tests/` on the frozen repository (`5b115ae6a77ba08640d555e77717cc85b757668c`):

```text
============================= test session starts =============================
platform win32 -- Python 3.11.1, pytest-8.3.4, pluggy-1.5.0
rootdir: d:\cotop-implementation
collected 22 items

tests/test_baselines.py ....                                             [ 18%]
tests/test_comm_model.py ..                                              [ 27%]
tests/test_comp_model.py ....                                            [ 45%]
tests/test_energy_model.py ..                                            [ 54%]
tests/test_queue_model.py ..                                             [ 63%]
tests/test_reward.py ..                                                  [ 72%]
tests/test_state_builder.py ..                                           [ 81%]
tests/test_task_priority.py ..                                           [ 90%]
tests/integration/test_single_vehicle.py ..                               [100%]

============================== 22 passed in 5.15s ==============================
```
