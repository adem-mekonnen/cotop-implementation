# Action-to-Physics Execution Trace & Differentiation Audit

**Reference Paper**: IEEE Transactions on Mobile Computing (TMC 2026), DOI: 10.1109/TMC.2025.3631820  

This document details the exact execution pathway from agent action selection to physical delay, energy, queue updates, and reward calculation.

---

## 1. Action Space Architecture ($M = 6$ RSUs)

The action space is defined as $\mathcal{A} = \{0, 1, \dots, M\}$ (`gymnasium.spaces.Discrete(7)`):

| Action Index | Physical Semantics | Primary RSU | Target Secondary RSU | Execution Mode | Governing Equations |
| :---: | :--- | :---: | :---: | :--- | :--- |
| `0` | Standalone Processing | Nearest RSU $m$ | None | Case 1 | Eq. (3), (4), (5), (6), (11), (12) |
| `1` | Collaborative Offloading | Nearest RSU $m$ | RSU 0 | Case 2 (or Case 1 if $m=0$) | Eq. (1), (2), (7), (8), (9), (10), (11), (12) |
| `2` | Collaborative Offloading | Nearest RSU $m$ | RSU 1 | Case 2 (or Case 1 if $m=1$) | Eq. (1), (2), (7), (8), (9), (10), (11), (12) |
| `3` | Collaborative Offloading | Nearest RSU $m$ | RSU 2 | Case 2 (or Case 1 if $m=2$) | Eq. (1), (2), (7), (8), (9), (10), (11), (12) |
| `4` | Collaborative Offloading | Nearest RSU $m$ | RSU 3 | Case 2 (or Case 1 if $m=3$) | Eq. (1), (2), (7), (8), (9), (10), (11), (12) |
| `5` | Collaborative Offloading | Nearest RSU $m$ | RSU 4 | Case 2 (or Case 1 if $m=4$) | Eq. (1), (2), (7), (8), (9), (10), (11), (12) |
| `6` | Collaborative Offloading | Nearest RSU $m$ | RSU 5 | Case 2 (or Case 1 if $m=5$) | Eq. (1), (2), (7), (8), (9), (10), (11), (12) |

---

## 2. End-to-End Execution Trace

```
1. Agent selects action a in {0, ..., 6}
   │
2. Environment decodes target RSUs:
   ├── Primary RSU m = min_distance(vehicle.pos, rsu.locations)
   └── If a > 0: Secondary RSU m' = rsus[a - 1]
   │
3. V2R Communication:
   └── w_V2R = B_V2R * log2(1 + (P_V * K) / (omega * D_{n,m}^2))    [Eq. 1]
   └── t_up = rho_{n,i} / w_V2R                                    [Eq. 3]
   │
4. Execution Branching:
   ├── If a == 0 or m == m' (Case 1: Standalone):
   │   ├── t_pro = phi_{n,i} / F_m                                 [Eq. 4]
   │   ├── t_wait = N_m^queue / F_m                                [Eq. 5]
   │   ├── T_total = t_up + t_pro + t_wait                         [Eq. 6]
   │   ├── E_total = P_V * t_up + t_pro * E_RSU                    [Eq. 11, 12]
   │   └── N_m^queue += phi_{n,i}
   │
   └── If a > 0 and m != m' (Case 2: Collaborative Parallel):
       ├── R2R Rate w_R2R = B_R2R * log2(1 + (P_R * K)/(omega * D_{m,m'}^2)) [Eq. 2]
       ├── Task Partition: phi1 = F1 * t1, phi_rest = phi - phi1     [Eq. 7]
       ├── R2R Transfer: t2 = (rho * (phi_rest / phi)) / w_R2R       [Eq. 8]
       ├── Secondary Compute: t3 = phi_rest / F_{m'}                 [Eq. 9]
       ├── Parallel Processing: T_pro = max(t1, t2 + t3)
       ├── Secondary Wait: t_wait = N_{m'}^queue / F_{m'}            [Eq. 10]
       ├── T_total = t_up + T_pro + t_wait                           [Eq. 10]
       ├── E_total = P_V*t_up + t1*E1 + P_R*t2 + t3*E2               [Eq. 11, 12]
       ├── N_m^queue += phi1
       └── N_{m'}^queue += phi_rest
```
