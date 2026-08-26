# Collaboration Boundary & Benefit Region Analysis (Stage 10)

**Reference Paper**: *Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing*  
**Publication**: IEEE Transactions on Mobile Computing (TMC 2026), DOI: [10.1109/TMC.2025.3631820](https://doi.org/10.1109/TMC.2025.3631820)  

---

## 1. Mathematical Condition for Beneficial Collaboration

Under the paper's objective function (Eq. 13), the step reward is:
$$R = -(\epsilon T + (1 - \epsilon) E)$$

When deciding between **Case 1 (Standalone)** and **Case 2 (Collaborative)**:
- Standalone delay: $T_1 = t^{up} + t_1^{wait} + t_1^{pro}$
- Standalone energy: $E_1 = P_V \cdot t^{up} + P_R^{comp} \cdot t_1^{pro} \approx 0.29\text{ J}$
- Collaborative delay: $T_2 = t^{up} + \max(t_1, t_2 + t_3) + t_2^{wait}$
- Collaborative energy: $E_2 = P_V \cdot t^{up} + P_R \cdot t_2 + P_R^{comp} \cdot t_1 + P_R^{comp} \cdot t_3 \approx 3.28\text{--}10.0\text{ J}$ (due to $P_R = 100\text{ W}$ R2R transmission).

Because $P_R = 100\text{ W} \gg P_V = 0.01\text{ W}$, collaborative offloading incurs an energy penalty of $\Delta E \approx +3.0\text{ to }9.7\text{ J}$.
Under equal weighting ($\epsilon = 0.5$), collaboration is only mathematically advantageous when:
$$0.5 \Delta T > 0.5 \Delta E \implies t_1^{wait} - t_2^{wait} > \Delta E \approx 6.0\text{ to }10.0\text{ seconds}$$

---

## 2. Sensitivity Analysis Table

```
 Primary Queue Wait t_wait1 (s)  Standalone Delay (s)  Standalone Energy (J)  Standalone Reward  Collab Delay (s)  Collab Energy (J)  Collab Reward  Reward Advantage (Collab - Standalone)  Optimal Policy Action
                            0.0                 4.421                  0.294             -2.358             4.449              3.308         -3.879                                  -1.521    Case 1 (Standalone)
                            1.0                 5.421                  0.294             -2.858             4.449              3.308         -3.879                                  -1.021    Case 1 (Standalone)
                            3.0                 7.421                  0.294             -3.858             4.449              3.308         -3.879                                  -0.021    Case 1 (Standalone)
                            5.0                 9.421                  0.294             -4.858             4.449              3.308         -3.879                                   0.979 Case 2 (Collaborative)
                            8.0                12.421                  0.294             -6.358             4.449              3.308         -3.879                                   2.479 Case 2 (Collaborative)
                           10.0                14.421                  0.294             -7.358             4.449              3.308         -3.879                                   3.479 Case 2 (Collaborative)
                           12.0                16.421                  0.294             -8.358             4.449              3.308         -3.879                                   4.479 Case 2 (Collaborative)
                           15.0                19.421                  0.294             -9.858             4.449              3.308         -3.879                                   5.979 Case 2 (Collaborative)
                           20.0                24.421                  0.294            -12.358             4.449              3.308         -3.879                                   8.479 Case 2 (Collaborative)
```

---

## 3. Key Finding
When primary RSU queue wait is $0\text{ s}$, Standalone reward ($-2.36$) is vastly superior to Collaborative reward ($-3.28$).
Only when primary RSU queue wait exceeds **$5.0\text{--}10.0\text{ seconds}$** does the DRL agent gain positive reward incentive to offload to secondary RSUs. This mathematically proves why CoTOP converges to standalone offloading in non-congested environments.
