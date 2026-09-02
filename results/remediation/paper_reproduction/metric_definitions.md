# Metric Definitions and Mathematical Formulations

This document formalizes the mathematical and algorithmic definitions of experimental metrics evaluated during the CoTOP research reproduction study.

---

## 1. Latency (Total Delay) Formulation

For task $k = (n, i)$ of vehicle $n$:

### Case 1: Standalone Primary RSU Execution ($a_{n,i} = 0$)
$$T_{n,i}^{\text{local}} = t_{n,i}^{\text{up}} + t_{n,i}^{\text{wait}} + t_{n,i}^{\text{comp}}$$
where:
- $t_{n,i}^{\text{up}} = \frac{\rho_{n,i}}{r_{n, \text{pri}}}$ is the V2R wireless uplink transmission delay (Shannon rate $r_{n, \text{pri}} = B_v \log_2(1 + \text{SINR})$).
- $t_{n,i}^{\text{wait}} = \frac{Q_{\text{pri}}}{f_0}$ is the queue waiting time at primary RSU.
- $t_{n,i}^{\text{comp}} = \frac{\phi_{n,i}}{f_0}$ is the execution time at primary RSU frequency $f_0$.

### Case 2: Parallel Collaborative Offloading ($a_{n,i} > 0$)
$$T_{n,i}^{\text{collab}} = t_{n,i}^{\text{up}} + \max\left(t_{n,i}^{\text{comp1}}, t_{n,i}^{\text{r2r}} + t_{n,i}^{\text{comp2}}\right)$$
where:
- $t_{n,i}^{\text{comp1}} = \frac{\phi_{n,i}^{(1)}}{f_0}$ is compute time on primary RSU for partition $\phi^{(1)}$.
- $t_{n,i}^{\text{r2r}} = \frac{\rho_{n,i}^{(2)}}{w_{\text{r2r}}}$ is R2R relay transmission delay over wired/wireless fiber.
- $t_{n,i}^{\text{comp2}} = \frac{\phi_{n,i}^{(2)}}{f_m}$ is compute time on secondary RSU $m$.

### Aggregate Mean Delay
$$\bar{T} = \frac{1}{K} \sum_{k=1}^K T_k$$
where $K$ is the total count of evaluated tasks (or completed tasks).

---

## 2. Dynamic Energy Consumption Formulation

### Case 1: Standalone Execution
$$E_{n,i}^{\text{local}} = P_V \cdot t_{n,i}^{\text{up}} + P_{\text{RSU}} \cdot t_{n,i}^{\text{comp}}$$
where:
- $P_V = 0.01\text{ W}$ ($10\text{ dBm}$) is vehicle uplink transmit power.
- $P_{\text{RSU}} = 50.0\text{ W}$ is RSU active processing power.

### Case 2: Collaborative Execution
$$E_{n,i}^{\text{collab}} = P_V \cdot t_{n,i}^{\text{up}} + P_{\text{RSU}} \cdot t_{n,i}^{\text{comp1}} + P_R \cdot t_{n,i}^{\text{r2r}} + P_{\text{RSU}} \cdot t_{n,i}^{\text{comp2}}$$
where:
- $P_R = 100.0\text{ W}$ ($50\text{ dBm}$) is RSU inter-relay transmission power.

### Aggregate Mean Energy
$$\bar{E} = \frac{1}{K} \sum_{k=1}^K E_k$$

---

## 3. Task Completion Ratio Formulation

$$\text{Completion Ratio} = \frac{\sum_{k=1}^K \mathbb{I}(\text{Completed}_k)}{K}$$
where a task completes ($\text{Completed}_k = 1$) if and only if:
1. $T_k \le D_k$ (Execution latency is strictly within deadline $D_k$).
2. The vehicle is within the physical coverage boundary of the serving RSU(s) at execution completion:
   - Case 1: $\text{dist}(\text{pos}_{\text{comp}}, \text{RSU}_{\text{pri}}) \le R_{\text{comm}}$.
   - Case 2: $\left(\text{dist}(\text{pos}_{\text{comp}}, \text{RSU}_{\text{pri}}) \le R_{\text{comm}}\right) \lor \left(\text{dist}(\text{pos}_{\text{comp}}, \text{RSU}_{\text{sec}}) \le R_{\text{comm}}\right)$.
