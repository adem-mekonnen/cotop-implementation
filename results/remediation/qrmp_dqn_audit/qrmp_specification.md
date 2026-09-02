# QRMP-DQN Formal Specification & Feasibility Audit

**Baseline Name**: QRMP-DQN (Quantile Regression Multi-Pass Deep Q-Network)  
**Cited Literature**: L. Guo, J. Jia, J. Chen, and X. Wang, 'QRMP-DQN Empowered Task Offloading and Resource Allocation for the STAR-RIS Assisted MEC Systems' (Reference [33] in Du et al. (IEEE TMC 2026))  
**Classification**: **NOT_REPRODUCIBLE_FROM_AVAILABLE_EVIDENCE**  

---

## 1. Algorithmic Identity
1. **Multi-Pass Deep Q-Network (MP-DQN)** (*Bester et al., AAAI 2019*):
   Designed for **Parameterized Action Space Markov Decision Processes (PAMDP)** where action $a = (k, x_k)$ pairs discrete action $k \in \{1, \dots, K\}$ with continuous parameter vector $x_k \in \mathbb{R}^{m_k}$.
2. **Quantile Regression DQN (QR-DQN)** (*Dabney et al., AAAI 2018*):
   Distributional RL approximating value distribution via $N$ uniform Dirac quantiles.
3. **Reference [33] Domain**:
   STAR-RIS (Simultaneously Transmitting and Reflecting Reconfigurable Intelligent Surface) MEC systems optimizing continuous reflection/transmission phase-shift matrices $\mathbf{\Phi}_t, \mathbf{\Phi}_r$ and continuous user power $\mathbf{p}$.

---

## 2. Incompatibility with Target Environment
In the discrete vehicular edge computing offloading environment of Du et al. (2026):
- Action space is **purely discrete**: $\mathcal{A} = \{0, 1, 2, 3, 4, 5, 6\}$.
- Continuous parameter vectors are **empty**: $x_k = \emptyset$.
- Under empty continuous parameters, MP-DQN mathematically collapses to single-pass DQN:
  $$Q(s, k, x_k) \equiv Q(s, k)$$
- Du et al. provide **zero equations**, **zero architectures**, and **zero hyperparameters** for QRMP-DQN.
- The author codebase contains **zero QRMP-DQN code files, classes, or checkpoints**.
