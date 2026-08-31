# QRMP-DQN Forensic Disposition & Formal Exclusion Record

**Document ID**: `DOC-DISPOSITION-QRMP-DQN-001`  
**Classification**: Formal Methodological Exclusion Record  
**Target Paper**: *Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing* (Du et al., IEEE TMC 2026)  
**Cited Baseline Reference**: Reference [33] (Guo et al.)

---

## 1. Reference Identity

- **Citation in Target Paper**: Reference [33]
- **Full Bibliographic Reference**:
  > L. Guo, J. Jia, J. Chen, and X. Wang, *"QRMP-DQN Empowered Task Offloading and Resource Allocation for the STAR-RIS Assisted MEC Systems"*, IEEE Transactions on Wireless Communications / IEEE Access / Computer Communications.
- **Citation Context in Du et al. (IEEE TMC 2026)**:
  - Mentioned in Section V-B (*Baseline Methods*, page 5548, lines 66–67):
    > *"QRMP-DQN [33]: The Quantile Regression Multi-Pass Deep Q-Network (QRMP-DQN) is an algorithm based on quantile regression and multi-pass deep Q-network..."*
  - Included as a comparative curve in Figure 4, Figure 5, Figure 6, and tabular comparisons in Table IV and Table V of Du et al.

---

## 2. Algorithm Definition (What QRMP-DQN Actually Is)

To understand QRMP-DQN, one must trace its two foundational components in the reinforcement learning literature:

1. **Multi-Pass Deep Q-Network (MP-DQN)** (*Bester et al., AAAI 2019*):
   - Formulated specifically for **Parameterized Action Space Markov Decision Processes (PAMDP)**.
   - In a PAMDP, an action $a = (k, x_k)$ consists of a discrete action choice $k \in \{1, \dots, K\}$ and a continuous parameter vector $x_k \in \mathcal{X}_k \subseteq \mathbb{R}^{m_k}$ associated with discrete action $k$.
   - Standard DQN cannot evaluate continuous parameters. MP-DQN uses an actor network $x_k = \mu(s; \theta_\mu)$ to generate continuous parameter vectors for all $K$ discrete actions, and then conducts $K$ distinct forward passes through a Q-network $Q(s, k, x_k; \theta_Q)$ to select $\arg\max_k Q(s, k, x_k)$.

2. **Quantile Regression DQN (QR-DQN)** (*Dabney et al., AAAI 2018*):
   - A distributional reinforcement learning algorithm where the scalar value function $Q(s, a)$ is replaced by a distribution over returns modeled by $N$ uniform Dirac quantiles $\theta_i(s, a)$ trained via the quantile Huber loss.

3. **QRMP-DQN in Reference [33] (Guo et al.)**:
   - Guo et al. addresses a **Simultaneously Transmitting and Reflecting Reconfigurable Intelligent Surfaces (STAR-RIS)** assisted Multi-Access Edge Computing (MEC) system.
   - The action in [33] is explicitly hybrid-parameterized:
     $$a = (d, \mathbf{\Phi}_t, \mathbf{\Phi}_r, \mathbf{p})$$
     where $d \in \{1, \dots, D\}$ represents the discrete MEC server offloading decision, $\mathbf{\Phi}_t, \mathbf{\Phi}_r \in \mathbb{C}^{M \times M}$ represent continuous phase-shift matrices for the STAR-RIS elements, and $\mathbf{p} \in \mathbb{R}^K$ represents continuous user transmit power allocation.
   - QRMP-DQN couples the Multi-Pass parameterized action architecture with Quantile Regression to optimize the continuous reflection/transmission coefficients while selecting the discrete MEC server.

---

## 3. Target Paper's Description of QRMP-DQN

In Du et al. (IEEE TMC 2026), Section V-B describes QRMP-DQN as follows:
> *"QRMP-DQN [33]: The Quantile Regression Multi-Pass Deep Q-Network (QRMP-DQN) is an algorithm based on quantile regression and multi-pass deep Q-network to achieve collaborative offloading in vehicular edge computing systems."*

Beyond this single introductory sentence, the target manuscript provides:
- **Zero mathematical formulations** for the algorithm.
- **Zero neural network architecture specifications** (e.g., number of layers, hidden units, quantile counts $N$).
- **Zero continuous parameter mappings** or actor network definitions.
- **Zero hyperparameter entries** in Table III (which lists parameters for CoTOP, but completely omits QRMP-DQN).

---

## 4. Action-Space Compatibility Analysis

A fundamental structural conflict exists between the target vehicular environment (Du et al.) and the operational requirement of Reference [33]:

| Dimension | Target Environment (Du et al. 2026) | Reference [33] (Guo et al. STAR-RIS) | Compatibility Status |
| :--- | :--- | :--- | :--- |
| **Action Space Structure** | Purely Discrete 7-Action Space $\mathcal{A} = \{0, 1, \dots, 6\}$ | Hybrid Parameterized Action Space $(d, x_d) \in \mathcal{D} \times \mathbb{R}^m$ | **INCOMPATIBLE** |
| **Physical Entities** | 6 Roadside Units (RSUs) along road corridor/grid | STAR-RIS reconfigurable surfaces + MEC servers | **INCOMPATIBLE** |
| **Continuous Control Variables** | **None** (Transmit power $P_v, P_R$, bandwidth $B$, and CPU $F_m$ are fixed environment constants) | Continuous phase-shift matrices $\mathbf{\Phi}_t, \mathbf{\Phi}_r$ and continuous power $\mathbf{p}$ | **INCOMPATIBLE** |
| **Execution Mechanics** | Action selects Standalone ($a=0$) or Collaborative RSU ($a=1..6$) | Multi-pass evaluates discrete server paired with continuous RIS phase vector | **INCOMPATIBLE** |

### Mathematical Collapse of MP-DQN in a Discrete Space:
In a purely discrete action space where continuous parameter vector $x_k = \emptyset$:
$$Q(s, k, x_k) \equiv Q(s, k)$$
The Multi-Pass mechanism becomes degenerate: evaluating $K$ forward passes over identical empty parameter vectors is mathematically identical to a single-pass standard DQN, while multiplying computational cost by $K$ without functional purpose.

---

## 5. Missing Adaptation Mechanisms in Du et al.

To deploy QRMP-DQN in a purely discrete environment, Du et al. would have had to invent an unpublished adaptation mechanism. Specifically, the following critical components are entirely missing:
1. **Parameterization Definition**: If dummy continuous variables were introduced (e.g., continuous task splitting fractions), no equations or bounds are provided in Du et al.
2. **Quantile Distribution Hyperparameters**: The number of quantiles $N$, the quantile thresholds $\tau_i = \frac{2i-1}{2N}$, and the Huber loss threshold $\kappa$ are completely omitted.
3. **Actor Network Architecture**: Layer dimensions, activations, and learning rates for the parameter-generating actor network $\mu(s; \theta_\mu)$ are omitted.
4. **Replay Buffer & Target Update Schedules**: Replay capacity, mini-batch size, exploration decay $\epsilon$, and target network synchronization frequency $C$ are unstated.

---

## 6. Repository Forensic Evidence

A forensic audit of the author's open-source release codebase (`bd34c65`) reveals:
1. **Zero QRMP-DQN Files**: There are no files named `qrmp*`, `quantile*`, `mp_dqn*`, or `distributional*` anywhere in the repository.
2. **Zero Code References**: Full-text regex search across all scripts, comments, docstrings, and configuration files yields 0 matches for `QRMP`.
3. **Missing Baseline Implementation**: While `a3c_agent.py` was provided for CoTOP, the author repository provided zero baseline implementation code for either DDQN or QRMP-DQN.

---

## 7. Why Generic QR-DQN is Scientifically Unacceptable

An engineering temptation when encountering this gap is to implement generic, single-pass **QR-DQN** (*Dabney et al., AAAI 2018*) on discrete actions and label the resulting curve "QRMP-DQN". 

We reject this substitution on strict scientific grounds:
1. **Misattribution of Reference [33]**: Generic QR-DQN is not QRMP-DQN. Reference [33] is defined by its Multi-Pass (MP) parameterization for STAR-RIS. Stripping the "MP" component destroys the specific algorithmic identity claimed in the paper.
2. **Methodological Pollution**: Passing off an ad-hoc QR-DQN surrogate as the author's baseline violates scientific reproduction ethics and obscures the literature gap.
3. **Unverifiable Hyperparameter Space**: Even generic QR-DQN requires unstated quantile numbers ($N \in \{50, 100, 200\}$) and Huber thresholds, creating arbitrary degrees of freedom.

---

## 8. Exact Experiments & Cells Where QRMP-DQN is N/A

QRMP-DQN is formally designated as **`N/A (EXCLUDED — REF [33] STAR-RIS DOMAIN MISMATCH)`** across the entire experimental matrix:

| Experiment Suite | Description | Total Cells | QRMP-DQN Status |
| :--- | :--- | :---: | :--- |
| **Stage 10: Primary Factorial Matrix** | 2 Geometries $\times$ 3 Workloads $\times$ 5 Seeds | 30 Cells | **`N/A (EXCLUDED)`** |
| **Stage 11: Table 4 & 5 Reproduction** | 2 Geometries $\times$ 3 Workloads $\times$ 5 Seeds | 30 Cells | **`N/A (EXCLUDED)`** |
| **Stage 13: Corrective Validations** | Multi-lane Highway Corridor | 5 Seeds | **`N/A (EXCLUDED)`** |
| **Secondary Ablation Matrix** | Dwell-Time & Weight Sweeps | All Cells | **`N/A (EXCLUDED)`** |

---

## 9. Scientific Consequence & Transparent Reporting

1. **Formal Literature Gap Finding**: The inability to reproduce QRMP-DQN is documented as a formal scientific finding: *Reference [33] was cited as a discrete baseline despite belonging to a continuous STAR-RIS domain, without disclosing the domain adaptation mapping.*
2. **Matrix Integrity Preserved**: The primary factorial matrix is locked to the 60 trained replications of **CoTOP vs. DDQN** alongside 60 deterministic baseline replications of **Greedy and Local** (120 total audited evaluations).
3. **Transparent Manuscript Presentation**: All manuscript tables (Table IV, Table V, Table VI) and text explicitly label QRMP-DQN as `N/A (EXCLUDED — REF [33] STAR-RIS DOMAIN MISMATCH)`. **QRMP-DQN is never silently omitted.**
