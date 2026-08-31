# PHASE 2: QRMP-DQN SCIENTIFIC DISPOSITION AND FORMAL EXCLUSION RECORD

**Document ID**: `DOC-PHASE2-DISPOSITION-QRMP-DQN-001`  
**Classification**: Formal Methodological Exclusion Record  
**Target Paper**: *Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing* (Du et al., IEEE TMC 2026)  
**Cited Baseline Reference**: Reference [33] (L. Guo, J. Jia, J. Chen, and X. Wang, *"QRMP-DQN Empowered Task Offloading and Resource Allocation for the STAR-RIS Assisted MEC Systems"*)  

---

## 1. Final Decision

> ### **FINAL SCIENTIFIC DISPOSITION: SCIENTIFICALLY UNRESOLVED / EXCLUDED**
> **QRMP-DQN is formally excluded from the primary factorial experimental matrix.**  
> It is explicitly labeled as `N/A (EXCLUDED — REF [33] STAR-RIS DOMAIN MISMATCH)` across all comparative tables and narrative summaries. It is **never** silently omitted, nor is an ad-hoc ungrounded surrogate (such as generic discrete QR-DQN) substituted in its place.

---

## 2. Reference [33] Mathematical Structure

Reference [33] (Guo et al.) was developed specifically for **Simultaneously Transmitting and Reflecting Reconfigurable Intelligent Surfaces (STAR-RIS)** assisted Multi-Access Edge Computing (MEC) networks.

### A. Parameterized Action Space Markov Decision Process (PAMDP)
In Reference [33], the decision space is fundamentally hybrid (parameterized discrete-continuous):
$$a = (d, \mathbf{x}_d) \in \mathcal{D} \times \mathcal{X}_d$$
where:
1. $d \in \{1, \dots, D\}$ is a discrete decision representing the index of the MEC server selected for task offloading.
2. $\mathbf{x}_d = (\mathbf{\Phi}_t, \mathbf{\Phi}_r, \mathbf{p}) \in \mathbb{C}^{M \times M} \times \mathbb{C}^{M \times M} \times \mathbb{R}^K$ is a continuous parameter vector containing:
   - $\mathbf{\Phi}_t = \text{diag}(e^{j \theta_1^t}, \dots, e^{j \theta_M^t})$: Continuous transmission phase-shift matrix of the STAR-RIS elements,
   - $\mathbf{\Phi}_r = \text{diag}(e^{j \theta_1^r}, \dots, e^{j \theta_M^r})$: Continuous reflection phase-shift matrix of the STAR-RIS elements,
   - $\mathbf{p} = (p_1, \dots, p_K)$: Continuous user transmit power allocation vector.

### B. Multi-Pass Deep Q-Network (MP-DQN) Mechanics
To optimize PAMDPs, MP-DQN (*Bester et al., AAAI 2019*) employs an actor network $\mu(s; \theta_\mu)$ that outputs continuous parameter vectors $\mathbf{x}_d = \mu_d(s)$ for all $D$ discrete actions simultaneously. The critic network $Q(s, d, \mathbf{x}_d; \theta_Q)$ performs $D$ separate forward passes ("multi-pass") to evaluate each $(d, \mathbf{x}_d)$ pair, selecting:
$$d^* = \arg\max_{d \in \mathcal{D}} Q(s, d, \mathbf{x}_d; \theta_Q)$$

### C. Distributional Quantile Regression Extension
QRMP-DQN replaces the scalar Q-value with a probability distribution over returns modeled by $N$ uniform Dirac quantiles $\theta_i(s, d, \mathbf{x}_d)$ trained with the quantile Huber loss (*Dabney et al., AAAI 2018*).

---

## 3. Target Paper Action Structure (Du et al. 2026)

In Du et al., the edge offloading problem is defined with a purely discrete action space:
$$\mathcal{A} = \{0, 1, 2, \dots, M\}$$
where:
- $a = 0$: Standalone offloading to the primary nearest RSU (Case 1).
- $a = m \in \{1, \dots, M\}$: Collaborative offloading where the primary RSU executes a portion of the task and forwards the remaining subtask to collaborative RSU $m$ (Case 2).

All other physical parameters in the target environment are fixed constants:
- Vehicle transmit power: Fixed scalar $P_v = 1.0\text{ W}$ (Table III)
- RSU transmit power: Fixed scalar $P_R = 10.0\text{ W}$ (Table III)
- Channel bandwidth: Fixed scalar $B = 10\text{ MHz}$ (Table III)
- RSU compute frequency: Fixed scalar $F_m = 4\text{ GHz}$ (Table III)
- STAR-RIS elements: **None** (STAR-RIS does not exist in Du et al.)

---

## 4. Fundamental Mathematical Incompatibilities

| Evaluation Dimension | Reference [33] (Guo et al.) | Target Paper (Du et al. 2026) | Compatibility Status |
| :--- | :--- | :--- | :--- |
| **Action Space Structure** | Parameterized PAMDP $(d, \mathbf{x}_d)$ | Purely Discrete $\mathcal{A} = \{0, 1, \dots, 6\}$ | **FATAL MISMATCH** |
| **Continuous Control Variables** | Continuous phase shifts $\mathbf{\Phi}_t, \mathbf{\Phi}_r$ & continuous power $\mathbf{p}$ | **Zero continuous control variables** | **FATAL MISMATCH** |
| **Physical System Model** | STAR-RIS surface reflection/transmission | Terrestrial V2R and R2R wireless links | **FATAL MISMATCH** |
| **MP Mechanism Behavior** | Evaluates continuous parameter candidates $\mathbf{x}_d$ across $D$ passes | Degenerates to single-pass discrete evaluation since $\mathbf{x}_d \equiv \emptyset$ | **DEGENERATE** |

### Mathematical Degeneracy Proof:
When the continuous parameter set is empty ($\mathbf{x}_d = \emptyset$):
$$Q(s, d, \mathbf{x}_d) \equiv Q(s, d)$$
The multi-pass architecture evaluates $D$ passes over identical empty inputs, rendering the "Multi-Pass" actor-critic structure redundant and mathematically identical to standard single-pass Q-learning.

---

## 5. Missing Domain Adaptation Information

In order for Reference [33] to be applied to the target paper's environment, the authors of Du et al. would have had to invent an unpublished domain adaptation mapping. However, the manuscript and open-source repository provide:

1. **Zero Equations**: No mathematical formulation mapping STAR-RIS phase angles to VEC vehicular variables.
2. **Zero Architecture Specifications**: No layer widths, depths, or activation functions for the continuous parameter actor $\mu(s; \theta_\mu)$.
3. **Zero Quantile Parameters**: No specification of quantile count $N$, quantile fractions $\tau_i$, or Huber threshold $\kappa$.
4. **Zero Repository Artifacts**: The authoritative codebase (`adem-mekonnen/cotop-implementation`) contains zero files, functions, or references matching `qrmp`, `mp_dqn`, `quantile`, or `distributional`.

---

## 6. Scientific Rules Adherence & Integrity Verification

To protect scientific validity, this investigation strictly adhered to the following rules:

1. **Rule 1 (No Pseudo-QRMP-DQN)**: We reject implementing standard single-pass discrete QR-DQN (*Dabney et al.*) and mislabeling it "QRMP-DQN". Stripping the "Multi-Pass" parameterization destroys the algorithmic identity of Reference [33].
2. **Rule 2 (No Invented Continuous Variables)**: We refuse to invent ungrounded continuous control variables (e.g., continuous task splitting fractions) that do not exist in Du et al.'s model.
3. **Rule 3 (No Invented Adaptation Equations)**: We do not formulate speculative bridge equations not disclosed in the published literature.
4. **Rule 4 (No Variable Reinterpretation)**: We do not treat STAR-RIS phase shift matrices as surrogate VEC variables.
5. **Rule 5 (No Environment Mutation)**: We preserve the target VEC environment without modifying its action space to fit Reference [33].

---

## 7. Reproducibility Consequences

1. **Literature Citation Discrepancy Identified**: Reference [33] is a domain-mismatched citation in Du et al. (citing a hybrid-action STAR-RIS algorithm as a baseline for a discrete vehicular offloading task without specifying the transformation).
2. **Transparent Reporting in Primary Matrix**: Rather than silently ignoring the baseline or publishing an ungrounded surrogate, QRMP-DQN is marked `N/A (EXCLUDED — REF [33] STAR-RIS DOMAIN MISMATCH)` in all comparison tables.
3. **Primary Experimental Matrix Unaffected**: The primary factorial comparison is rigorously established on the fully reproducible, mathematically faithful baselines: **CoTOP vs. DDQN vs. Greedy vs. Local** ($120$ audited evaluation conditions).
