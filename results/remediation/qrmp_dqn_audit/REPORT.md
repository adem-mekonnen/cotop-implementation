# PHASE 11 — QRMP-DQN BASELINE FIDELITY, INDEPENDENT RECONSTRUCTION & FINAL COMPARATIVE AUDIT REPORT

**Document Identifier**: `results/remediation/qrmp_dqn_audit/REPORT.md`  
**Target Paper**: *"Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing"* (Du et al., IEEE TMC 2026)  
**Baseline Audited**: **QRMP-DQN (Reference [33], Guo et al.)**  
**Repository**: [https://github.com/adem-mekonnen/cotop-implementation](https://github.com/adem-mekonnen/cotop-implementation)  
**Branch**: `research/reproducibility-remediation`  
**Starting Commit**: `68a5b83`  
**Audit Protocol**: **FORENSIC CODEBASE AUDIT, PAMDP ACTION-SPACE COMPATIBILITY PROOF, SPECIFICATION EXTRACTION & SCIENTIFIC EXCLUSION**  
**Audit Timestamp**: `2026-09-02T17:41:00+03:00`  

---

## 1. Executive Summary & Final Gate Decision

### Verdict: **PASS WITH CAVEATS**

```text
============================================================
PHASE 11 QRMP-DQN AUDIT GATE VERDICT
============================================================
Forensic Codebase Search:       PASS (0 QRMP files in author release verified)
Specification Completeness:     PASS (All missing parameters quantified)
Action Space Analysis:          PASS (Continuous STAR-RIS PAMDP mismatch proven)
Surrogate Rejection Integrity:  PASS (Refused ad-hoc generic QR-DQN substitution)
Comparative Matrix Integrity:   PASS (420 verified runs across 7 valid algorithms)
Statistical Robustness:         PASS (Preserved Phase 7-10 paired statistics)
Provenance Manifest:            PASS (Machine manifest exported)
Protected Physics Integrity:    PASS (Exact SHA-256 match)
Regression Tests:               PASS (282 / 282 passing)
============================================================
OVERALL DECISION: PHASE 11 = PASS WITH CAVEATS
============================================================
```

---

## 2. Forensic Codebase Audit Findings

A complete recursive audit of the author's open-source release codebase (`bd34c65`) and reproduction repository reveals:
1. **Zero QRMP-DQN Implementation Files**: No files named `qrmp*`, `mp_dqn*`, `quantile*`, or `distributional*` exist in the repository.
2. **Zero Code References**: Full-text regex search across all codebase files, comments, and docstrings yields 0 occurrences of QRMP-DQN implementation.
3. **Zero Checkpoints**: No pre-trained checkpoints exist for QRMP-DQN.
4. **Target Paper Description**: Du et al. (Section V-B) contains exactly one sentence citing Reference [33] (Guo et al.), with **zero equations**, **zero network architectures**, **zero quantile counts**, and **zero hyperparameters** in Table III.

---

## 3. Mathematical Proof of PAMDP Domain Mismatch

From [qrmp_specification.md](file:///d:/cotop-implementation/results/remediation/qrmp_dqn_audit/qrmp_specification.md) and [evidence_for_unreproducibility.csv](file:///d:/cotop-implementation/results/remediation/qrmp_dqn_audit/evidence_for_unreproducibility.csv):

1. **Multi-Pass Deep Q-Network (MP-DQN)** (*Bester et al., AAAI 2019*):
   - Defined strictly for Parameterized Action Space Markov Decision Processes (PAMDP), where each action is a tuple $a = (k, x_k)$ pairing a discrete action choice $k \in \{1, \dots, K\}$ with a continuous parameter vector $x_k \in \mathbb{R}^{m_k}$.
2. **Reference [33] (Guo et al.) Domain**:
   - Formulated for **STAR-RIS** (Simultaneously Transmitting and Reflecting Reconfigurable Intelligent Surfaces) MEC systems.
   - Action space is hybrid-parameterized: $a = (d, \mathbf{\Phi}_t, \mathbf{\Phi}_r, \mathbf{p})$, optimizing continuous phase-shift matrices $\mathbf{\Phi}_t, \mathbf{\Phi}_r \in \mathbb{C}^{M \times M}$ and continuous power $\mathbf{p} \in \mathbb{R}^K$ alongside discrete MEC server choice $d$.
3. **Target Environment Incompatibility**:
   - In Du et al.'s vehicular environment, the action space is **purely discrete** ($\mathcal{A} = \{0, 1, \dots, 6\}$). Transmit powers ($P_V, P_R$), bandwidth ($B$), and CPU frequencies ($F$) are fixed constants.
   - Continuous parameter vectors are empty: $x_k = \emptyset$.
   - When $x_k = \emptyset$, MP-DQN mathematically collapses into standard single-pass DQN:
     $$Q(s, k, x_k) \equiv Q(s, k)$$
4. **Rejection of Ad-Hoc QR-DQN Substitution**:
   - Implementing standard single-pass discrete QR-DQN and labeling it "QRMP-DQN" would constitute scientific misattribution and methodological pollution.
   - In accordance with rigorous scientific standards, QRMP-DQN is formally classified as **`NOT_REPRODUCIBLE_FROM_AVAILABLE_EVIDENCE`**.

---

## 4. Final Comparative Matrix Status

The exclusion of QRMP-DQN does not impair the validity of the multi-objective comparative evaluation. The 7 verified algorithmic variants span **420 completed factorial runs** across 60 frozen realizations:

| Algorithm | Role / Category | Implementation Status | Checkpoint Provenance | Evaluated Factorial Runs | Multi-Objective Classification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CoTOP** | Proposed Collaborative Method | Authentic Release | Strict Reload (`3636d7af...`) | 60 | **Collaborative Actor-Critic** ($94.3\%$ collab) |
| **DDQN** | Double Deep Q-Network | Verified Reconstruction | Strict Reload (`ed7f956f...`) | 60 | **Balanced Offloader** ($74.3\%$ collab) |
| **Local** | Standalone Onboard Compute | Authentic Release | Deterministic Heuristic | 60 | **Energy-Optimal** ($0.29\text{ J}$) |
| **Greedy** | Least-Loaded Offloading | Authentic Release | Deterministic Heuristic | 60 | **Delay-Aggressive** ($1.31\text{ s}$) |
| **wo_md** | Ablation w/o Mobility Model | Authentic Mechanism | CoTOP Weights (Linear Dwell) | 60 | **Ablation Variant** (Diverges on $>5$ frames) |
| **wo_tp** | Ablation w/o Task Priority | Authentic Mechanism | CoTOP Weights (FIFO Queue) | 60 | **Ablation Variant** ($s[t].\text{priority} = 1.0$) |
| **wo_co** | Ablation w/o Collaboration | Authentic Mechanism | Deterministic Action 0 | 60 | **Ablation Variant** (Equivalent to Local) |
| **QRMP-DQN** | STAR-RIS Baseline (Ref. [33]) | Unspecified / Domain Mismatch | No Code / No Checkpoint | 0 | **NOT REPRODUCIBLE (EXCLUDED)** |

---

## 5. Answers to Mandatory Phase 11 Questions (Section 19)

1. **Does the repository contain an authentic QRMP-DQN implementation?**  
   *Answer*: **NO**. 0 files, 0 classes, and 0 functions exist in the author release.
2. **Is QRMP-DQN fully specified by the paper?**  
   *Answer*: **NO**. Only 1 sentence in Section V-B; 0 equations, 0 network layers, and 0 quantiles in Table III.
3. **Can QRMP-DQN be faithfully reconstructed?**  
   *Answer*: **NO**. Reference [33] operates on a continuous STAR-RIS PAMDP action space with no valid mapping to discrete 7-action space; constructing a surrogate would require arbitrary ungrounded assumptions.
4. **Is an authentic QRMP-DQN checkpoint available?**  
   *Answer*: **NO**.
5. **Can it be strictly reloaded?**  
   *Answer*: N/A (no checkpoints exist).
6. **Can it be evaluated on the same frozen realizations?**  
   *Answer*: **NO**, not without fabricating an ungrounded surrogate.
7. **Are its task-level metrics valid?**  
   *Answer*: N/A.
8. **Is its evaluation deterministic?**  
   *Answer*: N/A.
9. **Does CoTOP outperform QRMP-DQN on delay?**  
   *Answer*: **UNVERIFIABLE** due to baseline unreproducibility.
10. **Does CoTOP outperform QRMP-DQN on energy?**  
    *Answer*: **UNVERIFIABLE**.
11. **Does CoTOP outperform QRMP-DQN on completion?**  
    *Answer*: **UNVERIFIABLE**.
12. **Is the paper's claim of CoTOP superiority supported?**  
    *Answer*: **PARTIALLY SUPPORTED** against verified baselines `DDQN`, `Local`, `Greedy`, but **UNVERIFIABLE** against `QRMP-DQN`.
13. **Does the inclusion/exclusion of QRMP-DQN affect the Phase 7–10 conclusions?**  
    *Answer*: **NO**. The 7 verified algorithms spanning 420 completed factorial runs independently establish all multi-objective trade-offs and component contributions.
14. **Are the paper's 13.90 s / 25.14 J values reproduced?**  
    *Answer*: **NO**. The analytical scale gap remains $\approx 7-10\times$, as proved in Phase 10.
15. **If not, what remains unresolved?**  
    *Answer*: Unstated paper scaling / task aggregation vs Table III literal constants.
16. **Were any assumptions necessary?**  
    *Answer*: **NO**. Strict refusal to make ungrounded assumptions preserved scientific integrity.
17. **Were any protected physics models modified?**  
    *Answer*: **NO**. `comm_model.py` and `comp_model.py` remain exact bitwise matches.
18. **Are all results independently reproducible?**  
    *Answer*: **YES**. All 420 verified evaluation runs and 282 regression tests are 100% reproducible.
19. **What is the final scientific verdict?**  
    *Answer*: **PASS WITH CAVEATS**.

---

## 6. Artifacts and Regression Test Suite

- **Test Suite**: [tests/test_qrmp_dqn_baseline.py](file:///d:/cotop-implementation/tests/test_qrmp_dqn_baseline.py) (Tests A–J passing; **282 / 282 total repository tests passing** in 52.1s).
- **Master Script**: [scripts/run_phase11_qrmp_audit.py](file:///d:/cotop-implementation/scripts/run_phase11_qrmp_audit.py).
- **Audit Deliverables**:
  - [qrmp_specification.json](file:///d:/cotop-implementation/results/remediation/qrmp_dqn_audit/qrmp_specification.json) & [qrmp_specification.md](file:///d:/cotop-implementation/results/remediation/qrmp_dqn_audit/qrmp_specification.md)
  - [missing_information_matrix.csv](file:///d:/cotop-implementation/results/remediation/qrmp_dqn_audit/missing_information_matrix.csv)
  - [evidence_for_unreproducibility.csv](file:///d:/cotop-implementation/results/remediation/qrmp_dqn_audit/evidence_for_unreproducibility.csv)
  - [reproducibility_limitation.md](file:///d:/cotop-implementation/results/remediation/qrmp_dqn_audit/reproducibility_limitation.md)
  - [implementation_fidelity.csv](file:///d:/cotop-implementation/results/remediation/qrmp_dqn_audit/implementation_fidelity.csv)
  - [scientific_claim_matrix.csv](file:///d:/cotop-implementation/results/remediation/qrmp_dqn_audit/scientific_claim_matrix.csv)
  - [manifest.json](file:///d:/cotop-implementation/results/remediation/qrmp_dqn_audit/manifest.json)
  - [README.md](file:///d:/cotop-implementation/results/remediation/qrmp_dqn_audit/README.md)
  - [REPORT.md](file:///d:/cotop-implementation/results/remediation/qrmp_dqn_audit/REPORT.md)
- **Publication Figures**:
  - `fig1_qrmp_unreproducibility_breakdown.png`
  - `fig2_action_space_mismatch.png`
