# Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing (CoTOP)

An independent, reproduction-grade scientific replication of the IEEE Transactions on Mobile Computing (TMC 2026) paper:

> **"Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing"**  
> *Jiaxin Du, Jinfan Zhang, Guangjie Han, Mengmeng Wang, Guojiang Shen, Zhi Liu, and Xiangjie Kong*  
> IEEE TMC, Vol. 25, No. 4, April 2026. DOI: [10.1109/TMC.2025.3631820](https://doi.org/10.1109/TMC.2025.3631820)

---

## 1. Scientific Reproduction Verdict

```
Mathematical Fidelity: PASS (0.00% Analytical Deviation)
Implementation Integrity: PASS (100% Immutability Preserved)
Unit Tests: PASS (22/22 Tests Passing)
A3C Convergence: PASS (Asymptotic Stability by Epoch 35-40)
Multi-Seed Stability: PASS (Evaluated across Seeds 42, 123, 456, 789, 2026)
Baseline Comparison: PASS (Fully Paired N=250 Test Episodes)
Statistical Validation: PASS (Paired t-tests, Wilcoxon, Cohen's dz, Holm/FDR)
Published 13.90 s Reproduction: NOT REPRODUCED (Clean channel physics yields 4.40s)
Published 25.14 J Reproduction: NOT REPRODUCED (Single-task physics yields 0.32J)
ApolloScape Dataset Reproduction: NOT ACHIEVED (Synthetic Kinematics Fallback)
Queue Explanation: PLAUSIBLE / UNCONFIRMED (Post-Hoc Diagnostic)
Energy Scope Explanation: PLAUSIBLE / UNCONFIRMED (Post-Hoc Diagnostic)
Overall Reproduction Class: CLASS B — METHOD-LEVEL REPRODUCTION
```

---

## 2. Core System Architecture

**CoTOP** combines spatiotemporal trajectory prediction with multi-agent reinforcement learning:
1. **Spatiotemporal Mobility Prediction (GAT-GRU)**: 4-head Graph Attention Network with GRU temporal units predicting vehicle dwell time $T^{\text{stay}}$ within RSU wireless coverage (Eq. 15–22, Table II).
2. **Task Prioritization**: Prioritizes parallel DAG subtasks using dwell time, data size, and deadline urgency: $P_i = \alpha e^{-1/T^{\text{stay}}} + \beta \frac{\rho_i}{d_i}$ (Eq. 23).
3. **Collaborative Offloading (DRL / A3C)**: Adaptively selects between Standalone execution on the serving RSU (Case 1) and Inter-RSU Collaborative processing (Case 2) using an Asynchronous Advantage Actor-Critic algorithm (Algorithm 1).

```
   [Vehicle]  -- (V2R Upload) --> [Primary RSU]
                                        |
                            Is Dwell Time Exceeded?
                           /                       \
                     [No: Case 1]              [Yes: Case 2]
                     (Standalone)             (Collaborative)
                          |                          |
                     Compute Local             Relay remaining task
                                               to Secondary RSU via R2R
```

---

## 3. Final Performance Summary ($N=250$ Paired Evaluation Episodes)

Evaluated under strict Table III parameters (6 RSUs, 400m spacing, 10–30 vehicles at 30–40 m/s, 2–5 MB tasks, 10 Mcycles demand):

| Method | Mean Delay (s) | Delay $95\%\text{ CI}$ | Mean Energy (J) | Energy $95\%\text{ CI}$ | Completion | Collab Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Local Baseline** | $4.425 \pm 0.023\text{ s}$ | $[4.397, 4.453]$ | $0.320 \pm 0.005\text{ J}$ | $[0.314, 0.326]$ | $100.00\%$ | $0.00\%$ |
| **CoTOP (Proposed)** | $4.402 \pm 0.060\text{ s}$ | $[4.327, 4.477]$ | $0.319 \pm 0.005\text{ J}$ | $[0.313, 0.325]$ | $100.00\%$ | $0.40\%$ |
| **Greedy Baseline** | $4.393 \pm 0.050\text{ s}$ | $[4.331, 4.455]$ | $4.525 \pm 0.068\text{ J}$ | $[4.441, 4.609]$ | $100.00\%$ | $95.00\%$ |

### Key Statistical Conclusions:
- **CoTOP vs Local**: No statistically significant difference detected under clean channel conditions ($t(249) = -1.542, p = 0.1244$). Both rationally execute Standalone offloading.
- **CoTOP vs Greedy**: Statistically significant **92.95% energy reduction** ($p < 10^{-4}$ after Holm and FDR adjustments, paired Cohen's $d_z = -62.40$, CLES $= 100.0\%$). Greedy incurs severe penalties from $100\text{ W}$ inter-RSU relay links.

---

## 4. Repository Structure

```
cotop-implementation/
├── configs/
│   └── paper_parameters.yaml          # Strict Table III physical parameters
├── docs/
│   ├── FINAL_REPRODUCTION_REPORT.md   # Comprehensive 16-section final scientific report
│   ├── REPRODUCTION_PROTOCOL.md       # Step-by-step reproduction guide
│   ├── STATISTICAL_METHODS.md         # Statistical methodologies and multiple testing
│   ├── LIMITATIONS_AND_THREATS.md     # Threats to validity and boundary conditions
│   └── CLAIM_EVIDENCE_MATRIX.md       # Line-by-line verification of Claims A through G
├── envs/
│   ├── comm_model.py                  # Eq. 1 (V2R) & Eq. 2 (R2R) Shannon capacities
│   ├── comp_model.py                  # Eq. 3-10 (Delays) & Eq. 11-12 (Energy)
│   ├── entities.py                    # Dataclasses: Vehicle, Task, RSU, Config
│   ├── state_builder.py               # Eq. 24 41-dim normalized state vector
│   └── vec_env.py                     # Gymnasium environment coordinating SUMO
├── figures/
│   └── final/                         # 7 publication-ready visualization figures
├── models/
│   ├── a3c_agent.py                   # Actor-Critic network architecture
│   ├── mobility_gat.py                # 4-head GAT-GRU mobility model (Table II)
│   └── baselines/                     # Local (standalone) & Greedy (min-queue)
├── notebooks/
│   └── CoTOP_Stage11_Colab_Reproduction.ipynb  # End-to-end reproducible Colab notebook
├── results/
│   └── final/                         # 8 final publication-ready CSV tables
├── tests/                             # 22 automated pytest unit tests
├── sanity_check.py                    # Analytical hand-calculation verifier (0.00% error)
├── train.py                           # Multiprocessing SharedAdam A3C trainer
├── evaluate.py                        # Dynamic checkpoint evaluator with CSV logging
└── requirements.txt                   # Dependency specifications
```

---

## 5. Quickstart & Execution Guide

### Prerequisites
- Python 3.10+
- PyTorch 2.4.1+
- Eclipse SUMO 1.25.0+

### Installation
```bash
git clone https://github.com/adem-mekonnen/cotop-implementation.git
cd cotop-implementation
pip install -r requirements.txt
```

### 1. Analytical Sanity Check (Layer 1 Verification)
```bash
python sanity_check.py
```
*Passes with 0.00% analytical deviation across all 16 governing equations.*

### 2. Automated Test Suite (Layer 2 Verification)
```bash
pytest -q
```
*Passes 22/22 unit tests in ~5.2s.*

### 3. Model Training (A3C Across 5 Seeds)
```bash
python train.py --config configs/paper_parameters.yaml --episodes 500 --seed 42 --save_dir results/checkpoints/42
```

### 4. Controlled Evaluation
```bash
python evaluate.py --mode cotop --checkpoint_path results/checkpoints/42/a3c_agent.pth --episodes 50 --seed 42 --config configs/paper_parameters.yaml
python evaluate.py --mode local --episodes 50 --seed 42 --config configs/paper_parameters.yaml
python evaluate.py --mode greedy --episodes 50 --seed 42 --config configs/paper_parameters.yaml
```

### 5. Automated Generation of Publication Package
```bash
python experiments/stage17_final_controlled_reproduction.py
python experiments/stage18_package_final_results.py
```

---

## 6. Scientific Documentation Suite

For complete in-depth scientific audits and derivations, consult the documentation in `docs/`:
- **[Final Reproduction Report](docs/FINAL_REPRODUCTION_REPORT.md)**: Full 16-section publication-ready assessment.
- **[Reproduction Protocol](docs/REPRODUCTION_PROTOCOL.md)**: Exact reproduction steps and Colab execution guide.
- **[Statistical Methods](docs/STATISTICAL_METHODS.md)**: Paired t-tests, Wilcoxon signed-rank tests, Cohen's $d_z$, CLES, and FDR corrections.
- **[Limitations & Threats](docs/LIMITATIONS_AND_THREATS.md)**: Metric scope ambiguity and undisclosed protocol parameters.
- **[Claim-to-Evidence Matrix](docs/CLAIM_EVIDENCE_MATRIX.md)**: Direct mapping from primary claims (A–G) to verified code and data.

---

## 7. Citation & Provenance

```bibtex
@article{du2026mobility,
  title={Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing},
  author={Du, Jiaxin and Zhang, Jinfan and Han, Guangjie and Wang, Mengmeng and Shen, Guojiang and Liu, Zhi and Kong, Xiangjie},
  journal={IEEE Transactions on Mobile Computing},
  volume={25},
  number={4},
  pages={5540--5555},
  year={2026},
  publisher={IEEE},
  doi={10.1109/TMC.2025.3631820}
}
```
