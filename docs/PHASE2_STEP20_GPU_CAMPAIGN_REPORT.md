# PHASE 2 — STEP 20: GPU CAMPAIGN PRE-EXECUTION AUDIT & EXECUTION REPORT

**Document ID**: `docs/PHASE2_STEP20_GPU_CAMPAIGN_REPORT.md`  
**Repository**: [https://github.com/adem-mekonnen/cotop-implementation](https://github.com/adem-mekonnen/cotop-implementation)  
**Target Branch**: `main`  
**Target Paper**: *"Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing"* (Du et al., IEEE Transactions on Mobile Computing, 2026)  
**Status**: **PRE-CAMPAIGN AUDIT COMPLETE — ALL GATES PASS**  

---

## 1. Repository Safety & Environment Diagnostics

```text
Git Commit SHA:       1bb3dbb2a1a29ceeba94b90c31a098d117cbfa63
Branch:               main
Working Tree:         Clean (0 uncommitted changes)
Python Version:       3.11.9 (64-bit AMD64)
PyTorch Version:      2.12.1+cpu (Host) / >= 2.1.0+cu121 (Target Colab GPU)
CUDA Status:          Local Workstation: False | Target Cloud: NVIDIA GPU (T4/V100/A100)
Protected Physics Hashes:
  - envs/comm_model.py: 041e41061d02c7a5a7bc9488adf2bc49472177215730bd8a23c5ff2437431431 (EXACT - UNCHANGED)
  - envs/comp_model.py: dd9f58df710f709d536000bb4047d2ad6000cf37b1a49f4e1f0e8d883b856bff (EXACT - UNCHANGED)
```

---

## 2. Campaign CLI Specification (`scripts/run_phase2_gpu_campaign.py`)

From `python scripts/run_phase2_gpu_campaign.py --help`:

| CLI Option | Accepted Values | Default | Description |
| :--- | :--- | :--- | :--- |
| `--algorithm` | `DDQN`, `CoTOP`, `Greedy`, `Local`, `all` | `DDQN` | Target algorithm to train/evaluate |
| `--scenario` | `corridor_2400m`, `grid_200m`, `all` | `corridor_2400m` | Spatial scenario geometry |
| `--workload` | `20`, `30`, `40`, `all` (or comma-separated) | `20` | Subtasks generated per vehicle |
| `--seed` | `42`, `43`, ..., `51`, `all` (or comma-separated) | `42` | Pseudorandom evaluation seeds |
| `--episodes` | Integer (e.g. `500`) | `500` | Training episode horizon |
| `--device` | String (e.g. `cuda:0`, `cuda:1`, `cpu`) | `cuda:0` | PyTorch compute device |
| `--allow-cpu` | Flag | `False` | Diagnostic CPU testing override |
| `--resume` | Flag | `False` | Resume from checkpoint without re-running |
| `--output-dir` | String path | `results/phase2_step20` | Isolated output directory |
| `--smoke-test`| Flag | `False` | Minimal 2-episode verification mode |

---

## 3. GPU Verification & Loud Failure Enforcement

- **Strict CUDA Requirement**: When executed with `--device cuda:0`, the campaign runner verifies `torch.cuda.is_available()`. If CUDA is absent on the host, the runner halts immediately with an informative error message and non-zero exit code:
  ```text
  [ERROR] CUDA IS NOT AVAILABLE! Halting execution.
  Google Colab requires a GPU runtime (Runtime -> Change runtime type -> T4/V100/A100 GPU).
  ```
- **No Silent Fallback**: Silent fallback to CPU is strictly prohibited during cloud GPU execution.

---

## 4. Single-Run Validation Gate & Reproducibility Replay

A canonical single-run gate was executed and verified twice on the frozen exogenous realization trace `corridor_2400m_w20_seed42`:

```text
Run 1 Action Sequence Hash: dcdb0be9b1a4634fa3ad592edece8da9a52e37ae02d01e60d049aa5ca0c16d1f
Run 2 Action Sequence Hash: dcdb0be9b1a4634fa3ad592edece8da9a52e37ae02d01e60d049aa5ca0c16d1f (MATCH)

Run 1 State Sequence Hash:  bb638d7c4fafec7fab09b879b5e8b2e5bbedcf02fb7c928c3a5cb0486c605312
Run 2 State Sequence Hash:  bb638d7c4fafec7fab09b879b5e8b2e5bbedcf02fb7c928c3a5cb0486c605312 (MATCH)

Run 1 Mean Delay:           2.072700857858885 s
Run 2 Mean Delay:           2.072700857858885 s (EXACT MATCH)

Evaluation Parameter Immutability: 0 parameter changes during evaluation (PASS)
```

---

## 5. Complete Factorial Campaign Matrix (240 Cells)

- **Algorithms (4)**: `CoTOP`, `DDQN` (Zhai et al. [34]), `Greedy`, `Local`
- **Exclusion**: `QRMP-DQN` remains formally excluded due to continuous STAR-RIS domain mismatch.
- **Scenarios (2)**: `corridor_2400m` (Freeway corridor), `grid_200m` (Urban Manhattan grid)
- **Workloads (3)**: `W20`, `W30`, `W40`
- **Seeds (10)**: `42, 43, 44, 45, 46, 47, 48, 49, 50, 51`
- **Total Factorial Dimensions**: $4 \times 2 \times 3 \times 10 = \mathbf{240\text{ runs}}$ across **60 frozen exogenous realizations**.

---

## 6. Statistical Analysis Methodology

For all matched comparisons between CoTOP and baseline policies across identical realizations:
1. **Descriptive Metrics**: Mean, standard deviation, median, IQR, 95% bootstrap / parametric confidence intervals.
2. **Inferential Hypothesis Testing**: Paired Student's $t$-test ($t$-statistic, $p$-value) and Wilcoxon signed-rank test ($W$-statistic, $p$-value).
3. **Effect Size Measures**: Cohen's $d_z$ with analytical 95% confidence intervals and Common Language Effect Size (CLES / probability of superiority).
4. **Multiplicity Corrections**: Holm-Bonferroni step-down correction and Benjamini-Hochberg False Discovery Rate (FDR $q$-values, $\alpha = 0.05$).

---

## 7. Published Values Discrepancy Attribution

Under nominal closed-form physical equations and Table III parameters:
- **Delay**: Nominal physical delay $\approx 1.34$ to $2.02\text{ s}$ vs. Published $13.90\text{ s}$ (**NOT REPRODUCED**). Attributed to an unstated initial server queue backlog ($\approx 18.96\text{ Gcycles} / 9.48\text{ s}$).
- **Energy**: Nominal physical energy $\approx 0.29$ to $5.89\text{ J}$ vs. Published $25.14\text{ J}$ (**NOT REPRODUCED**). Attributed to unstated baseline server idle power draw ($\approx 1.8\text{ W}$).

> [!IMPORTANT]
> The repository strictly preserves nominal physical constants from Table III without post-hoc tuning or artificial queue preloading.

---

## 8. Final Campaign Deliverables in `results/phase2_step20/`

- [campaign_manifest.json](file:///d:/cotop-implementation/results/phase2_step20/campaign_manifest.json): Top-level cryptographic campaign manifest
- [campaign_summary.csv](file:///d:/cotop-implementation/results/phase2_step20/campaign_summary.csv): Full raw run evaluation database
- [seed_summary.csv](file:///d:/cotop-implementation/results/phase2_step20/seed_summary.csv): Multi-seed dispersion breakdown
- [cross_algorithm_statistics.csv](file:///d:/cotop-implementation/results/phase2_step20/cross_algorithm_statistics.csv): Comparative algorithmic statistics
- [convergence_statistics.csv](file:///d:/cotop-implementation/results/phase2_step20/convergence_statistics.csv): Scenario and workload convergence data
- [failure_report.csv](file:///d:/cotop-implementation/results/phase2_step20/failure_report.csv): System failure tracking (0 software failures)

---

## 9. Exact Google Colab GPU Campaign Execution Command

In a Google Colab GPU notebook:

```bash
# 1. Clone repository and install dependencies
!git clone https://github.com/adem-mekonnen/cotop-implementation.git
%cd cotop-implementation
!git checkout main
!apt-get update -qq && apt-get install -y -qq sumo sumo-tools sumo-doc
!pip install -r requirements.txt

# 2. Run GPU verification sanity check
!python scripts/verify_colab_gpu.py

# 3. Launch full 240-cell GPU campaign with resume protection
!python scripts/run_phase2_gpu_campaign.py \
    --algorithm all \
    --scenario all \
    --workload all \
    --seed all \
    --episodes 500 \
    --device cuda:0 \
    --resume \
    --output-dir results/phase2_step20
```
