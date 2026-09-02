# PHASE 2 — STEP 20: GPU CAMPAIGN INFRASTRUCTURE VALIDATION REPORT

**Document ID**: `docs/PHASE2_STEP20_GPU_CAMPAIGN_VALIDATION.md`  
**Phase**: Phase 2 — Step 20 (GPU Campaign Infrastructure Validation)  
**Target Paper**: *"Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing"* (Du et al., IEEE TMC 2026)  
**Status**: **VALIDATED & READY FOR CLOUD GPU EXECUTION**  

---

## 1. Executive Summary

This document validates the complete experimental execution and checkpointing infrastructure for the final Google Colab GPU campaign. The runner ([scripts/run_phase2_gpu_campaign.py](file:///d:/cotop-implementation/scripts/run_phase2_gpu_campaign.py)) enforces loud failure if CUDA is absent, provides periodic checkpointing, enables seamless resumption after Colab runtime interruptions, isolates experiment outputs, records complete cryptographic manifests, and performs deterministic evaluation over frozen exogenous realizations.

---

## 2. Infrastructure & Device Diagnostics

### 2.1 Hardware Verification & Loud Failure Policy
The campaign runner verifies CUDA availability prior to batch initialization. If executed in an environment without an active NVIDIA CUDA device (e.g. CPU-only runtime), the runner immediately exits with a non-zero exit code:
```text
[ERROR] CUDA IS NOT AVAILABLE! Halting execution.
Google Colab requires a GPU runtime (Runtime -> Change runtime type -> T4/V100/A100 GPU).
```
CPU execution is permitted only in diagnostic smoke testing via `--allow-cpu`.

### 2.2 Tensor Device Routing
- **ActorCritic / MobilityGAT_GRU**: State representations $\mathbb{R}^{114}$ and graph spatial sequences $\mathbb{R}^{N \times T \times 2}$ with edge indices are transferred directly to device memory. Softmax masking with $-10^9$ occurs on-device.
- **DDQNAgent**: High-performance preallocated NumPy ring buffer (`ReplayBuffer`) transfers sample minibatches $(B=64)$ directly to GPU tensors (`to(device)`) during gradient updates.
- **Evaluation Loop**: Tensors are evaluated within `torch.no_grad()` contexts to avoid memory leaks during long multi-seed evaluations.

---

## 3. Checkpoint & Resume Architecture

### 3.1 Checkpoint Payload Specification
Every generated checkpoint `checkpoint.pt` captures the complete execution state:
```python
checkpoint_data = {
    "episode": current_episode,
    "global_step": train_step_count,
    "epsilon": current_epsilon,
    "online_net_state_dict": agent.online_net.state_dict(),
    "target_net_state_dict": agent.target_net.state_dict(),
    "optimizer_state_dict": agent.optimizer.state_dict(),
    "git_sha": "...",
    "physics_hashes": {"comm": COMM_SHA256, "comp": COMP_SHA256},
    "rng_state": {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state()
    }
}
```

### 3.2 Interruption Protection & Resumption
When `--resume` is specified:
1. The runner inspects the target directory for existing `checkpoint.pt` and `evaluation_metrics.json`.
2. If `evaluation_metrics.json` exists, the run is recognized as completed and skipped automatically.
3. If partial training exists, the model, optimizer, step counters, and RNG states are restored, and training resumes from episode $E+1$ up to the configured episode target.

---

## 4. Output Isolation & Directory Hierarchy

Every experiment outputs to a deterministic, isolated directory path:
```text
results/phase2_step20/
  ├── <algorithm>/          (DDQN, CoTOP)
  │   ├── <scenario>/       (corridor_2400m, grid_200m)
  │   │   ├── <workload>/   (w20, w30, w40)
  │   │   │   ├── seed_<seed>/
  │   │   │   │   ├── checkpoint.pt
  │   │   │   │   ├── config.yaml
  │   │   │   │   ├── run_manifest.json
  │   │   │   │   ├── realization_manifest.json
  │   │   │   │   ├── training_metrics.json
  │   │   │   │   ├── evaluation_metrics.json
  │   │   │   │   ├── training_curve.csv
  │   │   │   │   └── evaluation_results.csv
```

---

## 5. Experiment Manifest Schema

Every run generates an authoritative `run_manifest.json`:
```json
{
  "git_commit_sha": "2156339e828ff847170524daa55ac6939e86bb5c",
  "git_branch": "main",
  "timestamp": "2026-09-02T07:48:42Z",
  "algorithm": "DDQN",
  "scenario": "corridor_2400m",
  "workload": 20,
  "seed": 42,
  "episodes": 500,
  "hardware": {
    "device": "cuda:0",
    "gpu_name": "NVIDIA A100-SXM4-40GB",
    "gpu_mem_mb": 40960.0,
    "cuda_ver": "12.1"
  },
  "software": {
    "python_version": "3.11.9",
    "pytorch_version": "2.12.1"
  },
  "physics_hashes": {
    "comm_model_sha256": "041e41061d02c7a5a7bc9488adf2bc49472177215730bd8a23c5ff2437431431",
    "comp_model_sha256": "dd9f58df710f709d536000bb4047d2ad6000cf37b1a49f4e1f0e8d883b856bff"
  },
  "realization_sha256": "...",
  "checkpoint_sha256": "...",
  "status": "COMPLETED"
}
```

---

## 6. Smoke Experiment Validation

A smoke test executed on `DDQN/corridor_2400m/w20/seed_42`:
- **Training**: Completed 2 episodes without exceptions.
- **Checkpoint Generation**: Saved complete state dictionary ($804\text{ KB}$).
- **Resume Validation**: Rerunning with `--resume` verified automatic skip of completed runs.
- **Deterministic Evaluation**: Generated `evaluation_results.csv` ($19.6\text{ KB}$) and `evaluation_metrics.json`.
- **Physics Invariance**: `comm_model.py` and `comp_model.py` verified untouched.

---

## 7. Cryptographic Physics Integrity

```text
envs/comm_model.py: 041e41061d02c7a5a7bc9488adf2bc49472177215730bd8a23c5ff2437431431 (EXACT)
envs/comp_model.py: dd9f58df710f709d536000bb4047d2ad6000cf37b1a49f4e1f0e8d883b856bff (EXACT)
```

---

## 8. Exact Full Colab Campaign Commands

In a Google Colab notebook cell with GPU runtime enabled:

```bash
# 1. Clone repository and install dependencies
!git clone https://github.com/adem-mekonnen/cotop-implementation.git
%cd cotop-implementation
!git checkout main
!apt-get update -qq && apt-get install -y -qq sumo sumo-tools sumo-doc
!pip install -r requirements.txt

# 2. Run GPU verification sanity gate
!python scripts/verify_colab_gpu.py

# 3. Launch full multi-seed GPU campaign with resume protection
!python scripts/run_phase2_gpu_campaign.py \
    --algorithm all \
    --scenario all \
    --workload all \
    --seed 42,43,44,45,46 \
    --episodes 500 \
    --device cuda:0 \
    --resume \
    --output-dir results/phase2_step20
```
