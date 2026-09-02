# GOOGLE COLAB GPU REPRODUCTION ENVIRONMENT SPECIFICATION

**Document ID**: `docs/COLAB_GPU_ENVIRONMENT.md`  
**Phase**: Phase 2 — Step 19 (Colab GPU Reproduction Environment)  
**Target Paper**: *"Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing"* (Du et al., IEEE TMC 2026)  
**Authoritative Git SHA**: `9426c1c979a9d28f12b89858bb4fd372ca96f0b4`  
**Status**: **VERIFIED & READY**  

---

## 1. Executive Summary

This document establishes the official Google Colab GPU execution environment and verification protocol for reproducing the experiments from Du et al. (IEEE TMC 2026). The protocol guarantees that the computational environment conforms strictly to physical theory, immutable cryptographic checkpoints, and experimental safety rules without parameter tuning toward published headline results.

---

## 2. Hardware & Runtime Specifications

### 2.1 Target Colab Hardware
- **Compute Accelerator**: NVIDIA GPU (NVIDIA T4 16GB, V100 16GB, A100 40/80GB, or L4 24GB)
- **Host Architecture**: x86_64 Linux (Ubuntu 22.04 LTS / Colab standard image)
- **CUDA Runtime**: CUDA 12.0+ / cuDNN 8.9+
- **PyTorch**: $\ge 2.1.0$ with CUDA support

### 2.2 Immutable Protected Physics Files
The physics computation and communication models must match these exact SHA-256 hashes:

| File | Canonical SHA-256 Hash | Status |
| :--- | :--- | :--- |
| `envs/comm_model.py` | `041e41061d02c7a5a7bc9488adf2bc49472177215730bd8a23c5ff2437431431` | **IMMUTABLE** |
| `envs/comp_model.py` | `dd9f58df710f709d536000bb4047d2ad6000cf37b1a49f4e1f0e8d883b856bff` | **IMMUTABLE** |

---

## 3. Step-by-Step Colab Environment Setup

In a Google Colab notebook cell with a GPU runtime selected (`Runtime -> Change runtime type -> T4/V100/A100 GPU`):

```bash
# 1. Clone the authoritative main commit
!git clone https://github.com/adem-mekonnen/cotop-implementation.git
%cd cotop-implementation
!git checkout 9426c1c979a9d28f12b89858bb4fd372ca96f0b4

# 2. Install SUMO and Linux traffic simulation prerequisites
!apt-get update -qq
!apt-get install -y -qq sumo sumo-tools sumo-doc
import os
os.environ["SUMO_HOME"] = "/usr/share/sumo"

# 3. Install exact Python dependencies
!pip install -r requirements.txt
!pip install torch-geometric

# 4. Verify GPU runtime and execute diagnostic smoke test
!python scripts/verify_colab_gpu.py
```

---

## 4. Verification & Diagnostic Protocol

The script `scripts/verify_colab_gpu.py` executes an automated 6-step sanity gate:

1. **Cryptographic Physics Verification**: Computes SHA-256 hashes of `envs/comm_model.py` and `envs/comp_model.py`. Any discrepancy halts execution immediately.
2. **CUDA Hardware Check**: Verifies `torch.cuda.is_available() == True`. Fails loudly if a CPU runtime is detected.
3. **GPU Matrix Multiplication Smoke Test**: Allocates $(1000 \times 1000)$ tensors on CUDA, performs dense GEMM operations, and verifies numerical stability.
4. **DDQN Minimal Baseline Test**: Instantiates `DDQNAgent` on the active GPU device, steps through experience storage, computes Huber loss, and verifies target network mechanics.
5. **CoTOP & Mobility GAT-GRU Test**: Evaluates `ActorCritic` and `MobilityGAT_GRU` on GPU, performing spatial graph attention convolutions across vehicle nodes.
6. **Evaluation Determinism**: Verifies bitwise determinism across forward passes under fixed seeds.

---

## 5. Experimental Safety Rules for Colab Execution

1. **No Parameter Tuning**: Do not alter learning rates, queue capacities, task arrival rates, or energy coefficients to force target convergence to $13.90\text{ s}$ / $25.14\text{ J}$.
2. **QRMP-DQN Exclusion**: Reference [33] remains formally classified as `EXCLUDED (DOMAIN MISMATCH - STAR-RIS)` per `docs/PHASE2_QRMP_DQN_DISPOSITION.md`.
3. **Exogenous Realization Integrity**: All algorithmic comparisons must execute against cryptographically verified exogenous realizations (`utils/realization.py`).
4. **Auditability**: Every generated artifact, evaluation CSV, and training log must record its execution timestamp, device name, and Git SHA.
