# PHASE 2: CHECKPOINT FORENSIC AUDIT & PROVENANCE RECORD

**Document ID**: `DOC-PHASE2-CHECKPOINT-AUDIT-001`  
**Classification**: Cryptographic Artifact Audit  
**Authoritative Branch**: `merge/reconcile-reproduction-branches`  
**Audit Date**: August 31, 2026  

---

## 1. Checkpoint Classification Criteria

Every checkpoint in the repository is classified under one of four strict forensic categories:

- **`VERIFIED`**: Full provenance (generating script, configuration YAML, random seed, realization trace SHA-256, model weights SHA-256, and evaluation metrics) confirmed.
- **`REPRODUCIBLE`**: Generating code exists and passes regression tests, but weights are auxiliary/exploratory rather than primary factorial.
- **`UNVERIFIABLE`**: Missing configuration, seed, or realization trace; cannot be deterministically reproduced.
- **`REJECTED`**: Generated under obsolete or broken physics / single-vehicle abstractions; excluded from reproduction tables.

---

## 2. Checkpoint Forensic Inventory

| Checkpoint Group / Directory | Number of Files | Generating Script | Realization Provenance | Manifest Present | Forensic Classification | Decision |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: |
| **`results/phase2_multiseed/` (Canonical)** | 60 | `scripts/run_phase2_multiseed_training.py` | 30 Locked JSON Realizations | **YES** (`run_manifest.json`) | **`VERIFIED`** | **RETAIN (AUTHORITATIVE)** |
| **`results/checkpoints/mobility_model.pth`** | 1 | `models/mobility_gat.py` | SUMO trajectory dataset | **YES** | **`VERIFIED`** | **RETAIN (AUTHORITATIVE)** |
| **`results/stage13/checkpoints/`** | 5 | `scripts/run_stage13_validation.py` | Stage 13 corridor trace | **YES** | **`REPRODUCIBLE`** | **ARCHIVE (AUDIT RECORD)** |
| **`results/stage9/checkpoints/`** | 10 | `scripts/run_stage9_pilot.py` | Single-condition pilot | **YES** | **`REPRODUCIBLE`** | **ARCHIVE (AUDIT RECORD)** |
| **`phase2_algorithmic_fidelity/` (PVA branch)** | 35 | Older Colab script | Unlocked legacy trace | **NO** | **`UNVERIFIABLE`** | **EXCLUDE / QUARANTINE** |
| **Miscellaneous orphaned `.pt` files** | ~50 | Unknown / Scratch | None | **NO** | **`REJECTED`** | **EXCLUDE** |

---

## 3. Quarantine and Exclusion Rules

1. **No Orphan Import**: No checkpoint file from `reproduction/published-value-audit` or `reproduction/multivehicle-contention` is imported into the authoritative canonical results directory without its corresponding generating script, seed, and realization manifest.
2. **Canonical Exclusivity**: All published tables (Table IV, Table V, Table VI) and figures draw strictly from the 60 **`VERIFIED`** canonical checkpoints in `results/phase2_multiseed/`.
