# FRESH-CLONE SCIENTIFIC REPRODUCTION VERIFICATION REPORT

**Document Identifier**: `results/final_reproduction/FRESH_CLONE_VERIFICATION.md`  
**Target Manuscript**: *"Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing"* (Du et al., IEEE TMC 2026, DOI: `10.1109/TMC.2025.3631820`)  
**Verified Commit SHA**: `e8cba553cad2e193c8ac5b112cf938bd532ed2f4`  
**Canonical Branch**: `main`  
**Test Clone Location**: `d:\cotop-fresh-clone-test`  
**Verification Mode**: `all`  
**Verification Date**: `2026-09-05T06:27:19.845906+00:00`  
**Campaign Evaluations**: `420 / 420` successful (0 failed, 0 duplicate, 0 missing)  
**Raw Dataset SHA-256**: `7c9330dc9555f3fb47efd9e2f19112816ad1d11ddaa4ec116463fdf3535d7a2a`  

---

## 1. Executive Summary & Verification Evidence

An isolated clean clone was created at `d:\cotop-fresh-clone-test` and verified against the canonical repository:
1. Cloned repository `HEAD` matches `execution_git_sha` byte-for-byte (`e8cba553cad2e193c8ac5b112cf938bd532ed2f4`).
2. Workspace cleanliness verified (`git status --porcelain` is empty).
3. Path, `sys.path`, `PYTHONPATH`, and environment variables isolated from original workspace.
4. Protected physics models verified byte-for-byte under full 64-character SHA-256 checks.
5. All authentic reproducibility checkpoints verified.
6. Full regression test suite passed with `0 failed, 0 skipped`.
7. Pipeline execution was completely isolated to `d:\cotop-fresh-clone-test\results\fresh_clone_verification` with zero access or mutation to `results/final_reproduction/`.
8. Complete 420-run factorial evaluation executed independently with 100% data fidelity.
9. Raw dataset checksum `7c9330dc9555f3fb47efd9e2f19112816ad1d11ddaa4ec116463fdf3535d7a2a` matches canonical evidence.

Status: **PASS (100% Independent End-to-End Repeatability Verified under Frozen Inputs)**
