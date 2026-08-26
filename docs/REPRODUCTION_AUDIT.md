# CoTOP Reproduction Audit

## Critical Issues

| ID | Component | Paper | Current Code | Impact | Fix |
|----|-----------|-------|--------------|--------|-----|
| 1 | Action Space (Algorithm 1) | Discrete(M+1): 0=Standalone, >0=Collaborate with RSU M | Previously continuous/unmapped, now Discrete(M+1) | Critical decision failure | Fixed in Phase 2 |
| 2 | Total Delay Math (Eq. 6/10) | Distinct formulas for $T_{standalone}$ vs $T_{collab}$ | Was mathematically inaccurate without clear separation | Wrong reward function bounds | Fixed in Phase 1 |
| 3 | Energy Eq (Eq. 11, 12) | Distinguishes RF transmit power from compute power | Conflated $P_R$ (transmit) with compute power | Invalid energy numbers | Fixed in Phase 1 |
| 4 | Queue Delay Eq (Eq. 5) | $N_{queue} / F^{RSU}$, representing queued computational cycles | Queue handled as integer task count | Dimensionally invalid delay | Pending fix |
| 5 | RSU Geometry | 400m spacing based on vehicle/RSU range constraints | Hardcoded within a 200m bounding box originally | Geometrically impossible offloading | Fixed in Phase 2 |
| 6 | A3C Threading | Asynchronous shared parameter optimization across multiple workers | Python `threading` with manual `_grad` assignment | Race conditions, no parallel safety | Fixed in Phase 4 |
| 7 | Deterministic Seeding | Reproducible testing environments | No global seed enforced | Random results not reproducible | Fixed in Phase 4 |
| 8 | Mobility Spatial Graph | Multi-vehicle topological structure via GAT | Degenerate graph mapping (self-loops) | Fails to detect real interaction | Pending fix |
| 9 | Local Baseline | Separate standalone logic isolating RSU offloading | Overwrites current env step casually | Unreliable comparison metric | Fixed in Phase 3 |

## Major Issues

| ID | Component | Paper | Current Code | Impact | Fix |
|----|-----------|-------|--------------|--------|-----|
| 10 | Task Data Units | Bits for communication ($W$), CPU Cycles for compute | Passed Bytes straight into transmission eq | Values off by factor of 8 | Fixed in Phase 1 |
| 11 | State Dimensions (Eq. 24) | $s_t^v = \{x_t, y_t\}$ | Includes speed and dwell_time | Potentially violates strict paper state definition | Documenting in Decisions |
| 12 | State Normalization | Needs bounded range [0,1] or [-1,1] | Unnormalized values [1e0, 1e9] | RL collapse | Fixed in Phase 3 |
| 13 | V2R Rate Eq (Eq. 1) | Shannon capacity with path loss $D^{-\sigma}$ | Verify implementation matches paper $log_2$ and units | Transmission delay error | Pending audit |
| 14 | Evaluation Integrity | Dedicated mode without overriding physics dynamically | Injecting infinity into `dwell_time` during loop | Masked algorithmic failures | Pending fix |

## Minor Issues

| ID | Component | Paper | Current Code | Impact | Fix |
|----|-----------|-------|--------------|--------|-----|
| 15 | Parameter Truth | Table III variables | Scattered magic numbers in code | Hard to tune | Pending fix |
| 16 | Ablation Control | Explicit parameters (e.g. `no_mobility`) | Mixed with arg passing | Confusing testing matrix | Pending fix |
