# Stage 7 Reproduction & Scientific Audit Log

This document records the resolution of the identical-action bug and the full verification of the physical and algorithmic layers.

---

## Summary of Critical Fixes

1. **Collaboration Fallback Elimination**:
   - **Root Cause**: `calculate_case2_collaboration` compared `cpu_processed_rsu1 = rsu1_cpu_f * t1_dwell_time` with `task_cpu_phi`. Because raw dwell time ($10\text{ s}$) at $2\text{ GHz}$ yielded $20\text{ Gcycles} \gg 10\text{ Mcycles}$, the condition `cpu_processed_rsu1 >= task_cpu_phi` always evaluated to True, forcing Case 1 standalone fallback for all collaboration actions.
   - **Resolution**: Implemented parallel task partitioning (Eq. 7–10) allocating compute duration $t_1 = \min(t_1\_dwell\_time, \text{part\_ratio} \times \frac{\phi}{F_1})$, correctly transferring remaining load $\phi^{rest}$ to the collaborative RSU.

2. **Secondary Queue Routing**:
   - **Root Cause**: `envs/vec_env.py` passed primary RSU wait time `t_wait_target` to Case 2.
   - **Resolution**: Routed `t_wait_secondary = secondary_rsu.queued_cpu_cycles / secondary_rsu.cpu_capacity_f` per Eq. 10.

3. **Mobility Coordinate Normalization**:
   - **Root Cause**: Model trained on unnormalized raw meter coordinates ($0\text{--}2400\text{ m}$), causing gradient explosion and high MSE.
   - **Resolution**: Scaled coordinates by $2400.0\text{ m}$ into $[0, 1]$ during training/inference.
