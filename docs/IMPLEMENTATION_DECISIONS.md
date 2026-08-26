# Implementation Decisions

This document outlines scenarios where the paper left ambiguity, required explicit interpretation, or where necessary programmatic deviations were made to preserve the theoretical intent of the CoTOP paper.

## 1. Queue Delay (Eq. 5) Interpretation
- **Paper Equation**: $T^{wait}_{m,i}(t) = N^{queue}_m(t) / F^{RSU}_m$
- **Ambiguity**: The paper denotes $N^{queue}_m(t)$ as the queue length. However, dividing an integer task count by $F^{RSU}_m$ (measured in cycles/second) results in dimensionally invalid units (tasks * seconds / cycles).
- **Decision**: We interpret $N^{queue}_m(t)$ to mean the *total computational cycles* of all tasks currently in the queue, rather than the raw number of tasks. This aligns the dimensional analysis to seconds.

## 2. Vehicle State Space (Eq. 24)
- **Paper Equation**: $s_t^v = \{x_t, y_t\}$
- **Ambiguity**: The state definition technically only specifies spatial coordinates. However, velocity/speed and dwell_time critically inform Markovian transition probabilities for mobility detection.
- **Decision**: We explicitly augment the vehicle state space inside the environment to include normalized `speed` and `dwell_time_T_stay`. This is necessary for stable RL formulation in dynamic vehicular scenarios.

## 3. Mobility Graph Construction (GAT-GRU)
- **Paper Concept**: Spatial graph attention mechanism aggregating features from neighboring vehicles.
- **Ambiguity**: The paper does not specify the exact geometric radius or connection policy for establishing graph edges.
- **Decision**: We construct edges dynamically based on a physical proximity radius (e.g., 200m). If vehicles are within this range, an edge is formed. This strictly avoids the degenerate "self-loop only" graph structure.

## 4. RSU Geometry and Spacing
- **Paper Constraints**: Vehicle transmission range is bounded, and RSU coverage must physically intersect with the simulation area.
- **Ambiguity**: The exact road map length isn't strictly bounded beyond generic urban simulation traits. 
- **Decision**: We bind the total simulated road length to ~2400m matching real-world SUMO coordinates and explicitly distribute the 6 RSUs uniformly every 400m to ensure valid continuous coverage models.

## 5. State Normalization
- **Paper Equation**: The paper passes raw features into the A3C network.
- **Ambiguity**: Deep Reinforcement Learning fails catastrophically when feature magnitudes span from $1$ to $10^9$ (e.g. CPU capacity vs Priority). 
- **Decision**: A strict normalization pass is enforced over all $s(t)$ inputs before feeding them to the Actor-Critic networks, scaling everything to the `[0,1]` range using Table III theoretical maximums.

## 6. Local Baseline Isolation
- **Paper Concept**: Standalone task execution without collaboration.
- **Decision**: Rather than implementing "Local" as a hack inside the CoTOP environment loop (e.g., overriding `dwell_time` to infinity), we implement it as an independent `LocalPolicy` class that structurally guarantees `action=0` (Case 1) execution for rigid benchmarking.
