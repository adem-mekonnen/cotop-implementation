# PHASE 2: CoTOP vs DDQN Causal Pilot Comparison

## Pipeline Validation Experiment
This pilot validates the strict realization decoupling and fairness boundaries between CoTOP and DDQN baselines. Both algorithms were trained and evaluated on an exact canonical trace.

### 1. Invariant Assurances
- **H1 (Identical Exogenous Realization)**: PASS
- **H2 (CoTOP Deterministic Eval)**: PASS
- **H3 (DDQN Deterministic Eval)**: PASS
- **H4 (Eval Cannot Mutate Weights)**: PASS
- **H5 (Eval Cannot Mutate Realization)**: PASS
- **H6 (State/Action Semantics)**: PASS
- **H7 (Task Accounting Identical)**: PASS (Both faced exactly 200 tasks)
- **H8 (Latency Decomposition Identical)**: PASS
- **H9 (Energy Decomposition Identical)**: PASS

### 2. Disaggregated Evaluation Results

#### 2.1 CoTOP Metrics
- **Mean Delay**: 2.0682 s
- **Mean Energy**: 3.7349 J
- **Completed**: 193
- **Failed**: 7

#### 2.2 DDQN Metrics
- **Mean Delay**: 2.0544 s
- **Mean Energy**: 5.5292 J
- **Completed**: 194
- **Failed**: 6

### 3. Paired Differentials (CoTOP - DDQN)
- **Δ Mean Delay**: +0.0137 s
- **Δ Mean Energy**: -1.7943 J
- **Δ Completed**: -1 tasks
- **Δ Failed**: +1 tasks

*Note: Statistical significance is omitted (n=1). This confirms pipeline fairness readiness.*
