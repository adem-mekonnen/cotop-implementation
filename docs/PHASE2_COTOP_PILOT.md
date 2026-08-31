# PHASE 2 CoTOP PILOT REPORT

## 1. Execution Context
- **Condition**: Linear Corridor (2400m), Workload I=20, Seed 42
- **Algorithm**: CoTOP (A3C, mathematical literal mapping)
- **Git SHA**: `e27fd31dd9bcb392abe1c65cc64dfa6fb0cce12d`
- **Checkpoint Hash**: `4187464a946b0737c724a57cf4730f5ae9c143c8d9f5e8406f8d9bfe489ffed4`

## 2. Platform Telemetry
- **Hardware**: Windows 10 (AMD64), CPUs: 12
- **Python**: 3.11.9
- **PyTorch**: 2.12.1+cpu
- **SUMO**: Eclipse SUMO sumo 1.27.1

## 3. Evaluation Metrics

### Subtask Disaggregated Metrics (True Physical Value)
- **Mean Delay**: 2.0722 s
- **Median Delay**: 1.7177 s
- **Delay StdDev**: 1.6194 s
- **Mean Energy**: 4.3199 J

### A1 Aggregation Hypothesis Metrics (Target Comparable)
- **Aggregate Delay**: 41.4430 s (Target: 13.90 s) -> **Discrepancy: 198.15%**
- **Aggregate Energy**: 86.3983 J (Target: 25.14 J) -> **Discrepancy: 243.67%**

### System Accounting
- **Total Generated**: 200
- **Completed**: 193 (Ratio: 96.5%)
- **Failed**: 7 (Ratio: 3.5%)
  - *Deadline*: 0
  - *Coverage*: 7
  - *Dual*: 0
  - *Departure*: 0

### Acceptance Status
- `No NaN / Inf`: **PASS**
- `Task Conservation Identity`: **PASS**
- `Queue Non-Negativity`: **PASS**
- `Deterministic Evaluation`: **PASS**
- `Frozen Trace Intact`: **PASS**

*Note: No parameter tuning was performed to force agreement with 13.90s / 25.14J. The remaining discrepancy is recorded purely mathematically.*
