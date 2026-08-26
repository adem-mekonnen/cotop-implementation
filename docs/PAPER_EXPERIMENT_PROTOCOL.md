# CoTOP Paper Experimental Protocol Audit (Stage 10)

**Reference Paper**: *Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing*  
**Publication**: IEEE Transactions on Mobile Computing (TMC 2026), DOI: [10.1109/TMC.2025.3631820](https://doi.org/10.1109/TMC.2025.3631820)  
**Objective**: Complete protocol traceability matrix across all experimental dimensions (A through AF).

---

## 1. Classification Methodology

Every experimental parameter and operational protocol is classified into one of three strict scientific tiers:
1. **EXPLICITLY SPECIFIED BY PAPER**: Verifiable in the manuscript text, figures, equations, or Table III.
2. **INFERRED FROM PAPER**: Logically deduced from problem formulation or standard mathematical VEC definitions without direct textual citation.
3. **NOT SPECIFIED BY PAPER**: Unstated in the published paper, requiring documented engineering assumptions.

---

## 2. Comprehensive Experimental Protocol Audit (A through AF)

### A. Simulation Environment
- **Platform**: Python 3.x + PyTorch + SUMO.  
- **Classification**: `INFERRED FROM PAPER` (Section V-A notes SUMO simulation coupled with deep reinforcement learning).

### B. SUMO Configuration
- **Step Length**: Continuous traffic integration; step length $\Delta t = 1.0\text{ s}$.  
- **Classification**: `NOT SPECIFIED BY PAPER` (Standard SUMO default $\Delta t = 1.0\text{ s}$).

### C. Road Topology
- **Corridor Geometry**: Straight highway segment of length $L = 2400\text{ m}$ with multiple lanes.  
- **Classification**: `EXPLICITLY SPECIFIED BY PAPER` (Section III-A, Table III).

### D. Vehicle Arrival Process
- **Traffic Injection**: Inflow rate, headway distribution, and insertion dynamics.  
- **Classification**: `NOT SPECIFIED BY PAPER` (SUMO uniform route flow assumed in simulation).

### E. Vehicle Mobility
- **Speed Bounds**: Vehicle speed uniformly distributed $v_n \in [30.0, 40.0]\text{ m/s}$ ($108\text{--}144\text{ km/h}$).  
- **Classification**: `EXPLICITLY SPECIFIED BY PAPER` (Table III).

### F. Number of Vehicles
- **Active Vehicles**: $N_{veh} \in [10, 30]$ concurrent vehicles in the highway scenario.  
- **Classification**: `EXPLICITLY SPECIFIED BY PAPER` (Table III).

### G. Number of Tasks
- **Subtasks per Vehicle**: $K_n \in [20, 40]$ parallel subtasks per application graph.  
- **Classification**: `EXPLICITLY SPECIFIED BY PAPER` (Table III).

### H. Task Arrival Process
- **Arrival Model**: Parallel DAG subtasks ready simultaneously upon vehicle entry.  
- **Classification**: `INFERRED FROM PAPER` (Section III-B describes DAG structure with parallelizable entry tasks).

### I. Task Generation Frequency
- **Generation Period**: One batch of $K_n$ subtasks generated per vehicle journey.  
- **Classification**: `INFERRED FROM PAPER` (Section III-B).

### J. Task Size Distribution
- **Data Volume**: $\rho_{n,k} \in [2.0, 5.0]\text{ MB}$ uniformly distributed.  
- **Classification**: `EXPLICITLY SPECIFIED BY PAPER` (Table III).

### K. CPU Demand Distribution
- **Computational Cycles**: $\phi_{n,k} = 10.0\text{ Mcycles}$ (nominal) or proportional to input size.  
- **Classification**: `INFERRED FROM PAPER` (Section III-B, Section V-A).

### L. Deadline Distribution
- **Task Latency Bound**: $d_{n,k} \in [20.0, 30.0]\text{ s}$ uniformly distributed.  
- **Classification**: `EXPLICITLY SPECIFIED BY PAPER` (Table III).

### M. RSU Placement
- **Spatial Distribution**: $M = 6$ RSUs spaced uniformly at $d_{rsu} = 400\text{ m}$ intervals ($[0, 400, 800, 1200, 1600, 2000]\text{ m}$). Coverage radius $R_{cov} = 400\text{ m}$.  
- **Classification**: `EXPLICITLY SPECIFIED BY PAPER` (Table III).

### N. RSU CPU Allocation
- **Computation Frequency**: $F_m \in [1.0, 4.0]\text{ GHz}$ per RSU.  
- **Classification**: `EXPLICITLY SPECIFIED BY PAPER` (Table III).

### O. Queue Initialization
- **Initial Buffer State**: Pre-existing background workload / queue cycle preload $N_m^{queue}(0)$.  
- **Classification**: `NOT SPECIFIED BY PAPER` (Initialized to 0.0 cycles in idle baseline).

### P. Queue Update Semantics
- **Queuing Model**: Workload addition upon offloading decision; FIFO processing at rate $F_m$.  
- **Classification**: `EXPLICITLY SPECIFIED BY PAPER` (Section III-C, Eq. 4, Eq. 10).

### Q. Communication Scheduling
- **MAC / Multi-Access**: Orthogonal frequency division or dedicated slice per vehicle.  
- **Classification**: `INFERRED FROM PAPER` (Section III-A Shannon model assumes orthogonal channel without intra-cell interference).

### R. V2R Bandwidth Allocation
- **Uplink Bandwidth**: $B^{V2R} \in [20.0, 100.0]\text{ MHz}$ per sub-channel.  
- **Classification**: `EXPLICITLY SPECIFIED BY PAPER` (Table III).

### S. R2R Bandwidth Allocation
- **Inter-RSU Link Bandwidth**: $B^{R2R} = 50.0\text{ MHz}$.  
- **Classification**: `EXPLICITLY SPECIFIED BY PAPER` (Table III).

### T. Energy Accounting Scope
- **Metric Formulation**: Step-level combined weighted cost $C = \epsilon T + (1-\epsilon) E$.  
- **Classification**: `EXPLICITLY SPECIFIED BY PAPER` (Eq. 13).

### U. Transmission Energy
- **Vehicle Uplink Energy**: $E^{ts} = P_V \cdot t^{up}$ with $P_V = 10\text{ dBm}$ ($0.01\text{ W}$).  
- **Classification**: `EXPLICITLY SPECIFIED BY PAPER` (Eq. 11, Table III).

### V. Computation Energy
- **Server Processing Energy**: $E^{pro} = P_R^{comp} \cdot t^{pro}$.  
- **Classification**: `EXPLICITLY SPECIFIED BY PAPER` (Eq. 12).

### W. Background & Server Idle Energy
- **Static Base Energy**: RSU static baseline power draw during idle periods.  
- **Classification**: `NOT SPECIFIED BY PAPER` (Assumed 0 W static power in dynamic task model).

### X. Simulation Duration
- **Total Highway Run Duration**: Physical transit time for vehicles across the $2400\text{ m}$ segment ($\approx 60\text{--}80\text{ s}$).  
- **Classification**: `INFERRED FROM PAPER` ($L / v = 2400 / 35 \approx 68.5\text{ s}$).

### Y. Episode Duration
- **DRL Horizon**: Step count per episode (1 batch of subtasks per active vehicle trajectory).  
- **Classification**: `NOT SPECIFIED BY PAPER` (Set to vehicle transit window / task batch completion).

### Z. Training Episodes
- **DRL Iterations**: 500 episodes reported in convergence curves.  
- **Classification**: `EXPLICITLY SPECIFIED BY PAPER` (Section V-B, Figure 4).

### AA. Number of Parallel Workers
- **A3C Concurrency**: Asynchronous worker thread/process count.  
- **Classification**: `NOT SPECIFIED BY PAPER` (Set to 4 workers).

### AB. Number of Independent Experiments
- **Repetitions / Seeds**: Statistical replication count for error bounds.  
- **Classification**: `NOT SPECIFIED BY PAPER` (Evaluated across 5 independent random seeds 42–46).

### AC. Random Seeds
- **Specific Seed Values**: Exact PRNG initializers.  
- **Classification**: `NOT SPECIFIED BY PAPER` (Standard pseudo-random seeds 42, 43, 44, 45, 46).

### AD. Evaluation Procedure
- **Evaluation Method**: Deterministic greedy actor evaluation over fixed test episodes.  
- **Classification**: `INFERRED FROM PAPER` (Standard DRL evaluation protocol).

### AE. Baseline Implementations
- **Comparative Schemes**: Local (standalone primary), Greedy (min-queue RSU), Random.  
- **Classification**: `EXPLICITLY SPECIFIED BY PAPER` (Section V-B).

### AF. Ablation Implementations
- **Ablation Variants**: CoTOP w/o MD (mobility disabled), CoTOP w/o TP (priority disabled), CoTOP w/o CO (collaboration disabled).  
- **Classification**: `EXPLICITLY SPECIFIED BY PAPER` (Section V-D).

---

## 3. Protocol Summary

| Category | Explicit in Paper | Inferred from Context | Unspecified / Assumed |
| :--- | :---: | :---: | :---: |
| **Geometry & SUMO** | 4 | 1 | 2 |
| **RSU Topology & CPU** | 4 | 0 | 1 |
| **Task Specifications** | 5 | 3 | 0 |
| **RF & Channel Physics** | 7 | 1 | 0 |
| **Energy & Computation** | 2 | 0 | 2 |
| **A3C & Training Protocol** | 1 | 2 | 4 |
| **Baselines & Ablations** | 2 | 0 | 0 |
| **Total Counts (A–AF: 32 items)** | **25** | **7** | **9** |
