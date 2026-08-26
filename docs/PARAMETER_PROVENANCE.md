# Parameter Provenance and Scientific Evidence Matrix

**Paper Title**: *Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing*  
**Publication**: IEEE Transactions on Mobile Computing (TMC 2026), DOI: 10.1109/TMC.2025.3631820  

This document classifies every parameter used across the CoTOP codebase into one of four scientific categories:
1. **PAPER_EXPLICIT**: Value explicitly provided in Table III or body text of the paper.
2. **PAPER_DERIVED**: Mathematically derived from explicit paper equations or standards.
3. **DOCUMENTED_ASSUMPTION**: Required engineering parameter not stated in the paper, fully justified and documented.
4. **IMPLEMENTATION_DEFAULT**: Standard DRL / runtime configuration.

---

## 1. System Parameters Matrix

| Parameter | Paper Location | Paper Specification | Implementation Value | Internal Unit | Provenance Status | Justification / Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Road Length** | Section III-A | 2400 m | 2400.0 | meters ($m$) | `PAPER_EXPLICIT` | Corridor highway length |
| **Number of RSUs ($M$)** | Table III | 6 | 6 | count | `PAPER_EXPLICIT` | Table III RSU count |
| **RSU Spacing** | Section III-A, Table III | 400 m | 400.0 | meters ($m$) | `PAPER_EXPLICIT` | Evenly spaced along 2400m corridor |
| **RSU Comm. Range ($R$)**| Table III | 400 m | 400.0 | meters ($m$) | `PAPER_EXPLICIT` | Table III communication range |
| **Vehicle Count ($N$)** | Table III | [10, 30] | [10, 30] | count | `PAPER_EXPLICIT` | Table III vehicle count range |
| **Vehicle Speed ($v$)** | Table III | [30.0, 40.0] m/s | [30.0, 40.0] | $\text{m/s}$ | `PAPER_EXPLICIT` | Table III vehicle speed |
| **Tasks Per Vehicle ($I$)**| Table III | [20, 40] | [20, 40] | count | `PAPER_EXPLICIT` | Table III task count |
| **Task Data Size ($\rho$)** | Table III | [2, 5] MB | $[2\times 10^6, 5\times 10^6]$ | Bytes ($B$) | `PAPER_EXPLICIT` | $1\text{ Byte} = 8\text{ bits}$ |
| **Task CPU Demand ($\phi$)**| Section III-F | 10 Mcycles | $10.0\times 10^6$ | Cycles | `PAPER_EXPLICIT` | Average workload per task |
| **Task Deadline ($d$)** | Table III | [20, 30] s | [20.0, 30.0] | seconds ($s$) | `PAPER_EXPLICIT` | Table III deadline range |
| **RSU CPU Capacity ($F$)** | Table III | [1, 4] Gcycles/s | $[1.0\times 10^9, 4.0\times 10^9]$| $\text{Hz}$ | `PAPER_EXPLICIT` | $1\text{ GHz} = 10^9\text{ cycles/s}$ |
| **Vehicle TX Power ($P_V$)**| Table III | 10 dBm | $0.01$ | Watts ($W$) | `PAPER_DERIVED` | $10^{(10-30)/10} = 0.01\text{ W}$ |
| **RSU TX Power ($P_R$)** | Table III | 50 dBm | $100.0$ | Watts ($W$) | `PAPER_DERIVED` | $10^{(50-30)/10} = 100.0\text{ W}$ |
| **RSU Compute Power ($E_{RSU}$)**| Section III-D (Eq. 11)| Not in Table III | $50.0$ | Watts ($W$) | `DOCUMENTED_ASSUMPTION`| Standard edge server CPU power consumption |
| **V2R Bandwidth ($B^{V2R}$)**| Table III | [20, 100] MHz | $[20\times 10^6, 100\times 10^6]$| $\text{Hz}$ | `PAPER_EXPLICIT` | Table III V2R bandwidth |
| **R2R Bandwidth ($B^{R2R}$)**| Table III | 50 MHz | $50.0\times 10^6$ | $\text{Hz}$ | `PAPER_EXPLICIT` | Table III R2R bandwidth |
| **Noise Power ($\omega$)** | Table III | 0.001 dBm | $0.001$ | Watts ($W$) | `PAPER_EXPLICIT` | Table III noise power |
| **Fixed Loss ($K$)** | Table III | 30 dB | $1000.0$ | ratio | `PAPER_DERIVED` | $10^{30/10} = 1000.0$ |
| **Path Loss Exponent ($\sigma$)**| Table III | 2.0 | $2.0$ | dimensionless | `PAPER_EXPLICIT` | Table III free-space attenuation |
| **Priority Weights ($\alpha, \beta$)**| Section V-C | 0.3, 0.7 | 0.3, 0.7 | weights | `PAPER_EXPLICIT` | Section V-C ($\alpha + \beta = 1$) |
| **Reward Tradeoff ($\epsilon$)**| Section IV-D1 (Eq. 25)| Not Specified | $0.5$ | weight | `DOCUMENTED_ASSUMPTION`| Equal weighting between delay and energy |
| **Deadline Penalty ($Z$)** | Section IV-D1 (Eq. 25)| Not Specified | $100.0$ | unitless | `DOCUMENTED_ASSUMPTION`| Negative reward penalty for deadline violation |
| **Learning Rate** | Section V-C | 0.0002 | 0.0002 | lr | `PAPER_EXPLICIT` | Section V-C A3C learning rate |
| **Discount Factor ($\gamma$)**| Section IV-D2 (Eq. 27)| Not in Table III | $0.99$ | discount | `IMPLEMENTATION_DEFAULT`| Standard DRL discount factor |
