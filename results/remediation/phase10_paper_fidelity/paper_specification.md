# Target Paper Formal Specification

**Title**: Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing  
**Venue**: IEEE Transactions on Mobile Computing (TMC) (2026)  
**Authors**: Du et al.  

## 1. System Model & Topology
- **Topologies**: Linear highway corridor (corridor_2400m) and 2D Manhattan grid (grid_200m)
- **Vehicles ($N$)**: [10, 30] (Nominal: 10)
- **RSUs ($M$)**: 6 RSUs with communication range $R = 400.0\text{ m}$
- **Vehicle Speed ($v$)**: $[30.0, 40.0]\text{ m/s}$
- **Time Slot**: $\Delta t = 1.0\text{ s}$, Horizon $T = 300.0\text{ s}$

## 2. Task & Compute Parameters (Table III)
- **Tasks per Vehicle**: $[20, 40]$
- **Task Payload ($\rho$)**: $[2.0, 5.0]\text{ MB}$ ($[2e+06, 5e+06]\text{ B}$)
- **Task Deadline ($d$)**: $[20.0, 30.0]\text{ s}$
- **Task CPU Demand ($\phi$)**: Nominal $10\text{ Mcycles}$ ($10^7\text{ cycles}$)
- **RSU Compute Capacity ($F$)**: $[1.0, 4.0]\text{ GHz}$ ($[10^9, 4\times 10^9]\text{ Hz}$)
- **Vehicle TX Power ($P_V$)**: $10\text{ dBm}$ ($0.01\text{ W}$)
- **RSU Optical Wireless TX Power ($P_R$)**: $50\text{ dBm}$ ($100.0\text{ W}$)
- **V2R Bandwidth ($B^{V2R}$)**: $[20.0, 100.0]\text{ MHz}$
- **R2R Optical Bandwidth ($B^{R2R}$)**: $50.0\text{ MHz}$

## 3. Published Headline Reference Values
- **Mean Delay**: **$13.9\text{ s}$**
- **Mean Energy**: **$25.14\text{ J}$**
- **Completion Ratio**: **$99.0\%$**
