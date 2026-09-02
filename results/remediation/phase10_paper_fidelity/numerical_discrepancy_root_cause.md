# Scientific Root Cause Analysis: Numerical Scale Discrepancy

**Target Values in Paper**: Mean Total Delay $\approx 13.90\text{ s}$, Mean Energy $\approx 25.14\text{ J}$  
**Reproduced Values**: Mean Total Delay $= 1.3513\text{ s}$, Mean Energy $= 4.0355\text{ J}$  
**Discrepancy Scale Factor**: $\approx 10.28\times$ in Delay, $\approx 6.23\times$ in Energy  

---

## 1. Physical Equation Trace & Unit Dimensionality
Under the literal parameters specified in **Table III**:
1. **Computation Latency ($T^{pro}$)**:
   $$\phi = 10\text{ Mcycles} = 1.0\times 10^7\text{ cycles}$$
   $$F_{RSU} \in [1.0, 4.0]\text{ GHz} = [1.0\times 10^9, 4.0\times 10^9]\text{ Hz}$$
   $$T^{pro} = \frac{\phi}{F_{RSU}} = \frac{10^7}{2\times 10^9} = 0.005\text{ s}\quad (5\text{ ms})$$
2. **Communication Latency ($T^{up}$)**:
   $$\rho = 2.0\text{ MB} = 1.6\times 10^7\text{ bits}$$
   $$W_{v,m} \approx 15\text{ Mbps}$$
   $$T^{up} = \frac{1.6\times 10^7}{1.5\times 10^7} \approx 1.07\text{ s}$$
3. **Total Physical Latency**:
   $$T_{total} = T^{up} + T^{pro} + T^{wait} \approx 1.07\text{ s} + 0.005\text{ s} + 0.05\text{ s} \approx 1.13 - 1.35\text{ s}$$

## 2. Energy Decomposition
1. **Vehicle Uplink Transmission**: $P_V \times T^{up} = 0.01\text{ W} \times 1.07\text{ s} = 0.0107\text{ J}$.
2. **Optical Wireless Forwarding**: $P_R \times T^{ts} = 100.0\text{ W} \times 0.038\text{ s} = 3.80\text{ J}$.
3. **RSU Computation**: $P_{comp} \times T^{pro} = 50.0\text{ W} \times 0.005\text{ s} = 0.25\text{ J}$.
4. **Total Energy Integral**: $E_{total} \approx 0.01 + 3.80 + 0.25 = 4.06\text{ J}$.

## 3. Conclusion & Integrity Decision
The repository reproduces the exact analytical output of Table III equations. For delay to physically reach $13.90\text{ s}$, task sizes would have to be $20-50\text{ MB}$ or CPU cycles $10\text{ Gcycles}$. We preserve the exact physical equations and document this discrepancy as an unstated paper scaling factor rather than fitting synthetic coefficients.
