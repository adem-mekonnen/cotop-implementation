# Table 2: Objective-by-Objective Performance Summary (N=60 Evaluation Configurations)

| Algorithm | Mean Delay (s) | Delay Std | Mean Energy (J) | Energy Std | Completion Ratio (%) | Collaboration Rate (%) | Status / Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Local** | 1.3335 | 0.6674 | 0.2892 | 0.0106 | 99.31% | 0.00% | Energy-Optimal Minimizer |
| **Greedy** | 1.3111 | 0.6882 | 5.1209 | 1.9998 | 99.23% | 87.22% | Delay-Aggressive Minimizer |
| **DDQN** | 1.3319 | 0.6766 | 1.6298 | 0.9320 | 99.21% | 40.04% | Balanced Q-Learning Offloader |
| **CoTOP** | 1.3566 | 0.6947 | 2.6747 | 1.8177 | 99.08% | 99.92% | Collaborative Actor-Critic |
| **wo_co** | 1.3335 | 0.6674 | 0.2892 | 0.0106 | 99.31% | 0.00% | Ablation: Collaboration Disabled |
| **wo_md** | 1.3348 | 0.6787 | 1.5402 | 0.8693 | 99.22% | 99.92% | Ablation: Mobility Attention Disabled |
| **wo_tp** | 1.3384 | 0.6904 | 3.6732 | 2.2876 | 99.12% | 100.00% | Ablation: Priority Queue Disabled |
| **QRMP-DQN** | N/A | N/A | N/A | N/A | N/A | N/A | **Not Reproducible From Available Evidence** |
