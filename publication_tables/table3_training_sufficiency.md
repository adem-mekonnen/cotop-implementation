| Training Horizon | Mean Cumulative Reward | Reward Std Across Seeds | Mean Delay (s) | Mean Energy (J) | Critic Loss (MSE) | Convergence Status | Scientific Conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 Epochs (100 Episodes) | -63.28 | 0.84 | 4.595 | 0.347 | 0.418 | Initial Stabilization | Initial learning phase; policy begins favoring standalone execution |
| 50 Epochs (500 Episodes) | -47.21 | 0.05 | 4.402 | 0.319 | 0.000582 | Full Asymptotic Convergence | Policy reaches optimal plateau by epoch 35-40; variance across seeds minimal |
| 100 Epochs (1000 Episodes) | -47.21 | 0.05 | 4.402 | 0.319 | 0.000421 | Mature Plateau | Zero material change in policy, delay, or energy; proves training sufficiency |
