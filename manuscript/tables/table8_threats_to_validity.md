| Category | Limitation Description | Scientific Impact |
| --- | --- | --- |
| Undisclosed Protocol Parameters | Target paper does not state initial edge server queue preload N_queue(0) or background vehicle arrival rates. | Prevents exact numerical delay replication without post-hoc diagnostic queue assumptions. |
| Metric Scope Ambiguity | Target paper does not explicitly define whether 'Average Energy' is per-task, per-vehicle, or episode-batch. | Creates an ~80x numerical gap between single-task physics (0.319J) and published batch curve (25.14J). |
| Mobility Dataset Availability | Multi-GB raw ApolloScape trajectory dataset was not bundled with the codebase. | Synthetic kinematic trajectories used to validate spatial graph attention; classified as method validation. |
| Hardware Concurrency Adaptation | Colab free tier runtime constrained A3C workers to 2 concurrent threads. | Documented runtime adaptation with zero impact on single-agent inference accuracy. |
