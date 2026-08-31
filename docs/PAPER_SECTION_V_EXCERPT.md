# Paper Section V & VI Excerpts: Baselines, Tables IV-VI, Experiments

## Page 11

5550 IEEE TRANSACTIONS ON MOBILE COMPUTING, VOL. 25, NO. 4, APRIL 2026
TABLE III
SIMULA TIONPARAMETERS
by the collaborative ofﬂoading policy, and it can be up to a
maximum of 1. Based on this, the average task arrival rate λm
is at a maximum of 30 task/s. In time slot t, the maximum CPU
resource requirement of each generated taskϕi(t) is 10 Mcycles,
so the average computation demand per task ¯ϕ is a maximum
of 10 Mcycles. At the same time, the computational capacity of
each Rm is a minimum of 1GHz. Based on this, the minimum
s e r v i c er a t eo fRm μm is 100. Therefore, the utilization factor of
a single RSU χm will not be more than 0.3, which can maintain
the stability of the system.
V. E XPERIMENTS AND RESULTS
This section evaluates the performance of the proposed Co-
TOP through a systematic comparison with state-of-the-art base-
line methods under varying road conditions. We ﬁrst describe
the experimental scenario and evaluation metrics, followed by
an introduction of the comparative methods. Then, we provide
an objective assessment of CoTOP by comparing it with existing
approaches. Finally, we validate the effectiveness of the CoTOP
under changing road conditions in real-world scene.
A. Experimental Scenario and Evaluation Metrics
Our experiments combine real-world trajectory data with
simulation-based trafﬁc modeling to evaluate our task ofﬂoad-
ing strategy. First, our GA T-based mobility detection model is
trained using the ApolloScape trajectory dataset [32], a high-
precision dataset collected from complex urban road scenarios in
China. The dataset contains spatio-temporal motion trajectories
and category labels, enabling our model to accurately detect
vehicle dwell times within RSU coverage areas. Simulation
experiments were conducted on a Windows 10 64-bit operating
system using Python 3.8 and PyTorch 2.4.1. The environment
settings for the experiments are shown in the Table III.A
simulation map was downloaded from OpenStreetMap and used
to generate trafﬁc data ﬂows through the Simulation of Urban
Mobility (SUMO). This map is a real urban scene in Hangzhou
Province, China. The size of the map is 200 m × 200 m.
In this environment, 10 to 30 vehicles were deployed, each
randomly generating initial position coordinates and speeds.
Six RSUs were evenly distributed along the roadside, with all
RSUs having the same communication range and computational
capabilities. Next, we present the evaluation metrics employed
in our experiments.
r Average Delay: Average delay represents the average time
required to complete all tasks in the VEC system. It is
an important metric for reﬂecting the efﬁciency of task
ofﬂoading.
r Average Energy Consumption: Average energy consump-
tion represents the average energy required for the trans-
mission and processing of all tasks in the VEC system.
r Completion Ratio: Completion ratio represents the ratio of
the number of tasks completed within their tolerable time
to the total number of tasks in the VEC system. It is a
critical metric for evaluating the reliability and stability of
the ofﬂoading strategy.
B. Approaches for Comparison
To verify the effectiveness of the proposed CoTOP scheme,
we conducted a comparative analysis with several commonly
used baseline methods. The detailed descriptions of these base-
line methods are as follows:
r QRMP-DQN [33]: The Quantile Regression Multi-Pass
Deep Q-Network (QRMP-DQN) is an algorithm based on
DQN that jointly optimizes task ofﬂoading and resource
allocation to minimize the average energy consumption of
the system. DQN is a reinforcement learning algorithm that
combines deep learning with Q-learning.
r DDQN [34]: Double Deep Q-Network (DDQN) is an
enhanced DRL algorithm based on the DQN framework. Its
core mechanism employs deep neural networks to achieve
nonlinear approximation of the optimal action-value func-
tion, thereby addressing the overestimation bias inherent
in the conventional DQN framework.
r Local: This is an ofﬂoading method without any collabora-
tion between RSUs. In this approach, each task generated
by a vehicle is ofﬂoaded to an RSU and can only be
computed by the current RSU without being forwarded
to other RSUs.
r Greedy: In this method, each RSU makes collaborative
decisions based on a certain optimization objective by
selecting the most suitable task at the current moment.
This method does not consider the long-term impact of
collaborative decisions on the system and only selects
locally optimal solutions.
C. Hyperparameter Analysis
We conducted experimental studies to investigate the impact
of the learning rate on the convergence performance of our
proposed CoTOP scheme. As shown in Fig. 4, the conver-
gence performance of CoTOP improves signiﬁcantly as the
learning rate decreases, manifested by an increase in reward
values, reduced post-convergence ﬂuctuations, and enhanced
stability. Considering both convergence behavior and reward
performance, we ultimately set the learning rate to 0.0002.
Under the scenario of 10 vehicles and 25 tasks, we systemati-
cally investigated the parametersα andβ in (23). As shown in the
Authorized licensed use limited to: University of Electronic Science and Tech of China. Downloaded on August 07,2026 at 15:03:39 UTC from IEEE Xplore.  Restrictions apply. 

## Page 12

DU et al.: MOBILITY -AW ARE COLLABORA TIVE TASK OFFLOADING FOR PARALLEL TASKS IN VEHICULAR EDGE COMPUTING 5551
Fig. 4. Convergence of CoTOP with different learning rates.
Fig. 5. The impact of hyperparameter α.
Fig. 6. Convergence of different task ofﬂoading methods.
Fig. 5, the average delay initially decreases then increases with
α, reaching the minimum value at α =0 .3 , while signiﬁcantly
rising to 16s asα increases from 0.2 to 0.8. The task completion
ratio peaks at 0.88 whenα =0 .4 . Energy consumption remains
stable around 25J near α =0 .3, but shows a marked increase
with higher α values. Comprehensive evaluation suggests α =
0.3 achieves optimal balanced performance for task ofﬂoading.
D. Performance Comparison
We ﬁrst analyze the convergence of the CoTOP . As shown in
Fig. 6, it can be observed that all three methods converge to a
stable reward value after a relatively small number of iterations.
TABLE IV
COMPARISON OF AVERAGE DELA YAMONG DIFFERENT METHODS UNDER
VARYING NUMBERS OF TASKS
TABLE V
COMPARISON OF COMPLETION RAT I OAMONG DIFFERENT METHODS UNDER
VARYING NUMBERS OF TASKS
However, the reward value achieved by the Greedy method is rel-
atively low. Although the QRMP-DQN method achieves a higher
reward value, its reward exhibits signiﬁcant ﬂuctuations in the
later stages of training. While DDQN demonstrates relatively
stable convergence, its convergence speed is slower compared
to the CoTOP . In contrast, the proposed CoTOP demonstrates
the best performance in terms of reward stability and maximum
value, indicating its superior robustness in dynamic and complex
environment.
In vehicular network environments, the number of tasks has
a signiﬁcant impact on task ofﬂoading efﬁciency. Tables IV
and V present the average delay and completion ratio under
different task numbers. As shown in the Table IV, the average
delay of all methods increases with the growing number of tasks.
Notably, the DDQN method demonstrates superior performance
compared to traditional DQN methods, beneﬁting from its dou-
ble Q-learning mechanism that mitigates value overestimation.
However, DDQN still underperforms relative to the CoTOP , as
the actor-critic architecture in A3C framework enables better
adaptation to high-dimensional state spaces in dynamic envi-
ronments. The CoTOP scheme consistently achieves the lowest
delay with minimal ﬂuctuation as task numbers increase. This
stability stems from CoTOP’s efﬁcient task allocation strat-
egy and dynamic policy adjustments that effectively handle
environmental complexity. The QRMP-DQN method exhibits
slightly higher delay than CoTOP , primarily due to its limited
adaptability in high-dimensional action spaces. Both Greedy
and Local method demonstrate substantially higher delay than
CoTOP and QRMP-DQN, with the Local approach showing
the worst performance due to its exclusive reliance on RSU
computational resources.
The task completion ratio, as illustrated in Table V, align
closely with the delay analysis, with CoTOP consistently outper-
forming other methods by achieving the highest task completion
ratio. QRMP-DQN and DDQN follow with moderate yet reliable
performance, while Greedy and Local methods exhibit notably
Authorized licensed use limited to: University of Electronic Science and Tech of China. Downloaded on August 07,2026 at 15:03:39 UTC from IEEE Xplore.  Restrictions apply. 

## Page 13

5552 IEEE TRANSACTIONS ON MOBILE COMPUTING, VOL. 25, NO. 4, APRIL 2026
Fig. 7. Performance comparison under different transmission rates.
Fig. 8. Performance comparison under different RSU computing capacities.
lower completion ratio, underscoring their inherent limitations
in handling complex task allocation scenarios.
In addition to the impact of task numbers on task ofﬂoading
efﬁciency, network conditions in vehicular environments also
play a signiﬁcant role. Fig. 7 illustrates the impact of trans-
mission rate on task ofﬂoading efﬁciency. From Fig. 7(a),i t
can be observed that as the transmission rate increases, the
average delay of all methods decreases signiﬁcantly. Among
the methods, the CoTOP achieves the lowest average delay
due to its efﬁcient task allocation strategy. The DDQN method
demonstrates competitive performance, achieving lower delay
than both Greedy and Local methods, though slightly higher than
CoTOP and QRMP-DQN. In contrast, the Local method exhibits
the highest average delay, as it does not perform collaborative
ofﬂoading, leading to a lower task completion ratio. The Greedy
method reduces queuing delays by ofﬂoading tasks to the RSU
with the least load. However, due to the mobility of vehicles, task
ofﬂoading reliability is compromised, causing task failures and
higher delays compared to CoTOP , QRMP-DQN, and DDQN.
The CoTOP method outperforms QRMP-DQN and DDQN in
task allocation and adjustment capabilities, resulting in lower
average delays.
Fig. 7(b) shows the task completion ratio under different
transmission rates. The CoTOP , QRMP-DQN, and DDQN meth-
ods exhibit higher completion ratio due to their exploration
capabilities, which allow them to learn better ofﬂoading strate-
gies. DDQN’s dual-network architecture enables stable Q-value
estimation, contributing to its reliable task completion perfor-
mance across varying transmission rates. In contrast, the Greedy
and Local methods lack exploration capabilities, leading to
relatively lower task completion ratio. Finally, Fig. 7(c) reveals
that as the transmission rate increases, the average energy con-
sumption of all algorithms shows a decreasing trend. This is
because higher transmission rates reduce the task transmission
time, thereby lowering the corresponding transmission energy
consumption. The DDQN and QRMP-DQN methods signiﬁ-
cantly outperform the Greedy and Local approaches in terms of
energy efﬁciency, while CoTOP consistently demonstrates the
lowest energy consumption.
We conducted comparative experiments to analyze the impact
of RSU computational capacity on ofﬂoading efﬁciency. As
shown in Fig. 8, the average delay of all methods decreases with
increasing RSU computing capacities. The Local method consis-
tently exhibits the highest delay, though its downward trend re-
mains relatively pronounced. The Greedy method demonstrates
slightly lower delays than the Local approach but still underper-
forms compared to QRMP-DQN. The DDQN method shows
intermediate characteristics in delay performance, positioned
between the Greedy and QRMP-DQN methods. This occurs
because the Greedy method fails to adapt to dynamic action
space variations, leading to suboptimal ofﬂoading decisions
under high-load conditions. In contrast, the CoTOP method
maintains the lowest delay consistently through its optimization
capabilities in dynamic task ofﬂoading scenarios. Regarding task
Authorized licensed use limited to: University of Electronic Science and Tech of China. Downloaded on August 07,2026 at 15:03:39 UTC from IEEE Xplore.  Restrictions apply. 

## Page 14

DU et al.: MOBILITY -AW ARE COLLABORA TIVE TASK OFFLOADING FOR PARALLEL TASKS IN VEHICULAR EDGE COMPUTING 5553
Fig. 9. Performance comparison under different numbers of vehicles.
TABLE VI
RESULTS OF ABLA TIONEXPERIMENT
completion and energy efﬁciency metrics, the DDQN method
outperforms conventional rule-based approaches but remains
slightly inferior to enhanced deep reinforcement learning meth-
ods. While QRMP-DQN can partially utilize RSU resources
in complex environments, its limitations in high-dimensional
action spaces constrain overall performance, resulting in higher
average delays than CoTOP . The experimental results on com-
pletion ratio and energy consumption further validate the effec-
tiveness of CoTOP in optimizing task ofﬂoading efﬁciency.
Fig. 9 illustrates the impact of the number of vehicles on task
ofﬂoading efﬁciency in vehicular network environments. As the
number of vehicles increases, the average task completion delay
rises for all methods, while the task completion ratio gradually
decreases. Additionally, the CoTOP , DDQN and QRMP-DQN
methods exhibit signiﬁcant ﬂuctuations in both average delay
and task completion ratio, whereas the Greedy and Local meth-
ods show relatively minor variations. The primary reason for this
phenomenon is that CoTOP , DDQN and QRMP-DQN methods
frequently adjust their task allocation strategies in response to
dynamic task distributions and the growing number of vehicles,
leading to greater performance ﬂuctuations. In contrast, the
Greedy and Local methods, which rely on simpler strategies,
handle task ofﬂoading more consistently but lack the efﬁciency
of more advanced methods. Despite the ﬂuctuations, the CoTOP
consistently achieves the lowest average delay and highest task
completion ratio, outperforming all other methods.
Finally, we conducted ablation experiments on individual
modules in the proposed CoTOP method, where MD denotes
the mobility detection module, TP represents the task priority
module, and CO indicates the collaborative ofﬂoading mod-
ule. The experimental results are illustrated in the Table VI.
As demonstrated, each module plays an important role in the
performance of task ofﬂoading, such as delay and energy con-
sumption. After removing the mobile detection module, the
average delay and energy consumption of completing tasks
Fig. 10. The simulation map of Hangzhou road network.
have signiﬁcantly increased, while the task completion ratio has
signiﬁcantly decreased, indicating the importance of the mobile
detection module for task ofﬂoading. After removing the task
priority module, the efﬁciency in task scheduling decreases,
resulting in a certain degree of performance degradation. If
collaboration is not carried out, the experimental results are
most severely affected, indicating that collaborative ofﬂoading is
crucial for optimizing collaboration efﬁciency. Through ablation
experiments, we have veriﬁed the important role of each module
in improving task ofﬂoading efﬁciency and resource utilization.
E. Evaluation of CoTOP in Real-World Road Scene
The experiments and result analysis in the above subsections
are derived on our synthetic road environment, and in order
to verify the practical feasibility of the CoTOP method, we
obtained the local road network data of Hangzhou through
“OpenStreet Map”, and the simulation map is shown in Fig. 10,
which contains 5 main roads (marked in red) and 8 intersections
(marked in yellow). We used SUMO to conduct simulation
experiments, deployed RSUs uniformly at each intersection, and
set up more than 100 vehicles to conduct task ofﬂoading tests to
simulate the urban dense road scenarios.
Under the aforementioned experimental scenario, we calcu-
lated the average delay, energy consumption, and completion
ratio of all tasks, and the corresponding results are presented
in Fig. 11. As illustrated in the Fig. 11, it can be observed
that in real road environments, when the number of vehicles
exceeds 100, the latency and energy consumption of the Local
method increase signiﬁcantly, while the task completion ratio
Authorized licensed use limited to: University of Electronic Science and Tech of China. Downloaded on August 07,2026 at 15:03:39 UTC from IEEE Xplore.  Restrictions apply. 

## Page 15

5554 IEEE TRANSACTIONS ON MOBILE COMPUTING, VOL. 25, NO. 4, APRIL 2026
Fig. 11. Performance comparison under different numbers of vehicles on real city roads.
drops sharply. The underlying reason is that once the task scale
reaches a certain level, the computing capability of the vehicles
themselves can no longer handle such a heavy computational
load, resulting in some tasks failing to be completed within
the maximum tolerable time. The Greedy method prioritizes
ofﬂoading tasks to vehicles or RSUs with lighter loads. As a
result, when the number of tasks increases, it does not suffer
from increased task latency or task failures due to insufﬁcient
computing resources. For the other three reinforcement learning
algorithms, their ability to dynamically perceive environmental
states and update policies in real time enables them to outperform
the aforementioned two methods in terms of both latency and en-
ergy consumption. Among them, CoTOP consistently achieves
the best performance; while DDQN exhibits higher latency and
energy consumption compared to CoTOP , it still outperforms the
QRMP-DQN algorithm. This is because, compared with DQN,
DDQN can effectively mitigate the overestimation of Q-values,
thereby achieving better decision-making accuracy and stability.
VI. D ISCUSSION
In this section, we analyze the feasibility and limitations of
CoTOP and outline directions for future work.
Integration feasibility: We assume that CoTOP is deployed
on each RSU, enabling the RSUs to perceive the surrounding
environmental state and make optimal task ofﬂoading decisions.
However, existing VEC systems typically consist of heteroge-
neous hardware platforms and software protocols, and the com-
putational capacity, storage resources, and communication inter-
faces of different edge computing nodes can vary signiﬁcantly,
posing challenges for modular integration. In future work, we
plan to implement CoTOP as a ﬂexible, microservice-based
architecture module that can be adaptively deployed on RSUs,
in-vehicle edge nodes, or other MEC servers, depending on the
available resources of each edge node.
Communication model: When the communication conditions
are simple and stable, our current communication model can re-
duce the complexity of analysis and experiments. However, this
model does not fully account for the various interference factors
present in real wireless environments. In future work, we plan to
incorporate more realistic communication characteristics, such
as time-varying channels, path loss, and interference, to more
accurately simulate the effects of task transmission latency and
data packet loss on system performance. This approach will
allow us to evaluate the robustness and adaptability of CoTOP in
practical network environments, providing more reliable theo-
retical and experimental support for its deployment in large-scale
vehicular edge computing scenarios.
Performance trade-offs: In vehicular edge computing en-
vironments, task execution latency remains the most critical
performance metric, as many tasks, which include autonomous
driving decisions, emergency braking, and collision warnings,
have stringent real-time requirements. While low latency is
the primary objective, system design must also balance energy
consumption and task completion rate. Future research will con-
sider scenario-speciﬁc factors, including vehicle density, task
types, edge node load, and network conditions, to quantitatively
evaluate and optimize energy efﬁciency and task completion
rate while maintaining low latency, thereby improving overall
system performance and user experience.
VII. C ONCLUSION
In this paper, we propose a novel CoTOP scheme that achieves
joint optimization of task execution delay and energy con-
sumption in dynamic VEC environments. CoTOP effectively
addresses communication instability caused by high-speed ve-
hicle mobility, thereby resolving issues of task interruptions or
failures. Additional, by establishing differentiated task priori-
ties, CoTOP effectively mitigates latency increase and energy
consumption growth caused by imbalanced resource alloca-
tion. CoTOP signiﬁcantly enhances adaptability to complex and
highly dynamic VEC environments, delivering an efﬁcient and
reliable solution for real-time services.
REFERENCES
[1] Q. Gao et al., “Optimization of models and strategies for computation
ofﬂoading in the Internet of V ehicles: Efﬁciency and trust,” IEEE Trans.
Mobile Comput., vol. 24, no. 4, pp. 3372–3389, Apr. 2025.
[2] S. Shen, G. Shen, Z. Dai, K. Zhang, X. Kong, and J. Li, “Asynchronous
federated deep-reinforcement-learning-based dependency task ofﬂoading
for UA V -assisted vehicular networks,”IEEE Internet Things J. , vol. 11,
no. 19, pp. 31561–31574, Oct. 2024.
Authorized licensed use limited to: University of Electronic Science and Tech of China. Downloaded on August 07,2026 at 15:03:39 UTC from IEEE Xplore.  Restrictions apply. 

## Page 16

DU et al.: MOBILITY -AW ARE COLLABORA TIVE TASK OFFLOADING FOR PARALLEL TASKS IN VEHICULAR EDGE COMPUTING 5555
[3] J. Wang, C. Jiang, K. Zhang, T. Q. S. Quek, Y . Ren, and L. Hanzo,
“V ehicular sensing networks in a smart city: Principles, technologies
and applications,” IEEE Wireless Commun., vol. 25, no. 1, pp. 122–132,
Feb. 2018.
[4] J. Yin, W. Rao, Y . Xiao, and K. Tang, “Cooperative path plan-
ning with asynchronous multiagent reinforcement learning,” IEEE
Trans. Mobile Comput. , vol. 24, no. 6, pp. 5016–5030, Jun. 2025,
doi: 10.1109/TMC.2025.3526979.
[5] S. Tian, X. Zhu, B. Feng, Z. Zheng, H. Liu, and Z. Li, “Partial ofﬂoading
strategy based on deep reinforcement learning in the Internet of V ehicles,”
IEEE Trans. Mobile Comput. , vol. 24, no. 7, pp. 6517–6531, Jul. 2025,
doi: 10.1109/TMC.2025.3543976.
[6] Z. Nan, S. Zhou, Y . Jia, and Z. Niu, “Joint task ofﬂoading and resource
allocation for vehicular edge computing with result feedback delay,”IEEE
Trans. Wireless Commun., vol. 22, no. 10, pp. 6547–6561, Oct. 2023.
[7] K. Li et al., “Computation ofﬂoading in resource-constrained multi-access
edge computing,”IEEE Trans. Mobile Comput., vol. 23, no. 11, pp. 10665–
10677, Nov. 2024.
[8] J. Wang, Z. Li, H. Liu, T. Qiu, and H. Luo, “A trust-based computa-
tion ofﬂoading framework in mobile cloud-edge computing networks,”
IEEE Trans. Mobile Comput. , vol. 24, no. 6, pp. 5370–5385, Jun. 2025,
doi: 10.1109/TMC.2025.3530480.
[9] H. Jiang, J. Cai, Z. Xiao, K. Y ang, H. Chen, and J. Liu, “V ehicle-
assisted service caching for task ofﬂoading in vehicular edge computing,”
IEEE Trans. Mobile Comput. , vol. 24, no. 7, pp. 6688–6700, Jul. 2025,
doi: 10.1109/TMC.2025.3545444.
[10] P . Dai, Y . Huang, K. Hu, X. Wu, H. Xing, and Z. Y u, “Meta reinforcement
learning for multi-task ofﬂoading in vehicular edge computing,” IEEE
Trans. Mobile Comput., vol. 23, no. 3, pp. 2123–2138, Mar. 2024.
[11] L. Zhao et al., “MESON: A mobility-aware dependent task ofﬂoading
scheme for urban vehicular edge computing,” IEEE Trans. Mobile Com-
put., vol. 23, no. 5, pp. 4259–4272, May 2024.
[12] K. Gu, Z. Liu, and W. Jia, “Location-aware reliable task cooperative-
computation scheme under fog computing-based IoVs,” IEEE Trans.
Intell. Transp. Syst., vol. 26, no. 1, pp. 425–442, Jan. 2025.
[13] Z. Li, C. Y ang, X. Huang, W. Zeng, and S. Xie, “CoOR: Collaborative task
ofﬂoading and service caching replacement for vehicular edge computing
networks,” IEEE Trans. V eh. Technol. , vol. 72, no. 7, pp. 9676–9681,
Jul. 2023.
[14] W. Fan, Y . Zhang, G. Zhou, and Y . Liu, “Deep reinforcement learning-
based task ofﬂoading for vehicular edge computing with ﬂexible RSU-
RSU cooperation,” IEEE Trans. Intell. Transp. Syst. , vol. 25, no. 7,
pp. 7712–7725, Jul. 2024.
[15] S. Pratap, P . Dass, and S. Misra, “CoTEV: Trustworthy and cooperative task
execution in Internet of V ehicles,”IEEE Trans. Mobile Comput. , vol. 23,
no. 4, pp. 2915–2926, Apr. 2024.
[16] X. Zhang, C. Wang, Y . Zhu, J. Cao, and T. Liu, “Multi-agent deep rein-
forcement learning with trajectory prediction for task migration-assisted
computation ofﬂoading,” IEEE Trans. Mobile Comput. , vol. 24, no. 7,
pp. 5839–5856, Jul. 2025, doi: 10.1109/TMC.2025.3539945.
[17] W. Zhao, Y . Cheng, Z. Liu, X. Wu, and N. Kato, “Asynchronous DRL-
based multi-hop task ofﬂoading in RSU-assisted IoV networks,” IEEE
Trans. Cogn. Commun. Netw., vol. 11, no. 1, pp. 546–555, Feb. 2025.
[18] X. Chen, J. Cao, Y . Sahni, M. Zhang, Z. Liang, and L. Y ang, “Mobility-
aware dependent task ofﬂoading in edge computing: A digital twin-assisted
reinforcement learning approach,” IEEE Trans. Mobile Comput. , vol. 24,
no. 4, pp. 2979–2994, Apr. 2025.
[19] H. Li, C. Chen, H. Shan, P . Li, Y . C. Chang, and H. Song, “Deep
deterministic policy gradient-based algorithm for computation ofﬂoading
in IoV,” IEEE Trans. Intell. Transp. Syst. , vol. 25, no. 3, pp. 2522–2533,
Mar. 2024.
[20] P . Hou, X. Jiang, Z. Lu, B. Li, and Z. Wang, “Joint computation ofﬂoading
and resource allocation based on deep reinforcement learning in C-V2X
edge computing,” Appl. Intell., vol. 53, no. 19, pp. 22446–22466, 2023.
[21] Y . Guo et al., “Deep deterministic policy gradient-based intelligent task
ofﬂoading for vehicular computing with priority experience playback,”
IEEE Trans. V eh. Technol., vol. 73, no. 7, pp. 10655–10667, Jul. 2024.
[22] N. Fofana, A. B. Letaifa, and A. Rachedi, “Intelligent task ofﬂoading in
vehicular networks: A deep reinforcement learning perspective,” IEEE
Trans. V eh. Technol., vol. 74, no. 1, pp. 201–216, Jan. 2025.
[23] X. Fan, W. Gu, C. Long, C. Gu, and S. He, “Optimizing task ofﬂoading and
resource allocation in vehicular edge computing based on heterogeneous
cellular networks,” IEEE Trans. V eh. Technol., vol. 73, no. 5, pp. 7175–
7187, May 2024.
[24] X. Chen, S. Hu, C. Y u, Z. Chen, and G. Min, “Real-time ofﬂoading
for dependent and parallel tasks in cloud-edge environments using deep
reinforcement learning,”IEEE Trans. Parallel Distrib. Syst., vol. 35, no. 3,
pp. 391–404, Mar. 2024.
[25] Y . Li, X. Ge, B. Lei, X. Zhang, and W. Wang, “Joint task partitioning
and parallel scheduling in device-assisted mobile edge networks,” IEEE
Internet Things J., vol. 11, no. 8, pp. 14058–14075, Apr. 2024.
[26] S. Munawar, Z. Ali, M. Waqas, S. Tu, S. A. Hassan, and G. Abbas,
“Cooperative computational ofﬂoading in mobile edge computing for
vehicles: A model-based DNN approach,” IEEE Trans. V eh. Technol. ,
vol. 72, no. 3, pp. 3376–3391, Mar. 2023.
[27] X. Xu, C. Y ang, M. Bilal, W. Li, and H. Wang, “Computation ofﬂoading for
energy and delay trade-offs with trafﬁc ﬂow prediction in edge computing-
enabled IoV,”IEEE Trans. Intell. Transp. Syst., vol. 24, no. 12, pp. 15613–
15623, Dec. 2023.
[28] Z. Xiao et al., “Multi-objective parallel task ofﬂoading and content caching
in D2D-aided MEC networks,” IEEE Trans. Mobile Comput. , vol. 22,
no. 11, pp. 6599–6615, Nov. 2023.
[29] X. Wang et al., “Augmented Intelligence of Things for priority-aware task
ofﬂoading in vehicular edge computing,” IEEE Internet Things J., vol. 11,
no. 22, pp. 36002–36013, Nov. 2024.
[30] X. Zhu, Y . Luo, A. Liu, M. Z. A. Bhuiyan, and S. Zhang, “Multiagent
deep reinforcement learning for vehicular computation ofﬂoading in IoT,”
IEEE Internet Things J. , vol. 8, no. 12, pp. 9763–9773, Jun. 2021.
[31] H. Shen, K. Zhang, M. Hong, and T. Chen, “Towards understanding
asynchronous advantage actor-critic: Convergence and linear speedup,”
IEEE Trans. Signal Process., vol. 71, pp. 2579–2594, 2023.
[32] Y . Ma, X. Zhu, S. Zhang, R. Y ang, W. Wang, and D. Manocha, “TrafﬁcPre-
dict: Trajectory prediction for heterogeneous trafﬁc-agents,” inProc. AAAI
Conf. Artif. Intell., 2019, pp. 6120–6127.
[33] L. Guo, J. Jia, J. Chen, and X. Wang, “QRMP-DQN empowered task of-
ﬂoading and resource allocation for the STAR-RIS assisted MEC systems,”
IEEE Trans. V eh. Technol., vol. 74, no. 1, pp. 1252–1266, Jan. 2025.
[34] H. Zhai, X. Zhou, H. Zhang, and D. Y uan, “Delay minimization in hybrid
edge computing networks: A DDQN-based task ofﬂoading approach,”
IEEE Trans. V eh. Technol., vol. 73, no. 10, pp. 15098–15108, Oct. 2024.
Jiaxin Du received the PhD degree from the School
of Software, Dalian University of Technology, China,
in 2023. She is currently a lecturer with the College
of Computer Science and Technology, Zhejiang Uni-
versity of Technology, China. Her current research
interests include edge intelligence, security for wire-
less sensor networks and intelligent transportation
systems.
Jinfan Zhang received the BE degree from Wenzhou
University, Wenzhou, China, in 2023. He is currently
working toward the master’s degree in computer sci-
ence and technology with the School of Computer
Science, Zhejiang University of Technology, China.
His research interests include edge computing and
intelligent transportation systems.
Authorized licensed use limited to: University of Electronic Science and Tech of China. Downloaded on August 07,2026 at 15:03:39 UTC from IEEE Xplore.  Restrictions apply. 

