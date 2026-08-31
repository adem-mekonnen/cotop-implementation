"""
experiments/realizations/runner.py

Execution Engine for Controlled Experiment Realizations (Stage 7).
Allows CoTOP, DDQN, Greedy, and Local to consume the EXACT same pre-materialized realization.
Strictly validates all 5 rejection gates before execution.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import torch

from envs.comm_model import compute_v2r_rate, compute_r2r_rate
from envs.comp_model import calculate_case1_standalone, calculate_case2_collaboration
from envs.entities import SimulationConfig
from experiments.realizations.schema import ExperimentRealization
from experiments.realizations.validator import RealizationValidator
from models.a3c_agent import ActorCritic
from models.baselines.ddqn_agent import DDQNAgent
from models.baselines.greedy import GreedyPolicy
from models.baselines.local import LocalPolicy


@dataclass
class RealizationRunResult:
    """Detailed results of an algorithm run over a controlled realization."""
    algorithm: str
    realization_id: str
    realization_hash: str
    geometry: str
    workload: int
    seed: int
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    completion_ratio: float
    mean_delay_s: float
    mean_energy_j: float
    comm_delay_s: float
    comp_delay_s: float
    wait_delay_s: float
    decisions: List[int]
    task_delays: List[float]
    task_energies: List[float]


class RealizationRunner:
    """
    Controlled Experiment Runner that feeds a pre-materialized realization
    to any supported algorithm policy under strict validation gates.
    """
    def __init__(self, config_path: str = "configs/paper_parameters.yaml"):
        self.config_path = config_path

    def run_algorithm(
        self,
        algorithm: str,
        realization: ExperimentRealization,
        agent_or_checkpoint: Optional[Any] = None,
        expected_geometry: Optional[str] = None,
        expected_workload: Optional[int] = None,
        expected_seed: Optional[int] = None,
        expected_env_fingerprint: Optional[str] = None,
        device: str = "cpu"
    ) -> RealizationRunResult:
        """
        Executes a policy (CoTOP, DDQN, Greedy, Local) on a single realization.
        Enforces all 5 rejection gates before starting.
        """
        # Validate realization against all 5 rejection gates
        RealizationValidator.validate(
            realization=realization,
            expected_geometry=expected_geometry,
            expected_workload=expected_workload,
            expected_seed=expected_seed,
            expected_env_fingerprint=expected_env_fingerprint
        )

        algo_lower = algorithm.lower().strip()
        num_rsus = len(realization.rsu_configurations)
        num_actions = num_rsus + 1
        obs_dim = 4 + (realization.workload * 4) + (num_rsus * 5)

        # 1. Initialize Algorithm Policy
        cotop_model = None
        ddqn_agent = None
        greedy_policy = None
        local_policy = None

        if algo_lower in ["cotop", "a3c"]:
            if isinstance(agent_or_checkpoint, ActorCritic):
                cotop_model = agent_or_checkpoint
            elif isinstance(agent_or_checkpoint, str):
                cotop_model = ActorCritic(input_dim=obs_dim, num_actions=num_actions, hidden_size=128)
                ckpt = torch.load(agent_or_checkpoint, map_location=device, weights_only=False)
                state_dict = ckpt.get("model_state_dict", ckpt)
                cotop_model.load_state_dict(state_dict)
            else:
                cotop_model = ActorCritic(input_dim=obs_dim, num_actions=num_actions, hidden_size=128)
            cotop_model.eval()

        elif algo_lower == "ddqn":
            if isinstance(agent_or_checkpoint, DDQNAgent):
                ddqn_agent = agent_or_checkpoint
            elif isinstance(agent_or_checkpoint, str):
                ddqn_agent = DDQNAgent(input_dim=obs_dim, num_actions=num_actions, hidden_dim=128, device=device)
                ddqn_agent.load_checkpoint(agent_or_checkpoint)
            else:
                ddqn_agent = DDQNAgent(input_dim=obs_dim, num_actions=num_actions, hidden_dim=128, device=device)

        elif algo_lower == "greedy":
            # Lightweight simulation config for GreedyPolicy
            sim_cfg = SimulationConfig(
                num_rsus=num_rsus,
                num_tasks_per_vehicle_range=[realization.workload, realization.workload],
                max_task_cpu=10.0
            )
            greedy_policy = GreedyPolicy(config=sim_cfg)

        elif algo_lower == "local":
            local_policy = LocalPolicy()
        else:
            raise ValueError(f"Unsupported algorithm '{algorithm}'. Supported: CoTOP, DDQN, Greedy, Local")

        # 2. Setup RSU Infrastructure and Queues
        rsus = []
        for r_cfg in realization.rsu_configurations:
            rsus.append({
                "id": r_cfg["rsu_id"],
                "loc": np.array(r_cfg["location"]),
                "cpu_f": float(r_cfg["cpu_capacity_f"]),
                "q_cycles": float(r_cfg["initial_queued_cycles"]),
                "p_tx": float(r_cfg["transmission_power_P_R"]),
                "b_v2r": float(r_cfg["bandwidth_v2r"]),
                "b_r2r": float(r_cfg["bandwidth_r2r"]),
                "range": float(r_cfg["comm_range"])
            })

        # Pre-index vehicle trajectories and mobility dwell times
        veh_traj_map = {vt["vehicle_id"]: vt for vt in realization.vehicle_trajectories}
        mob_map = {ms["vehicle_id"]: ms for ms in realization.mobility_states}

        # 3. Step Through Materialized Tasks
        task_delays = []
        task_energies = []
        comm_delays = []
        comp_delays = []
        wait_delays = []
        decisions = []
        completed_count = 0
        failed_count = 0

        map_scale = 200.0 if realization.geometry in ["grid_200m", "urban_manhattan"] else 2400.0

        for task_dict in realization.tasks:
            t_id = task_dict["task_id"]
            v_id = task_dict["vehicle_id"]
            gen_time = task_dict["generation_timestamp"]
            size_rho = task_dict["size_rho"]
            cpu_phi = task_dict["cpu_phi"]
            max_delay = task_dict["max_delay_d"]
            p_weight = task_dict["priority_weight"]

            # Lookup vehicle position at generation timestamp
            vt = veh_traj_map[v_id]
            points = vt["trajectory_points"]
            # Find closest waypoint in time
            closest_pt = min(points, key=lambda pt: abs(pt["timestamp"] - gen_time))
            v_pos = np.array([closest_pt["x"], closest_pt["y"]])
            v_speed = float(closest_pt["speed"])

            # Find primary (nearest) RSU
            dists = [np.linalg.norm(v_pos - r["loc"]) for r in rsus]
            primary_rsu_idx = int(np.argmin(dists))
            primary_dist = dists[primary_rsu_idx]

            # Dwell time
            dwell_dict = mob_map[v_id]["predicted_dwell_time_per_rsu"]
            t_stay = dwell_dict.get(str(primary_rsu_idx), 10.0)

            # Build state vector for policy input
            # 1. Ego vehicle (4)
            s_ego = [
                v_pos[0] / map_scale,
                v_pos[1] / map_scale,
                v_speed / 40.0,
                min(t_stay / 100.0, 1.0)
            ]
            # 2. Local tasks (workload * 4)
            s_tasks = [
                size_rho / 5.0e6,
                cpu_phi / 10.0e6,
                max_delay / 30.0,
                p_weight
            ] * realization.workload
            # 3. RSUs (num_rsus * 5)
            s_rsus = []
            for r in rsus:
                s_rsus.extend([
                    r["loc"][0] / map_scale,
                    r["loc"][1] / map_scale,
                    r["cpu_f"] / 4.0e9,
                    min(r["q_cycles"] / 1.0e9, 1.0),
                    r["p_tx"] / 100.0
                ])
            state_vec = np.array(s_ego + s_tasks + s_rsus, dtype=np.float32)

            # Build action feasibility mask (action 0 always valid; collaborative actions valid if in range)
            action_mask = np.zeros(num_actions, dtype=bool)
            action_mask[0] = True
            for r_idx, r in enumerate(rsus):
                # RSU collaborative action r_idx + 1
                if dists[r_idx] <= r["range"] * 2.0:
                    action_mask[r_idx + 1] = True

            # Select Action
            if cotop_model is not None:
                with torch.no_grad():
                    state_t = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
                    logits, _ = cotop_model(state_t)
                    # Apply action mask
                    mask_t = torch.tensor(action_mask, dtype=torch.bool, device=device)
                    logits = torch.where(mask_t, logits.squeeze(0), torch.tensor(-1e9, device=device))
                    action = torch.argmax(logits).item()

            elif ddqn_agent is not None:
                action = ddqn_agent.select_action(state_vec, action_mask=action_mask, deterministic=True)

            elif greedy_policy is not None:
                action = greedy_policy.select_action(state_vec)

            elif local_policy is not None:
                action = local_policy.select_action(state_vec)

            decisions.append(int(action))

            # Physical parameter constants from realization environment config
            env_cfg = realization.environment_configuration
            p_v = float(env_cfg.get("tx_power_vehicle", 0.01))
            p_r = float(env_cfg.get("tx_power_rsu", 100.0))
            p_comp = 50.0  # W
            noise_power = float(env_cfg.get("noise_power", 0.001))
            fixed_loss_k = float(env_cfg.get("fixed_loss_k", 1000.0))
            path_loss_factor = float(env_cfg.get("path_loss_factor", 2.0))

            # Apply Physical Execution Equations
            rsu1 = rsus[primary_rsu_idx]
            v2r_rate = compute_v2r_rate(
                distance=primary_dist,
                bandwidth_B=rsu1["b_v2r"],
                power_P_V=p_v,
                noise_power=noise_power,
                fixed_loss_k=fixed_loss_k,
                path_loss_factor=path_loss_factor
            )

            # Wait time at primary RSU
            t_wait_rsu1 = rsu1["q_cycles"] / rsu1["cpu_f"] if rsu1["cpu_f"] > 0 else 0.0

            if action == 0 or action - 1 == primary_rsu_idx:
                # Case 1: Standalone execution at primary RSU
                t_total, e_total = calculate_case1_standalone(
                    task_size_rho=size_rho,
                    task_cpu_phi=cpu_phi,
                    w_v2r=v2r_rate,
                    rsu_cpu_f=rsu1["cpu_f"],
                    power_v=p_v,
                    compute_power_rsu=p_comp,
                    t_wait=t_wait_rsu1
                )
                t_comm = (size_rho * 8.0) / v2r_rate if v2r_rate > 0 else 0.0
                t_comp = cpu_phi / rsu1["cpu_f"] if rsu1["cpu_f"] > 0 else 0.0
                t_wait = t_wait_rsu1
                # Update queue
                rsu1["q_cycles"] += cpu_phi

            else:
                # Case 2: Collaborative execution with secondary RSU
                sec_rsu_idx = action - 1
                rsu2 = rsus[sec_rsu_idx]
                r2r_dist = float(np.linalg.norm(rsu1["loc"] - rsu2["loc"]))
                r2r_rate = compute_r2r_rate(
                    distance=r2r_dist,
                    bandwidth_B=rsu1["b_r2r"],
                    power_P_R=p_r,
                    noise_power=noise_power,
                    fixed_loss_k=fixed_loss_k,
                    path_loss_factor=path_loss_factor
                )
                t_wait_rsu2 = rsu2["q_cycles"] / rsu2["cpu_f"] if rsu2["cpu_f"] > 0 else 0.0

                t_total, e_total = calculate_case2_collaboration(
                    task_size_rho=size_rho,
                    task_cpu_phi=cpu_phi,
                    w_v2r=v2r_rate,
                    w_r2r=r2r_rate,
                    rsu1_cpu_f=rsu1["cpu_f"],
                    rsu2_cpu_f=rsu2["cpu_f"],
                    t1_dwell_time=t_stay,
                    power_v=p_v,
                    tx_power_rsu1=p_r,
                    compute_power_rsu1=p_comp,
                    compute_power_rsu2=p_comp,
                    t_wait=t_wait_rsu2
                )
                t_comm = (size_rho * 8.0) / v2r_rate if v2r_rate > 0 else 0.0
                t_comp = max(t_total - t_comm - t_wait_rsu2, 0.0)
                t_wait = t_wait_rsu2
                # Update queue at RSU 2
                phi1 = min(rsu1["cpu_f"] * t_stay, cpu_phi)
                phi2 = max(cpu_phi - phi1, 0.0)
                rsu2["q_cycles"] += phi2

            # Deplete queues slightly over time
            for r in rsus:
                r["q_cycles"] = max(r["q_cycles"] - r["cpu_f"] * 0.1, 0.0)

            task_delays.append(float(t_total))
            task_energies.append(float(e_total))
            comm_delays.append(float(t_comm))
            comp_delays.append(float(t_comp))
            wait_delays.append(float(t_wait))

            if t_total <= max_delay:
                completed_count += 1
            else:
                failed_count += 1

        total_t = len(task_delays)
        comp_ratio = completed_count / max(total_t, 1)

        return RealizationRunResult(
            algorithm=algorithm,
            realization_id=realization.realization_id,
            realization_hash=realization.realization_hash,
            geometry=realization.geometry,
            workload=realization.workload,
            seed=realization.seed,
            total_tasks=total_t,
            completed_tasks=completed_count,
            failed_tasks=failed_count,
            completion_ratio=float(comp_ratio),
            mean_delay_s=float(np.mean(task_delays)),
            mean_energy_j=float(np.mean(task_energies)),
            comm_delay_s=float(np.mean(comm_delays)),
            comp_delay_s=float(np.mean(comp_delays)),
            wait_delay_s=float(np.mean(wait_delays)),
            decisions=decisions,
            task_delays=task_delays,
            task_energies=task_energies
        )
