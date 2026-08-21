import argparse
import yaml
import torch
import numpy as np
import torch.nn.functional as F

from envs.vec_env import VECEnv
from envs.entities import SimulationConfig
from models.a3c_agent import ActorCritic
from models.baselines.heuristic import local_only_policy, greedy_queue_policy


def load_config(path="configs/simulation.yaml") -> SimulationConfig:
    with open(path, "r") as f:
        config_data = yaml.safe_load(f)
    return SimulationConfig(**config_data)


def run_episodes(env, policy_fn, episodes: int):
    total_delay, total_energy = 0.0, 0.0
    completed_tasks, total_tasks = 0, 0

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = policy_fn(state, env)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if info:
                total_delay += info.get("delay", 0.0)
                total_energy += info.get("energy", 0.0)
                if not info.get("missed_deadline", True):
                    completed_tasks += 1
                total_tasks += 1

            state = next_state

    if total_tasks == 0:
        return 0.0, 0.0, 0.0
    return (total_delay / total_tasks, total_energy / total_tasks, completed_tasks / total_tasks)


def make_agent_policy(agent_model):
    def policy(state, env):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            policy_logits, _ = agent_model(state_tensor)
            probs = F.softmax(policy_logits, dim=-1)
            return torch.argmax(probs, dim=-1).item()
    return policy


def greedy_policy(state, env):
    return greedy_queue_policy(state, env.rsus)


def local_policy(state, env):
    return local_only_policy(state)


def evaluate(args):
    config = load_config()
    print(f"Starting Evaluation for {args.episodes} episodes per method...")
    print(f"Ablations -> no_mobility={args.no_mobility}, no_priority={args.no_priority}\n")

    cotop_env = VECEnv(config=config, port=1,
                        use_mobility_model=not args.no_mobility,
                        use_priority=not args.no_priority)
    input_dim = cotop_env.observation_space.shape[0]
    num_actions = cotop_env.action_space.n

    agent_model = ActorCritic(input_dim, num_actions)
    if args.a3c_model:
        try:
            agent_model.load_state_dict(torch.load(args.a3c_model, map_location="cpu"))
            print(f"Loaded A3C model from {args.a3c_model}")
        except FileNotFoundError:
            print(f"A3C model not found at {args.a3c_model}. Using untrained weights.")
    agent_model.eval()

    results = {}
    results["CoTOP"] = run_episodes(cotop_env, make_agent_policy(agent_model), args.episodes)
    cotop_env.close()

    greedy_env = VECEnv(config=config, port=2, use_mobility_model=False, use_priority=False)
    results["Greedy"] = run_episodes(greedy_env, greedy_policy, args.episodes)
    greedy_env.close()

    local_env = VECEnv(config=config, port=3, use_mobility_model=False, use_priority=False)
    results["Local"] = run_episodes(local_env, local_policy, args.episodes)
    local_env.close()

    print("\n" + "=" * 65)
    print(f"{'Method':<10}{'Avg Delay (s)':>18}{'Completion Ratio':>20}{'Avg Energy (J)':>17}")
    print("=" * 65)
    for name, (delay, energy, ratio) in results.items():
        print(f"{name:<10}{delay:>18.4f}{ratio * 100:>19.2f}%{energy:>17.4f}")
    print("=" * 65)
    print("Note: DDQN / QRMP-DQN baselines are empty stubs in models/baselines/ and are not included above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CoTOP DRL Agent")
    parser.add_argument("--a3c_model", type=str, default="results/checkpoints/a3c_agent.pth")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--no_mobility", action="store_true")
    parser.add_argument("--no_priority", action="store_true")
    args = parser.parse_args()
    evaluate(args)