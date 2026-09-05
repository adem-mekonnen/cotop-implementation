#!/usr/bin/env python3
"""
scripts/evaluate_fresh_training.py
Evaluates the freshly trained CoTOP A3C model from results/colab_training/cotop_trained.pt
across frozen evaluation realizations and generates dedicated audit artifacts in
results/colab_fresh_training_evaluation/.
"""

import os
import sys
import glob
import json
import yaml
import torch
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.a3c_agent import ActorCritic
from envs.entities import SimulationConfig
from envs.frozen_vec_env import FrozenVECEnv
from utils.checkpoint_io import load_checkpoint_strict, compute_file_sha256, compute_model_param_hash

def evaluate_fresh_training(
    checkpoint_path: str = "results/colab_training/cotop_trained.pt",
    output_dir: str = "results/colab_fresh_training_evaluation",
    realization_dir: str = "data/evaluation_realizations"
):
    os.makedirs(output_dir, exist_ok=True)
    assert os.path.exists(checkpoint_path), f"[FATAL] Trained checkpoint missing: {checkpoint_path}"

    with open("configs/paper_parameters.yaml", "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    sim_config = SimulationConfig(**config_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(input_dim=114, num_actions=7).to(device)
    load_checkpoint_strict(checkpoint_path, model, expected_algorithm="CoTOP", device=str(device))
    model.eval()

    realization_files = sorted([f for f in glob.glob(os.path.join(realization_dir, "realization_*.json")) if "manifest" not in os.path.basename(f).lower()])
    print(f"Evaluating fresh model across {len(realization_files)} realization files on device {device}...")

    eval_records = []
    action_counts = {i: 0 for i in range(7)}

    for r_file in realization_files:
        r_name = os.path.basename(r_file)
        env = FrozenVECEnv(sim_config, r_file)
        obs, _ = env.reset()

        delays = []
        energies = []
        collab_count = 0
        steps = 0

        while len(env.pending_tasks) > 0 and steps < 200:
            obs_t = torch.tensor(obs[:114], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(obs_t)
                action = torch.argmax(logits, dim=-1).item()

            action_counts[action] = action_counts.get(action, 0) + 1
            if action > 0:
                collab_count += 1
            steps += 1

            obs, reward, done, truncated, info = env.step(action)
            delays.append(info["delay"])
            energies.append(info["energy"])

        comp = len(env.completed_tasks)
        fail = len(env.failed_tasks)
        tot = comp + fail

        eval_records.append({
            "realization": r_name,
            "mean_delay_s": float(np.mean(delays)) if delays else 0.0,
            "mean_energy_j": float(np.mean(energies)) if energies else 0.0,
            "completion_ratio_pct": float((comp / max(tot, 1)) * 100.0),
            "collaboration_rate_pct": float((collab_count / max(steps, 1)) * 100.0),
            "steps": steps
        })

    df_eval = pd.DataFrame(eval_records)
    csv_path = os.path.join(output_dir, "fresh_cotop_evaluation.csv")
    df_eval.to_csv(csv_path, index=False)

    total_actions = sum(action_counts.values())
    action_dist = {f"action_{k}": round((v / max(total_actions, 1)) * 100.0, 2) for k, v in action_counts.items()}

    summary = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": compute_file_sha256(checkpoint_path),
        "model_param_hash": compute_model_param_hash(model),
        "total_realizations_evaluated": len(realization_files),
        "mean_delay_s": round(float(df_eval["mean_delay_s"].mean()), 4),
        "delay_std_s": round(float(df_eval["mean_delay_s"].std()), 4),
        "mean_energy_j": round(float(df_eval["mean_energy_j"].mean()), 4),
        "energy_std_j": round(float(df_eval["mean_energy_j"].std()), 4),
        "mean_completion_ratio_pct": round(float(df_eval["completion_ratio_pct"].mean()), 2),
        "mean_collaboration_rate_pct": round(float(df_eval["collaboration_rate_pct"].mean()), 2),
        "action_distribution_pct": action_dist
    }

    json_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_content = f"""# FRESHLY TRAINED COTOP A3C MODEL EVALUATION REPORT

**Document Identifier**: `results/colab_fresh_training_evaluation/FRESH_TRAINING_EVALUATION_REPORT.md`  
**Evaluated Checkpoint**: `{checkpoint_path}`  
**Checkpoint SHA-256**: `{summary['checkpoint_sha256']}`  
**Model Parameter Hash**: `{summary['model_param_hash']}`  
**Realizations Evaluated**: `{summary['total_realizations_evaluated']}`  

---

## 1. Metric Performance Summary

| Metric | Freshly Trained CoTOP | Physical Unit |
| :--- | :--- | :--- |
| **Mean Total Delay** | `{summary['mean_delay_s']:.4f} \pm {summary['delay_std_s']:.4f}` | seconds (s) |
| **Mean Dynamic Energy** | `{summary['mean_energy_j']:.4f} \pm {summary['energy_std_j']:.4f}` | Joules (J) |
| **Task Completion Ratio** | `{summary['mean_completion_ratio_pct']:.2f}%` | percentage (%) |
| **Collaboration Rate** | `{summary['mean_collaboration_rate_pct']:.2f}%` | percentage (%) |

---

## 2. Action Distribution

```json
{json.dumps(action_dist, indent=2)}
```

---

## 3. Comparison with Canonical Reference Checkpoint

- Canonical CoTOP Delay: `1.3566 s` | Freshly Trained CoTOP Delay: `{summary['mean_delay_s']:.4f} s`
- Canonical CoTOP Energy: `2.6747 J` | Freshly Trained CoTOP Energy: `{summary['mean_energy_j']:.4f} J`
- Canonical CoTOP Completion: `98.67%` | Freshly Trained CoTOP Completion: `{summary['mean_completion_ratio_pct']:.2f}%`
- Canonical CoTOP Collaboration: `89.04%` | Freshly Trained CoTOP Collaboration: `{summary['mean_collaboration_rate_pct']:.2f}%`
"""
    report_path = os.path.join(output_dir, "FRESH_TRAINING_EVALUATION_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[SUCCESS] Exported fresh training evaluation artifacts to '{output_dir}'.")
    return summary

if __name__ == "__main__":
    evaluate_fresh_training()
