#!/usr/bin/env python3
"""
scripts/verify_colab_gpu.py

Google Colab GPU Verification & Smoke Test Script for Phase 2 CoTOP Reproduction.

Verifies:
1. Protected physics model SHA-256 integrity.
2. CUDA availability and GPU device information.
3. GPU tensor operations and smoke matrix multiplications.
4. Minimal DDQN training and evaluation step.
5. Minimal CoTOP A3C and Mobility GAT-GRU forward pass.
6. Evaluation determinism.
"""

import sys
import os

# Ensure root workspace is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import hashlib
import subprocess
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# Expected Protected Physics Hashes
COMM_MODEL_SHA256 = "041e41061d02c7a5a7bc9488adf2bc49472177215730bd8a23c5ff2437431431"
COMP_MODEL_SHA256 = "dd9f58df710f709d536000bb4047d2ad6000cf37b1a49f4e1f0e8d883b856bff"

def get_git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "UNKNOWN"

def verify_physics_hashes():
    print("\n--- 1. Protected Physics Hashes Verification ---")
    comm_path = "envs/comm_model.py"
    comp_path = "envs/comp_model.py"
    
    if not os.path.exists(comm_path) or not os.path.exists(comp_path):
        raise FileNotFoundError("Protected physics files missing!")
        
    comm_hash = hashlib.sha256(open(comm_path, "rb").read()).hexdigest()
    comp_hash = hashlib.sha256(open(comp_path, "rb").read()).hexdigest()
    
    print(f"comm_model.py: {comm_hash}")
    print(f"comp_model.py: {comp_hash}")
    
    if comm_hash != COMM_MODEL_SHA256:
        raise ValueError(f"CRITICAL: comm_model.py hash mismatch! Expected {COMM_MODEL_SHA256}, got {comm_hash}")
    if comp_hash != COMP_MODEL_SHA256:
        raise ValueError(f"CRITICAL: comp_model.py hash mismatch! Expected {COMP_MODEL_SHA256}, got {comp_hash}")
        
    print("[PASS] Protected physics files verified intact.")
    return comm_hash, comp_hash

def check_environment(allow_cpu=False):
    print("\n--- 2. Hardware & Environment Diagnostics ---")
    python_ver = sys.version.split()[0]
    pytorch_ver = torch.__version__
    git_sha = get_git_sha()
    cuda_available = torch.cuda.is_available()
    cuda_ver = torch.version.cuda if hasattr(torch.version, "cuda") and torch.version.cuda else "N/A"
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "NO GPU"
    
    print(f"Python Version:   {python_ver}")
    print(f"PyTorch Version:  {pytorch_ver}")
    print(f"Git Commit SHA:   {git_sha}")
    print(f"CUDA Available:   {cuda_available}")
    print(f"CUDA Version:     {cuda_ver}")
    print(f"GPU Model:        {gpu_name}")
    
    if not cuda_available:
        if allow_cpu:
            print("[WARN] CUDA is not available. Proceeding in CPU diagnostic mode (--allow-cpu enabled).")
            device = torch.device("cpu")
        else:
            print("[ERROR] CUDA IS NOT AVAILABLE!")
            print("Google Colab requires a GPU runtime (Runtime -> Change runtime type -> T4/V100/A100 GPU).")
            sys.exit(1)
    else:
        device = torch.device("cuda:0")
        print(f"[PASS] CUDA verified active on device: {device} ({gpu_name})")
        
    return {
        "python": python_ver,
        "pytorch": pytorch_ver,
        "git_sha": git_sha,
        "cuda_available": cuda_available,
        "cuda_ver": cuda_ver,
        "gpu_name": gpu_name,
        "device": device
    }

def run_gpu_smoke_test(device):
    print(f"\n--- 3. Device Smoke Test ({device}) ---")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    c = torch.matmul(a, b)
    norm = c.norm().item()
    assert not np.isnan(norm) and not np.isinf(norm)
    print(f"[PASS] 1000x1000 Matrix Multiplication completed on {device}. Output Frobenius norm: {norm:.4f}")

def run_ddqn_smoke_test(device):
    print(f"\n--- 4. Minimal DDQN Baseline Test ({device}) ---")
    from models.baselines.ddqn_agent import DDQNAgent, QNetwork
    
    # Check QNetwork forward pass
    q_net = QNetwork(input_dim=114, num_actions=7).to(device)
    state = torch.randn(8, 114, device=device)
    q_vals = q_net(state)
    assert q_vals.shape == (8, 7)
    
    # Check DDQNAgent initialization and transition store
    agent = DDQNAgent(input_dim=114, num_actions=7, batch_size=4, device=device.type)
    for _ in range(10):
        s = np.random.randn(114).astype(np.float32)
        s_next = np.random.randn(114).astype(np.float32)
        mask = np.ones(7, dtype=bool)
        agent.store_transition(s, action=0, reward=1.0, next_state=s_next, done=False, next_action_mask=mask)
        
    loss = agent.update()
    print(f"[PASS] DDQNAgent forward and update verified on {device}. Loss: {loss:.6f}")

def run_cotop_smoke_test(device):
    print(f"\n--- 5. Minimal CoTOP Agent & Mobility GAT Test ({device}) ---")
    from models.a3c_agent import ActorCritic
    from models.mobility_gat import MobilityGAT_GRU
    
    # 1. Actor-Critic forward test
    ac_net = ActorCritic(input_dim=114, num_actions=7).to(device)
    state = torch.randn(4, 114, device=device)
    logits, value = ac_net(state)
    assert logits.shape == (4, 7)
    assert value.shape == (4, 1)
    
    # 2. Mobility GAT-GRU forward test
    gat_net = MobilityGAT_GRU(input_dim=2, embed_dim=64, num_heads=4, gru_hidden=64, output_dim=2).to(device)
    num_nodes = 4
    x_seq = torch.randn(num_nodes, 5, 2, device=device)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long, device=device)
    
    pred = gat_net(x_seq, edge_index)
    assert pred.shape == (num_nodes, 5, 2)
    print(f"[PASS] CoTOP ActorCritic & MobilityGAT_GRU forward verified on {device}.")

def verify_determinism(device):
    print(f"\n--- 6. Deterministic Evaluation Verification ---")
    from models.a3c_agent import ActorCritic
    
    torch.manual_seed(42)
    net1 = ActorCritic(input_dim=114, num_actions=7).to(device)
    net1.eval()
    
    state = torch.randn(1, 114, device=device)
    with torch.no_grad():
        out1_logits, out1_val = net1(state)
        out2_logits, out2_val = net1(state)
        
    assert torch.allclose(out1_logits, out2_logits)
    assert torch.allclose(out1_val, out2_val)
    print("[PASS] Deterministic evaluation verified: identical forward passes produced.")

def main():
    parser = argparse.ArgumentParser(description="Google Colab GPU Verification")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback for local debugging")
    args = parser.parse_args()
    
    print("=" * 60)
    print("   CoTOP GOOGLE COLAB GPU REPRODUCTION VERIFIER")
    print("=" * 60)
    
    comm_hash, comp_hash = verify_physics_hashes()
    env_info = check_environment(allow_cpu=args.allow_cpu)
    dev = env_info["device"]
    
    run_gpu_smoke_test(dev)
    run_ddqn_smoke_test(dev)
    run_cotop_smoke_test(dev)
    verify_determinism(dev)
    
    print("\n" + "=" * 60)
    print("   FINAL GPU VERIFICATION REPORT")
    print("=" * 60)
    print(f"GPU:            {env_info['gpu_name']}")
    print(f"CUDA:           {env_info['cuda_ver']}")
    print(f"PyTorch:        {env_info['pytorch']}")
    print(f"Python:         {env_info['python']}")
    print(f"Git SHA:        {env_info['git_sha']}")
    print(f"GPU available:  {'YES' if env_info['cuda_available'] else 'NO (CPU Mode)'}")
    print(f"Physics hashes: PASS (comm: {comm_hash[:8]}..., comp: {comp_hash[:8]}...)")
    print("Smoke test:     PASS")
    print("Status:         READY FOR COLAB GPU EXECUTION")
    print("=" * 60)

if __name__ == "__main__":
    main()
