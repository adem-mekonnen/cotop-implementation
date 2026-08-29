import os
import itertools
import json

geometries = ["linear_corridor", "urban_manhattan"]
algorithms = ["CoTOP", "DDQN"]
workloads = [20, 30, 40]
seeds = [42, 43, 44, 45, 46]

total = 0
completed = 0
missing = []

for g, a, w, s in itertools.product(geometries, algorithms, workloads, seeds):
    total += 1
    d = f"results/phase2_algorithmic_fidelity/{g}/{a}/w{w}/seed_{s}"
    ckpt = os.path.join(d, "checkpoint_ep500.pt")
    man = os.path.join(d, "run_manifest.json")
    if os.path.exists(ckpt) and os.path.exists(man):
        completed += 1
    else:
        missing.append(f"{g} | {a} | w{w} | seed_{s}")

print(f"Total cells: {total}")
print(f"Completed:   {completed} ({completed/total*100:.1f}%)")
print(f"Missing:     {len(missing)}")
if missing:
    print("Pending conditions:")
    for m in missing:
        print(f"  - {m}")
