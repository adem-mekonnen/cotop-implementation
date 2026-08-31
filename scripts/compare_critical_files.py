import subprocess

files = [
    "envs/vec_env.py",
    "utils/task_priority.py",
    "evaluate.py",
    "models/mobility_gat.py"
]

branches = [
    "reproduction/scientific-fidelity",
    "reproduction/published-value-audit",
    "reproduction/multivehicle-contention"
]

print("=== 3-WAY COMPARISON OF CRITICAL FILES ===")
for f in files:
    print(f"\n==================== FILE: {f} ====================")
    for b in branches:
        try:
            content = subprocess.check_output(f"git show {b}:{f}", shell=True).decode('utf-8', errors='ignore')
            lines = len(content.splitlines())
            print(f"[{b}] exists, {lines} lines")
        except Exception as e:
            print(f"[{b}] NOT FOUND or ERROR")
            
    # Check diffs against scientific-fidelity
    for other_b in ["reproduction/published-value-audit", "reproduction/multivehicle-contention"]:
        diff = subprocess.check_output(f"git diff reproduction/scientific-fidelity:{f} {other_b}:{f}", shell=True).decode('utf-8', errors='ignore')
        diff_lines = len(diff.splitlines())
        print(f"Diff (scientific-fidelity vs {other_b}): {diff_lines} diff lines")
