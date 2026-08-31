import subprocess

def inspect(path):
    try:
        out = subprocess.check_output(f"git show reproduction/multivehicle-contention:{path}", shell=True).decode('utf-8', errors='ignore')
        print(f"=== {path} ===")
        print(out[:1500])
    except Exception as e:
        print(f"Failed {path}: {e}")

inspect("results/multivehicle_contention_colab/statistical_analysis.csv")
inspect("results/multivehicle_contention_colab/seed_summary.csv")
