import subprocess
import json

def run(cmd):
    return subprocess.check_output(cmd, shell=True).decode('utf-8', errors='ignore').strip()

def get_files(diff_range):
    out = run(f"git diff --name-only {diff_range}")
    return set([line.strip() for line in out.splitlines() if line.strip()])

sci = get_files("main..reproduction/scientific-fidelity")
pva = get_files("main..reproduction/published-value-audit")
mvc = get_files("main..reproduction/multivehicle-contention")

print(f"Files in scientific-fidelity: {len(sci)}")
print(f"Files in published-value-audit: {len(pva)}")
print(f"Files in multivehicle-contention: {len(mvc)}")

mult = (sci & pva) | (sci & mvc) | (pva & mvc)
print(f"\nFiles changed in multiple branches ({len(mult)}):")
for f in sorted(mult):
    br = []
    if f in sci: br.append("scientific-fidelity")
    if f in pva: br.append("published-value-audit")
    if f in mvc: br.append("multivehicle-contention")
    print(f"  {f}: {', '.join(br)}")
