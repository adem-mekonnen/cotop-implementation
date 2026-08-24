import json

path = "d:/cotop-implementation/research_implementation.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        if any("CELL 3: SUMO INFRASTRUCTURE GENERATION" in line for line in source):
            # find net_cmd and modify
            new_source = []
            for i, line in enumerate(source):
                if "'--tls.guess', 'true'," in line:
                    continue # remove tls.guess because 1D grid doesn't have true intersections, might cause issues on linux
                
                if "print('stderr:', result.stderr[-500:])" in line:
                    new_source.append("    print('stdout:', result.stdout)\n")
                    new_source.append("    print('stderr:', result.stderr)\n")
                    continue
                
                if "print('stderr:', result2.stderr[-300:])" in line:
                    new_source.append("    print('stdout:', result2.stdout)\n")
                    new_source.append("    print('stderr:', result2.stderr)\n")
                    continue

                new_source.append(line)
            cell["source"] = new_source

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Cell 3 patched successfully.")
