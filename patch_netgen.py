import json

notebook_path = "research_implementation.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if 'CELL 3: SUMO INFRASTRUCTURE GENERATION' in source:
            # We will patch the net_cmd
            new_source = []
            skip = False
            for line in cell.get('source', []):
                if line.strip() == "net_cmd = [":
                    skip = True
                    new_source.append("net_cmd = [\n")
                    new_source.append("    NETGEN,\n")
                    new_source.append("    '--grid',\n")
                    new_source.append("    '--grid.x-number', '7',\n")
                    new_source.append("    '--grid.y-number', '1',\n")
                    new_source.append("    '--grid.length', '400',\n")
                    new_source.append("    '--default.lanenumber', '3',\n")
                    new_source.append("    '--default.speed', '13.89',\n")
                    new_source.append("    '--output-file', NET_FILE,\n")
                    new_source.append("    '--no-turnarounds', 'true',\n")
                    new_source.append("    '--tls.guess', 'true',\n")
                    new_source.append("]\n")
                elif skip and line.strip() == "]":
                    skip = False
                elif not skip:
                    new_source.append(line)
            cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Patched Cell 3 netgenerate command!")
