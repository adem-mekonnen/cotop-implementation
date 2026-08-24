import json

path = "d:/cotop-implementation/research_implementation.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        new_source = []
        skip = False
        for i, line in enumerate(source):
            if "if not os.path.exists(REPO_DIR):" in line and i > 20: 
                # Line 83
                new_source.append("if not os.path.exists(REPO_DIR):\n")
                new_source.append("    print('\\u2705 Cloning repository from GitHub...')\n")
                new_source.append("    # TODO: Replace YOUR_GITHUB_USERNAME with your actual GitHub username\n")
                new_source.append("    !git clone https://github.com/YOUR_GITHUB_USERNAME/cotop-implementation.git {REPO_DIR}\n")
                new_source.append("\n")
                new_source.append("# Final guard -- only chdir if directory confirmed present\n")
                new_source.append("if not os.path.exists(REPO_DIR):\n")
                new_source.append("    raise FileNotFoundError(\n")
                new_source.append("        '\\u274c ' + REPO_DIR + ' not found. '\n")
                new_source.append("        'Please ensure the GitHub clone URL is correct and the repository is public.'\n")
                new_source.append("    )\n")
                
                # skip until end of old block
                skip = True
                continue
            
            if skip:
                # We need to stop skipping once we are past the original `raise FileNotFoundError` block
                if "    )\n" in line and "Upload cotop-implementation.zip" in source[i-1]:
                    skip = False
                continue
            
            new_source.append(line)
        cell["source"] = new_source

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Modification complete.")
