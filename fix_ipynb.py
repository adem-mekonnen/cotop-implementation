import json

path = "d:/cotop-implementation/research_implementation.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        if any("STEP 2: Setup CoTOP repository" in line for line in source):
            # Find the start index
            start_idx = -1
            end_idx = -1
            for i, line in enumerate(source):
                if line == "REPO_DIR = '/content/cotop-implementation'\n":
                    start_idx = i + 2 
                    break
                    
            for i in range(len(source)-1, -1, -1):
                if "print('\\u2705 Repository ready:" in source[i]:
                    end_idx = i - 1 
                    break
                    
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                new_source = source[:start_idx] + [
                    "if not os.path.exists(REPO_DIR):\n",
                    "    print('\\u2705 Cloning repository from GitHub...')\n",
                    "    # TODO: Replace YOUR_GITHUB_USERNAME with your actual GitHub username\n",
                    "    !git clone https://github.com/YOUR_GITHUB_USERNAME/cotop-implementation.git {REPO_DIR}\n",
                    "\n",
                    "# Final guard -- only chdir if directory confirmed present\n",
                    "if not os.path.exists(REPO_DIR):\n",
                    "    raise FileNotFoundError(\n",
                    "        '\\u274c ' + REPO_DIR + ' not found. '\n",
                    "        'Please ensure the GitHub clone URL is correct and the repository is public.'\n",
                    "    )\n",
                    "\n"
                ] + source[end_idx:]
                cell["source"] = new_source

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Fixed successfully.")
