import json
import os
import shutil

notebook_path = "research_implimentation.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell['source']
        if len(source) > 0 and 'CELL 1: ENVIRONMENT SETUP' in source[1]:
            # Rewrite Cell 1 to not pull if the directory already exists (so it respects uploaded files)
            new_source = []
            for line in source:
                if 'else:' in line:
                    new_source.append(line)
                    new_source.append("    print(f'Repository already exists at {REPO_DIR}. Using local files.')\n")
                    new_source.append("    # !cd {REPO_DIR} && git pull\n")
                    break
                else:
                    new_source.append(line)
            
            # Continue after the break if there are other lines
            # Wait, the original source had:
            # else:
            #     print(f'Repository already exists at {REPO_DIR}, pulling latest...')
            #     !cd {REPO_DIR} && git pull
            # os.chdir(REPO_DIR)
            
            found_else = False
            skip = False
            new_source = []
            for line in source:
                if skip:
                    if not line.startswith('    '):
                        skip = False
                    else:
                        continue
                if not skip:
                    if 'else:' in line:
                        found_else = True
                        new_source.append(line)
                        new_source.append("    print(f'Repository already exists at {REPO_DIR}. Using local files.')\n")
                        new_source.append("    # !cd {REPO_DIR} && git pull\n")
                        skip = True
                    else:
                        new_source.append(line)
            
            cell['source'] = new_source
            break

new_notebook_path = "research_implementation.ipynb"
with open(new_notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated and renamed to research_implementation.ipynb!")
