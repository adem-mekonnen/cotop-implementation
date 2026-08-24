import json
import os

notebook_path = "research_implementation.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        
        # We need to fix the cell 1 git pull issue as well
        if len(source) > 0 and 'CELL 1: ENVIRONMENT SETUP' in source[1] if len(source)>1 else False:
            skip = False
            for line in source:
                if skip:
                    if not line.startswith('    '):
                        skip = False
                    else:
                        continue
                if not skip:
                    if 'else:' in line:
                        new_source.append(line)
                        new_source.append("    print(f'Repository already exists at {REPO_DIR}. Using local files.')\n")
                        new_source.append("    # !cd {REPO_DIR} && git pull\n")
                        skip = True
                    else:
                        new_source.append(line)
        else:
            # For other cells, just fix the specific syntax errors of print()\nprint
            # DO NOT replace '\\n' blindly everywhere, only for the 'print()\\nprint' case
            for line in source:
                if "print()\\nprint" in line:
                    new_line = line.replace("print()\\nprint", "print()\\n\",\n\"print")
                    # wait, this is manipulating the strings inside the JSON representation
                    # If line is "print()\\nprint('Generating...')\n"
                    # We can just change it to two lines!
                    # Actually, we can just replace "\\n" in that specific string if we format it as multiple elements
                    # A simpler way: replace "print()\\nprint" with "print()\nprint"
                    new_line = line.replace("print()\\nprint", "print()\\n\",\n\"print")
                    # wait, inside cell['source'] list of strings, each is a string.
                    # if the string is "print()\\nprint('foo')\n"
                    pass # We will do it more robustly below
        
        # A robust way to fix the print() line continuation issue without breaking other cells
        # We replace "print()\\nprint" with "print()\\n" and "print" split into two strings
        fixed_source = []
        for line in (new_source if new_source else source):
            if "print()\\nprint(" in line:
                parts = line.split("print()\\nprint(")
                fixed_source.append(parts[0] + "print()\\n")
                fixed_source.append("print(" + parts[1])
            else:
                fixed_source.append(line)
                
        cell['source'] = fixed_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Smartly fixed syntax errors and environment setup!")
