import json
import os

notebook_path = "research_implementation.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        new_source = []
        if isinstance(cell['source'], list):
            for line in cell['source']:
                # Replace literal backslash n with an actual newline character
                new_line = line.replace('\\n', '\n')
                new_source.append(new_line)
            cell['source'] = new_source
        elif isinstance(cell['source'], str):
            cell['source'] = cell['source'].replace('\\n', '\n')

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Fixed SyntaxError in notebook!")
