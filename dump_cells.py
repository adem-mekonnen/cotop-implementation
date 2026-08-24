import json

notebook_path = "research_implementation.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('dump.txt', 'w', encoding='utf-8') as f:
    f.write("CELL 10 SOURCE:\n")
    f.write("".join(nb['cells'][10].get('source', [])))
    
    f.write("\n\nCELL 14 SOURCE:\n")
    f.write("".join(nb['cells'][14].get('source', [])))
