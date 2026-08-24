import json
import ast

notebook_path = "research_implementation.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb.get('cells', [])):
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        
        # Remove colab magic commands for syntax checking
        clean_source = []
        for line in source.split('\n'):
            if line.strip().startswith('!'):
                clean_source.append(f"# {line}")
            else:
                clean_source.append(line)
        
        clean_source = "\n".join(clean_source)
        
        try:
            ast.parse(clean_source)
            print(f"Cell {i} Syntax: OK")
        except SyntaxError as e:
            print(f"Cell {i} Syntax Error: {e}")
            print(f"Code snippet: {clean_source[max(0, e.offset-20):e.offset+20]}")
