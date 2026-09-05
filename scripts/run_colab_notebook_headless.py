#!/usr/bin/env python3
"""
scripts/run_colab_notebook_headless.py
Executes notebooks/CoTOP_Final_Colab_Reproduction.ipynb end-to-end in a clean Python runtime,
verifying that every cell runs to completion with zero errors.
"""

import os
import sys
import json
import time
import traceback

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOTEBOOK_PATH = os.path.join(ROOT_DIR, "notebooks", "CoTOP_Final_Colab_Reproduction.ipynb")

def run_notebook():
    print("=" * 80)
    print("       COTOP NOTEBOOK HEADLESS END-TO-END EXECUTION RUNNER")
    print("=" * 80)
    print(f"Loading notebook: {NOTEBOOK_PATH}")
    sys.stdout.flush()
    
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    print(f"Total cells: {len(cells)}")
    sys.stdout.flush()

    # Shared globals context for sequential execution across cells
    exec_globals = {
        "__name__": "__main__",
        "__file__": NOTEBOOK_PATH
    }

    code_cell_idx = 0
    start_total_time = time.time()

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        code_cell_idx += 1
        source_lines = cell.get("source", [])
        code = "".join(source_lines)

        first_line = source_lines[0].strip() if source_lines else "EMPTY"
        second_line = source_lines[1].strip() if len(source_lines) > 1 else ""
        header = second_line if first_line.startswith("# ==") and second_line else first_line

        print("\n" + "#" * 80)
        print(f" EXECUTING CODE CELL {code_cell_idx} (Cell {idx}): {header}")
        print("#" * 80)
        sys.stdout.flush()

        t0 = time.time()
        try:
            compiled = compile(code, f"Cell_{code_cell_idx}", "exec")
            exec(compiled, exec_globals)
            elapsed = time.time() - t0
            print(f"[SUCCESS] Cell {code_cell_idx} completed in {elapsed:.2f}s")
            sys.stdout.flush()
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n[FATAL ERROR] Cell {code_cell_idx} failed after {elapsed:.2f}s:")
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            sys.exit(1)

    total_time = time.time() - start_total_time
    print("\n" + "=" * 80)
    print(f"ALL {code_cell_idx} NOTEBOOK CODE CELLS EXECUTED SUCCESSFULLY IN {total_time:.2f}s!")
    print("=" * 80)
    sys.stdout.flush()

if __name__ == "__main__":
    run_notebook()
