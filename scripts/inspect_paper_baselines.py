import sys
import os

pdf_path = "docs/Mobility-Aware_Collaborative_Task_Offloading_for_Parallel_Tasks_in_Vehicular_Edge_Computing.pdf"
print("Checking PDF at:", pdf_path, "Exists:", os.path.exists(pdf_path))

found_lib = False
for lib in ["pypdf", "PyPDF2", "fitz", "pdfplumber"]:
    try:
        m = __import__(lib)
        print(f"Library {lib} is installed.")
        found_lib = True
        if lib == "pypdf" or lib == "PyPDF2":
            reader = m.PdfReader(pdf_path)
            print(f"Total pages: {len(reader.pages)}")
            for i, p in enumerate(reader.pages):
                txt = p.extract_text()
                for line in txt.split("\n"):
                    if any(w in line.lower() for w in ["ddqn", "qrmp", "quantile", "greedy", "baseline", "double dqn", "table iv", "table v", "figure 4", "figure 5", "figure 6"]):
                        print(f"[P{i+1}] {line}")
            break
        elif lib == "fitz":
            doc = m.open(pdf_path)
            print(f"Total pages: {len(doc)}")
            for i, page in enumerate(doc):
                txt = page.get_text()
                for line in txt.split("\n"):
                    if any(w in line.lower() for w in ["ddqn", "qrmp", "quantile", "greedy", "baseline", "double dqn"]):
                        print(f"[P{i+1}] {line}")
            break
    except Exception as e:
        print(f"Could not use {lib}: {e}")

if not found_lib:
    print("No standard PDF library found.")
