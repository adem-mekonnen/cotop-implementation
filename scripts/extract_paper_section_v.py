import pypdf
import os

pdf_path = "docs/Mobility-Aware_Collaborative_Task_Offloading_for_Parallel_Tasks_in_Vehicular_Edge_Computing.pdf"
reader = pypdf.PdfReader(pdf_path)

out_file = "docs/PAPER_SECTION_V_EXCERPT.md"
with open(out_file, "w", encoding="utf-8") as f:
    f.write("# Paper Section V & VI Excerpts: Baselines, Tables IV-VI, Experiments\n\n")
    for page_num in range(10, 16):
        f.write(f"## Page {page_num + 1}\n\n")
        f.write(reader.pages[page_num].extract_text() + "\n\n")

print(f"Extracted pages 11-16 into {out_file}")
