import pypdf

reader = pypdf.PdfReader("docs/Mobility-Aware_Collaborative_Task_Offloading_for_Parallel_Tasks_in_Vehicular_Edge_Computing.pdf")

for page_idx in [11, 13]:
    page = reader.pages[page_idx]
    print(f"=== PAGE {page_idx+1} TEXT ===")
    for text_obj in page.extract_text().split("\n"):
        if any(w in text_obj for w in ["TABLE", "CoTOP", "DDQN", "QRMP", "Local", "Greedy", "wo_"]):
            print(text_obj)
