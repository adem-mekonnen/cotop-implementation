import pypdf

reader = pypdf.PdfReader("docs/Mobility-Aware_Collaborative_Task_Offloading_for_Parallel_Tasks_in_Vehicular_Edge_Computing.pdf")

with open("docs/EXTRACTED_TABLES.txt", "w", encoding="utf-8") as out:
    for page_idx in [10, 11, 12, 13, 14]:
        page = reader.pages[page_idx]
        out.write(f"\n=== PAGE {page_idx + 1} ===\n")
        text = page.extract_text()
        for line in text.split("\n"):
            out.write(line + "\n")
print("Saved docs/EXTRACTED_TABLES.txt")
