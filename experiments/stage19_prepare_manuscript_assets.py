import os
import sys
import shutil
import pandas as pd

def df_to_markdown(df):
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(val) for val in row) + " |")
    return "\n".join(lines)

def prepare_manuscript_assets():
    print("=" * 75)
    print("STAGE 19: PREPARING MANUSCRIPT ASSETS & TABLES")
    print("=" * 75)
    
    manuscript_dir = "manuscript"
    tables_dir = os.path.join(manuscript_dir, "tables")
    figures_dir = os.path.join(manuscript_dir, "figures")
    supp_dir = os.path.join(manuscript_dir, "supplementary")
    
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(supp_dir, exist_ok=True)
    
    # 1. Copy Figures from figures/final/ to manuscript/figures/
    src_fig_dir = "figures/final"
    if os.path.exists(src_fig_dir):
        for fig_file in os.listdir(src_fig_dir):
            if fig_file.endswith(".png"):
                shutil.copy(os.path.join(src_fig_dir, fig_file), os.path.join(figures_dir, fig_file))
        print("Copied publication figures to manuscript/figures/")
        
    # 2. Format and export all tables from results/final/
    results_final_dir = "results/final"
    table_files = [
        ("01_reproduction_fidelity.csv", "table1_implementation_fidelity"),
        ("02_final_performance_comparison.csv", "table4_performance_comparison"),
        ("03_final_statistical_analysis.csv", "table5_statistical_analysis"),
        ("04_training_sufficiency.csv", "table3_training_sufficiency"),
        ("05_published_vs_reproduced.csv", "table6_published_vs_reproduced"),
        ("06_claim_evidence_matrix.csv", "table7_claim_evidence_matrix"),
        ("07_limitations.csv", "table8_threats_to_validity"),
        ("08_final_reproduction_verdict.csv", "table9_final_verdict")
    ]
    
    for csv_name, out_base in table_files:
        csv_path = os.path.join(results_final_dir, csv_name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Export Markdown
            md_table = df_to_markdown(df)
            with open(os.path.join(tables_dir, f"{out_base}.md"), 'w', encoding='utf-8') as f:
                f.write(md_table + "\n")
            # Export LaTeX
            tex_table = df.to_latex(index=False, escape=True)
            with open(os.path.join(tables_dir, f"{out_base}.tex"), 'w', encoding='utf-8') as f:
                f.write(tex_table + "\n")
            print(f"Exported {out_base}.md and {out_base}.tex")
            
    # 3. Create Table 2: Experimental Configuration Table
    table2_data = [
        {"Parameter": "Corridor Geometry (Length)", "Value": "2400 m (straight road)", "Provenance": "Table III", "Implementation Status": "Exact Match"},
        {"Parameter": "RSU Infrastructure", "Value": "6 RSUs, 400 m uniform spacing, 400 m radius", "Provenance": "Table III", "Implementation Status": "Exact Match"},
        {"Parameter": "Vehicle Population", "Value": "10 to 30 concurrent vehicles", "Provenance": "Table III", "Implementation Status": "Exact Match"},
        {"Parameter": "Vehicle Velocity Range", "Value": "30.0 to 40.0 m/s (108 to 144 km/h)", "Provenance": "Table III", "Implementation Status": "Exact Match"},
        {"Parameter": "Task Data Size Range", "Value": "2.0 to 5.0 MB", "Provenance": "Table III", "Implementation Status": "Exact Match"},
        {"Parameter": "Nominal CPU Demand", "Value": "10.0 Mcycles", "Provenance": "Section III-F, V-A", "Implementation Status": "Exact Match"},
        {"Parameter": "Task QoS Latency Deadline", "Value": "20.0 to 30.0 s", "Provenance": "Table III", "Implementation Status": "Exact Match"},
        {"Parameter": "RSU CPU Frequency", "Value": "1.0 to 4.0 GHz (2.0 GHz nominal)", "Provenance": "Table III", "Implementation Status": "Exact Match"},
        {"Parameter": "Vehicle TX Power (P_V)", "Value": "10 dBm (0.01 W)", "Provenance": "Table III", "Implementation Status": "Exact Match"},
        {"Parameter": "RSU Relay TX Power (P_R)", "Value": "50 dBm (100.0 W)", "Provenance": "Table III", "Implementation Status": "Exact Match"},
        {"Parameter": "RSU Compute Power (E_RSU)", "Value": "50.0 W", "Provenance": "Eq. 11", "Implementation Status": "Inferred / Assumed"},
        {"Parameter": "V2R Wireless Bandwidth", "Value": "20.0 to 100.0 MHz", "Provenance": "Table III", "Implementation Status": "Exact Match"},
        {"Parameter": "R2R Wireless Bandwidth", "Value": "50.0 MHz", "Provenance": "Table III", "Implementation Status": "Exact Match"},
        {"Parameter": "Thermal Noise Power", "Value": "0.001 W (0.001 dBm)", "Provenance": "Table III", "Implementation Status": "Exact Match"},
        {"Parameter": "Path Loss Parameters", "Value": "K = 1000.0 (30 dB), gamma = 2.0", "Provenance": "Table III", "Implementation Status": "Exact Match"},
        {"Parameter": "Task Priority Weights", "Value": "alpha = 0.3, beta = 0.7", "Provenance": "Section V-C", "Implementation Status": "Exact Match"},
        {"Parameter": "A3C Optimizer & Learning Rate", "Value": "SharedAdam, lr = 0.0002", "Provenance": "Section V-C", "Implementation Status": "Exact Match"},
        {"Parameter": "Training Horizon", "Value": "500-1000 episodes (50-100 epochs)", "Provenance": "Section V-B, Fig 4", "Implementation Status": "Exact Match / Extended"}
    ]
    df_table2 = pd.DataFrame(table2_data)
    with open(os.path.join(tables_dir, "table2_experimental_configuration.md"), 'w', encoding='utf-8') as f:
        f.write(df_to_markdown(df_table2) + "\n")
    with open(os.path.join(tables_dir, "table2_experimental_configuration.tex"), 'w', encoding='utf-8') as f:
        f.write(df_table2.to_latex(index=False, escape=True) + "\n")
    print("Exported table2_experimental_configuration.md and .tex")

    # 4. Create references.bib
    bibtex_content = """@article{du2026mobility,
  author    = {Jiaxin Du and Jinfan Zhang and Guangjie Han and Mengmeng Wang and Guojiang Shen and Zhi Liu and Xiangjie Kong},
  title     = {Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing},
  journal   = {IEEE Transactions on Mobile Computing},
  volume    = {25},
  number    = {4},
  pages     = {5540--5555},
  year      = {2026},
  month     = {April},
  doi       = {10.1109/TMC.2025.3631820}
}

@article{velickovic2018graph,
  author    = {Petar Veli{\v{c}}kovi{\'{c}} and Guillem Cucurull and Arantxa Casanova and Adriana Romero and Pietro Li{\`{o}} and Yoshua Bengio},
  title     = {Graph Attention Networks},
  journal   = {International Conference on Learning Representations (ICLR)},
  year      = {2018}
}

@inproceedings{mnih2016asynchronous,
  author    = {Volodymyr Mnih and Adri{\`{a}} Puigdom{\`{e}}nech Badia and Mehdi Mirza and Alex Graves and Timothy P. Lillicrap and Tim Harley and David Silver and Koray Kavukcuoglu},
  title     = {Asynchronous Methods for Deep Reinforcement Learning},
  booktitle = {International Conference on Machine Learning (ICML)},
  pages     = {1928--1937},
  year      = {2016}
}

@inproceedings{krajzewicz2012recent,
  author    = {Daniel Krajzewicz and Jakob Erdmann and Michael Behrisch and Laura Bieker},
  title     = {Recent Development and Applications of {SUMO} - {Simulation of Urban MObility}},
  booktitle = {International Journal On Advances in Systems and Measurements},
  volume    = {5},
  number    = {3\&4},
  pages     = {128--138},
  year      = {2012}
}

@article{huang2018apolloscape,
  author    = {Xinyu Huang and Xinjing Cheng and Qichuan Geng and Binbin Cao and Dingfu Zhou and Peng Wang and Yuanqing Lin and Ruigang Yang},
  title     = {The {ApolloScape} Dataset for Autonomous Driving},
  journal   = {IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages     = {954--960},
  year      = {2018}
}

@article{cohen1988statistical,
  author    = {Jacob Cohen},
  title     = {Statistical Power Analysis for the Behavioral Sciences},
  publisher = {Lawrence Erlbaum Associates},
  edition   = {2nd},
  year      = {1988}
}

@article{benjamini1995controlling,
  author    = {Yoav Benjamini and Yosef Hochberg},
  title     = {Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing},
  journal   = {Journal of the Royal Statistical Society: Series B (Methodological)},
  volume    = {57},
  number    = {1},
  pages     = {289--300},
  year      = {1995}
}

@article{holm1979simple,
  author    = {Sture Holm},
  title     = {A Simple Sequentially Rejective Multiple Test Procedure},
  journal   = {Scandinavian Journal of Statistics},
  volume    = {6},
  number    = {2},
  pages     = {65--70},
  year      = {1979}
}
"""
    with open(os.path.join(manuscript_dir, "references.bib"), 'w', encoding='utf-8') as f:
        f.write(bibtex_content)
    print("Created references.bib")
    
    print("\n" + "=" * 75)
    print("MANUSCRIPT ASSET PREPARATION COMPLETED")
    print("=" * 75)

if __name__ == "__main__":
    prepare_manuscript_assets()
