# CoTOP Reproduction Protocol & Execution Guide

This document provides a step-by-step guide for independent researchers to execute and verify the complete CoTOP reproduction pipeline from scratch.

---

## 1. System Requirements

- **Operating System**: Windows 10/11 64-bit or Linux (Ubuntu 22.04 LTS recommended)
- **Python**: Python 3.10 or 3.11
- **PyTorch**: PyTorch 2.4.1+ (CPU or CUDA 12.1+)
- **Traffic Simulator**: Eclipse SUMO (Simulation of Urban MObility) 1.25.0+

---

## 2. Installation & Environment Setup

```bash
# Clone the repository
git clone https://github.com/adem-mekonnen/cotop-implementation.git
cd cotop-implementation

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 3. Step 1: Analytical Model Sanity Check

To verify that the mathematical system models (Eq. 1–13, 23, 25) match closed-form analytical physics with 0.00% deviation:

```bash
python sanity_check.py
```
*Expected Output*: `>>> ALL SYSTEM MODEL SANITY CHECKS PASSED <<<` with 0.00e+00 error.

---

## 4. Step 2: Automated Unit Test Suite

To run all 22 pytest unit tests covering communication models, computation models, priority sorting, GAT-GRU mobility models, and A3C agents:

```bash
pytest -q
```
*Expected Output*: `22 passed in ~5.2s`.

---

## 5. Step 3: Multi-Seed A3C Training

To train the A3C reinforcement learning agent across 5 independent seeds:

```bash
python train.py --config configs/paper_parameters.yaml --episodes 500 --seed 42 --save_dir results/checkpoints/42
python train.py --config configs/paper_parameters.yaml --episodes 500 --seed 123 --save_dir results/checkpoints/123
python train.py --config configs/paper_parameters.yaml --episodes 500 --seed 456 --save_dir results/checkpoints/456
python train.py --config configs/paper_parameters.yaml --episodes 500 --seed 789 --save_dir results/checkpoints/789
python train.py --config configs/paper_parameters.yaml --episodes 500 --seed 2026 --save_dir results/checkpoints/2026
```

---

## 6. Step 4: Controlled Evaluation & Baseline Comparison

To evaluate CoTOP, Local, and Greedy across 50 episodes per seed (250 episodes per method):

```bash
python evaluate.py --mode cotop --checkpoint_path results/checkpoints/42/a3c_agent.pth --episodes 50 --seed 42 --config configs/paper_parameters.yaml --output_csv results/eval_cotop_42.csv
python evaluate.py --mode local --episodes 50 --seed 42 --config configs/paper_parameters.yaml --output_csv results/eval_local_42.csv
python evaluate.py --mode greedy --episodes 50 --seed 42 --config configs/paper_parameters.yaml --output_csv results/eval_greedy_42.csv
```

---

## 7. Step 5: Full Automated Reproduction Package Generation

To automatically execute multi-seed evaluations, baseline comparisons, statistical hypothesis tests, and diagnostic sweeps:

```bash
python experiments/stage17_final_controlled_reproduction.py
python experiments/stage18_package_final_results.py
```

Generated outputs will populate:
- `results/final/` (8 publication CSV tables)
- `figures/final/` (7 publication PNG figures)

---

## 8. Reproducing via Google Colab

Open the verified self-contained notebook:
[`notebooks/CoTOP_Stage11_Colab_Reproduction.ipynb`](file:///d:/cotop-implementation/notebooks/CoTOP_Stage11_Colab_Reproduction.ipynb)

The notebook clones the GitHub repository, checks out the verified commit, runs unit tests, trains A3C across seeds, and evaluates all baselines in an isolated cloud runtime.
