# FRESHLY TRAINED COTOP A3C MODEL EVALUATION REPORT

**Document Identifier**: `results/colab_fresh_training_evaluation/FRESH_TRAINING_EVALUATION_REPORT.md`  
**Evaluated Checkpoint**: `results/colab_training/cotop_trained.pt`  
**Checkpoint SHA-256**: `69c45161acfa58fd8f508044e72cdc64365e35d34876eb8984da875f87cd8f80`  
**Model Parameter Hash**: `dc666f93dcca1b2c1b3ec65e12075f042fad6e4c79f8fcbe9c8e784ed77d7ae8`  
**Realizations Evaluated**: `62`  

---

## 1. Metric Performance Summary

| Metric | Freshly Trained CoTOP | Physical Unit |
| :--- | :--- | :--- |
| **Mean Total Delay** | `1.4951 \pm 0.8258` | seconds (s) |
| **Mean Dynamic Energy** | `2.7723 \pm 1.7168` | Joules (J) |
| **Task Completion Ratio** | `98.60%` | percentage (%) |
| **Collaboration Rate** | `99.98%` | percentage (%) |

---

## 2. Action Distribution

```json
{
  "action_0": 0.02,
  "action_1": 72.17,
  "action_2": 0.0,
  "action_3": 0.74,
  "action_4": 0.0,
  "action_5": 18.56,
  "action_6": 8.52
}
```

---

## 3. Comparison with Canonical Reference Checkpoint

- Canonical CoTOP Delay: `1.3566 s` | Freshly Trained CoTOP Delay: `1.4951 s`
- Canonical CoTOP Energy: `2.6747 J` | Freshly Trained CoTOP Energy: `2.7723 J`
- Canonical CoTOP Completion: `98.67%` | Freshly Trained CoTOP Completion: `98.60%`
- Canonical CoTOP Collaboration: `89.04%` | Freshly Trained CoTOP Collaboration: `99.98%`
