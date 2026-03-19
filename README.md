# COSC 3P96 — Machine Learning Project

**Project 3: Ensemble Learning for Robust Facial Attribute Classification**
Dataset: CelebA | Target attribute: *Smiling*

## Files

| File | Purpose |
|---|---|
| `project3_ensemble_learning.py` | Complete implementation (Decision Tree → Random Forest → AdaBoost) |
| `PRESENTATION_GUIDE.md` | Slide-by-slide presentation guide with talking points |
| `fig1_decision_tree_depth_vs_accuracy.png` | Bias–variance trade-off for the DT baseline |
| `fig2_random_forest_n_estimators.png` | RF accuracy vs number of trees |
| `fig3_roc_curves.png` | ROC curves for all three models |
| `fig4_feature_importance.png` | Top 15 features (Random Forest) |
| `fig5_metrics_comparison.png` | Accuracy / Precision / Recall / F1 / AUC comparison |
| `Project.pdf` | Original assignment specification |

## Quick Start

```bash
pip install pandas scikit-learn matplotlib seaborn

# Run with synthetic CelebA-equivalent data (no download needed)
python3 project3_ensemble_learning.py

# Run with real CelebA data
# 1. Download list_attr_celeba.txt from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# 2. Place it in this directory
# 3. python3 project3_ensemble_learning.py
```

## Submission

Compile `project3_ensemble_learning.py` source + all console output + all five
`fig*.png` figures into a single PDF and upload to Brightspace by
**March 25, 2026 (11:59 PM)**.