"""
COSC 3P96 – Machine Learning Projects
Project 3: Ensemble Learning for Robust Facial Attribute Classification
Dataset: CelebA (attributes only — no image download required)

This script:
1. Loads the CelebA attribute table (list_attr_celeba.txt).
   If the file is not present it is downloaded automatically from
   Google Drive using gdown (pip install gdown).
2. Trains a Decision Tree, Random Forest, and AdaBoost classifier to
   predict the 'Smiling' attribute from the remaining 39 attributes.
3. Evaluates all three models with accuracy, precision, recall, F1,
   and ROC-AUC.
4. Produces the following saved figures:
      fig1_decision_tree_depth_vs_accuracy.png
      fig2_random_forest_n_estimators.png
      fig3_roc_curves.png
      fig4_feature_importance.png
      fig5_metrics_comparison.png
5. Prints a concise results summary table.

Submission note
---------------
Paste this file's source + printed outputs + the five saved figures
into a single PDF before uploading to Brightspace.
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (works on any machine)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report,
    ConfusionMatrixDisplay, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Styling ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
COLORS = ["#4C72B0", "#DD8452", "#55A868"]   # DT, RF, AdaBoost

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
CELEBA_ATTR_PATH = "list_attr_celeba.txt"   # extracted from list_attr_celeba.zip if needed
CELEBA_ZIP_PATH  = "list_attr_celeba.zip"
TARGET_ATTR = "Smiling"


def load_celeba_attributes(path: str) -> pd.DataFrame:
    """Parse CelebA list_attr_celeba.txt into a tidy DataFrame.
    Values are converted from {-1, +1} to {0, 1}.
    """
    # First line: number of images; second line: column headers
    df = pd.read_csv(path, sep=r"\s+", header=1, index_col=0)
    df = (df + 1) // 2   # {-1,+1} → {0,1}
    df.index.name = "image_id"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def download_celeba_attributes(dest_path: str) -> None:
    """Download list_attr_celeba.txt from the official CelebA Google Drive mirror.

    Requires the ``gdown`` package (pip install gdown).
    The file is ~25 MB and contains binary attribute labels for 202,599 images.
    """
    try:
        import gdown  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'gdown' package is required to download the CelebA attribute file.\n"
            "Install it with:  pip install gdown"
        ) from exc

    # Official CelebA Google Drive file ID for list_attr_celeba.txt
    CELEBA_FILE_ID = "0B7EVK8r0v71pblRyaVFSWGxPY0U"
    url = f"https://drive.google.com/uc?id={CELEBA_FILE_ID}"
    print(f"Downloading CelebA attribute file to '{dest_path}' …")
    gdown.download(url, dest_path, quiet=False)
    if not os.path.exists(dest_path):
        raise RuntimeError(
            "Download failed. Possible causes: no internet connection, Google Drive\n"
            "quota exceeded, or the file ID has changed.\n"
            "As a fallback, download list_attr_celeba.txt manually from\n"
            "https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html\n"
            "and place it in the same directory as this script."
        )
    print("Download complete.")


if not os.path.exists(CELEBA_ATTR_PATH):
    if os.path.exists(CELEBA_ZIP_PATH):
        import zipfile
        print(f"Extracting '{CELEBA_ATTR_PATH}' from '{CELEBA_ZIP_PATH}' …")
        with zipfile.ZipFile(CELEBA_ZIP_PATH, "r") as zf:
            zf.extract("list_attr_celeba.txt")
        print("Extraction complete.")
    else:
        download_celeba_attributes(CELEBA_ATTR_PATH)

print(f"Loading CelebA attributes from '{CELEBA_ATTR_PATH}' …")
df = load_celeba_attributes(CELEBA_ATTR_PATH)

print(f"Dataset shape  : {df.shape[0]:,} samples × {df.shape[1]} attributes")
print(f"Target         : {TARGET_ATTR!r}  (positive class = {df[TARGET_ATTR].mean()*100:.1f}%)")
print(f"\nAttribute list :\n{list(df.columns)}\n")

# Features and target
y = df[TARGET_ATTR].values
X = df.drop(columns=[TARGET_ATTR]).values
feature_names = [c for c in df.columns if c != TARGET_ATTR]

# Train / validation / test split  (60% / 20% / 20%)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
)

print(f"Train  : {X_train.shape[0]:,} samples")
print(f"Val    : {X_val.shape[0]:,} samples")
print(f"Test   : {X_test.shape[0]:,} samples")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DECISION TREE BASELINE (with depth tuning)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3 — Decision Tree (Baseline)")
print("=" * 60)

depths = list(range(1, 21))
dt_train_acc, dt_val_acc = [], []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    dt_train_acc.append(accuracy_score(y_train, dt.predict(X_train)))
    dt_val_acc.append(accuracy_score(y_val, dt.predict(X_val)))

best_depth = depths[np.argmax(dt_val_acc)]
print(f"Best max_depth on validation set: {best_depth}  "
      f"(val_acc = {max(dt_val_acc):.4f})")

# Final DT trained on train+val with best depth
dt_best = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_best.fit(X_trainval, y_trainval)
dt_proba = dt_best.predict_proba(X_test)[:, 1]
dt_pred = dt_best.predict(X_test)

# ── Figure 1: DT depth vs accuracy ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(depths, dt_train_acc, "o-", label="Train accuracy", color="#4C72B0")
ax.plot(depths, dt_val_acc,   "s-", label="Validation accuracy", color="#DD8452")
ax.axvline(best_depth, color="gray", linestyle="--", alpha=0.7,
           label=f"Best depth = {best_depth}")
ax.set_xlabel("Tree Depth (max_depth)")
ax.set_ylabel("Accuracy")
ax.set_title("Decision Tree: Depth vs. Accuracy\n(Illustrates Bias–Variance Trade-off)")
ax.legend()
fig.tight_layout()
fig.savefig("fig1_decision_tree_depth_vs_accuracy.png", dpi=150)
plt.close(fig)
print("Saved fig1_decision_tree_depth_vs_accuracy.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — RANDOM FOREST (Bagging)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4 — Random Forest (Bagging)")
print("=" * 60)

n_est_options = [10, 25, 50, 100, 200, 300]
rf_val_acc = []

for n in n_est_options:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_val_acc.append(accuracy_score(y_val, rf.predict(X_val)))

best_n_rf = n_est_options[np.argmax(rf_val_acc)]
print(f"Best n_estimators on validation set: {best_n_rf}  "
      f"(val_acc = {max(rf_val_acc):.4f})")

rf_best = RandomForestClassifier(n_estimators=best_n_rf, random_state=42, n_jobs=-1)
rf_best.fit(X_trainval, y_trainval)
rf_proba = rf_best.predict_proba(X_test)[:, 1]
rf_pred  = rf_best.predict(X_test)

# ── Figure 2: RF n_estimators vs accuracy ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(n_est_options, rf_val_acc, "D-", color="#55A868")
ax.axvline(best_n_rf, color="gray", linestyle="--", alpha=0.7,
           label=f"Best n_estimators = {best_n_rf}")
ax.set_xlabel("Number of Trees (n_estimators)")
ax.set_ylabel("Validation Accuracy")
ax.set_title("Random Forest: Number of Trees vs. Accuracy\n(More trees → lower variance)")
ax.legend()
fig.tight_layout()
fig.savefig("fig2_random_forest_n_estimators.png", dpi=150)
plt.close(fig)
print("Saved fig2_random_forest_n_estimators.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — ADABOOST (Boosting)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 5 — AdaBoost (Boosting)")
print("=" * 60)

# Weak learner: shallow decision stump (depth=1)
stump = DecisionTreeClassifier(max_depth=1)
ada = AdaBoostClassifier(
    estimator=stump,
    n_estimators=200,
    learning_rate=0.5,
    random_state=42,
)
ada.fit(X_trainval, y_trainval)
ada_proba = ada.predict_proba(X_test)[:, 1]
ada_pred  = ada.predict(X_test)
print(f"AdaBoost trained  (n_estimators=200, lr=0.5, base=DecisionStump)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 6 — Evaluation on Test Set")
print("=" * 60)


def metrics(y_true, y_pred, y_proba, name):
    """Return a dict of evaluation metrics."""
    return {
        "Model":     name,
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_true, y_proba),
    }


results = [
    metrics(y_test, dt_pred,  dt_proba,  "Decision Tree"),
    metrics(y_test, rf_pred,  rf_proba,  "Random Forest"),
    metrics(y_test, ada_pred, ada_proba, "AdaBoost"),
]

results_df = pd.DataFrame(results).set_index("Model")
print("\nTest-set performance summary:")
print(results_df.to_string(float_format="{:.4f}".format))

# ── Figure 3: ROC Curves ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for (name, proba), color in zip(
    [("Decision Tree", dt_proba), ("Random Forest", rf_proba), ("AdaBoost", ada_proba)],
    COLORS,
):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=color, lw=2)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curves — Predicting '{TARGET_ATTR}' (CelebA)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("fig3_roc_curves.png", dpi=150)
plt.close(fig)
print("\nSaved fig3_roc_curves.png")

# ── Figure 4: Feature Importance (Random Forest) ────────────────────────────
importances = rf_best.feature_importances_
sorted_idx  = np.argsort(importances)[-15:]    # top 15

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(
    [feature_names[i] for i in sorted_idx],
    importances[sorted_idx],
    color="#55A868",
    edgecolor="white",
)
ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)")
ax.set_title(f"Random Forest — Top 15 Features for Predicting '{TARGET_ATTR}'")
fig.tight_layout()
fig.savefig("fig4_feature_importance.png", dpi=150)
plt.close(fig)
print("Saved fig4_feature_importance.png")

# ── Figure 5: Metrics comparison bar chart ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
x = np.arange(len(metrics_to_plot))
width = 0.25

for i, (model, color) in enumerate(zip(results_df.index, COLORS)):
    vals = results_df.loc[model, metrics_to_plot].values.astype(float)
    bars = ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

ax.set_xticks(x + width)
ax.set_xticklabels(metrics_to_plot)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score")
ax.set_title(f"Model Comparison — Predicting '{TARGET_ATTR}' (CelebA Dataset)")
ax.legend()
fig.tight_layout()
fig.savefig("fig5_metrics_comparison.png", dpi=150)
plt.close(fig)
print("Saved fig5_metrics_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — DISCUSSION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 7 — Discussion")
print("=" * 60)

best_model = results_df["F1"].idxmax()
print(f"""
Decision Tree (Baseline)
  - Simple, interpretable, but prone to overfitting (high variance).
  - Depth tuning reveals the bias–variance trade-off clearly (Fig. 1).
  - Accuracy: {results_df.loc['Decision Tree','Accuracy']:.4f}

Random Forest (Bagging)
  - Trains many diverse trees on bootstrap samples and averages votes.
  - Reduces variance without significantly increasing bias.
  - Accuracy: {results_df.loc['Random Forest','Accuracy']:.4f}
  - Most informative features: {', '.join([feature_names[i] for i in sorted_idx[-3:]][::-1])}.

AdaBoost (Boosting)
  - Trains weak learners (stumps) sequentially; each round up-weights
    misclassified samples so the next stump corrects prior errors.
  - Reduces bias; can overfit on noisy data.
  - Accuracy: {results_df.loc['AdaBoost','Accuracy']:.4f}

Best model by F1: {best_model}
  → Ensemble methods outperform the single Decision Tree, demonstrating
    that combining learners improves both accuracy and robustness.
""")

print("=" * 60)
print("All figures saved. Include them + this output in your PDF submission.")
print("=" * 60)
