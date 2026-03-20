"""
COSC 3P96 – Machine Learning Projects
Project 3: Ensemble Learning for Robust Facial Attribute Classification
Dataset: CelebA (attributes only — no image download required)

Overview
--------
This project compares three tree-based classifiers on a real-world binary
classification task: predicting whether a celebrity in a photo is *smiling*
using 39 other binary facial-attribute labels (no raw pixel data needed).

The three models follow a natural progression of complexity:
  1. Decision Tree   — single interpretable model; prone to overfitting.
  2. Random Forest   — bagging ensemble: many independent trees averaged
                       to reduce variance.
  3. AdaBoost        — boosting ensemble: trees built sequentially, each
                       correcting the mistakes of its predecessor, reducing
                       bias.

This script:
1. Loads the CelebA attribute table (list_attr_celeba.txt).
   The file must be present locally (plain .txt or bundled .zip).
2. Trains a Decision Tree, Random Forest, and AdaBoost classifier to
   predict the 'Smiling' attribute from the remaining 39 attributes.
3. Evaluates all three models with accuracy, precision, recall, F1,
   and ROC-AUC.
4. Produces the following saved figures:
      fig1_decision_tree_depth_vs_accuracy.png  — bias-variance trade-off
      fig2_random_forest_n_estimators.png       — effect of ensemble size
      fig3_roc_curves.png                       — ROC curves for all models
      fig4_feature_importance.png               — top-15 RF feature importances
      fig5_metrics_comparison.png               — grouped bar chart of all metrics
5. Prints a concise results summary table and discussion.

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
matplotlib.use("Agg")           # non-interactive backend — works without a display
                                # (required on headless servers / CI environments)
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

# Suppress sklearn convergence / version warnings to keep console output clean
warnings.filterwarnings("ignore")

# Fix the global random seed so every run produces the same results.
# This affects NumPy random draws AND sklearn estimators that accept random_state.
np.random.seed(42)

# ── Styling ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
# Consistent colour palette used across all figures.
# Blue  = Decision Tree baseline
# Orange = Random Forest (bagging)
# Green  = AdaBoost (boosting)
COLORS = ["#4C72B0", "#DD8452", "#55A868"]   # DT, RF, AdaBoost

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
# CelebA stores attribute annotations in a plain-text file with one row per
# image and one column per attribute.  Values are +1 (present) or -1 (absent).
# We convert to {0, 1} for compatibility with sklearn classifiers.
CELEBA_ATTR_PATH = "list_attr_celeba.txt"   # extracted from list_attr_celeba.zip if needed
CELEBA_ZIP_PATH  = "list_attr_celeba.zip"
TARGET_ATTR = "Smiling"                      # binary label we want to predict


def load_celeba_attributes(path: str) -> pd.DataFrame:
    """Parse CelebA list_attr_celeba.txt into a tidy DataFrame.
    Values are converted from {-1, +1} to {0, 1}.

    File format (first two lines are metadata):
      Line 1: total number of images (e.g. "202599")
      Line 2: space-separated attribute names
      Lines 3+: image_id  followed by attribute values (+1 or -1)

    The integer conversion  (df + 1) // 2  maps:
      +1  →  (1+1)//2 = 1   (attribute present)
      -1  →  (-1+1)//2 = 0  (attribute absent)
    """
    # header=1 skips the image-count line and uses the attribute-name line as column headers
    df = pd.read_csv(path, sep=r"\s+", header=1, index_col=0)
    df = (df + 1) // 2   # {-1,+1} → {0,1}
    df.index.name = "image_id"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
# Attempt to locate the attribute file; extract from zip if the plain .txt is missing.
# Priority order: plain .txt > bundled .zip
if not os.path.exists(CELEBA_ATTR_PATH):
    if os.path.exists(CELEBA_ZIP_PATH):
        import zipfile
        print(f"Extracting '{CELEBA_ATTR_PATH}' from '{CELEBA_ZIP_PATH}' …")
        with zipfile.ZipFile(CELEBA_ZIP_PATH, "r") as zf:
            zf.extract("list_attr_celeba.txt")
        print("Extraction complete.")
    else:
        raise FileNotFoundError(
            f"CelebA attribute file not found: '{CELEBA_ATTR_PATH}'.\n"
            "Place list_attr_celeba.txt (or list_attr_celeba.zip) in the same "
            "directory as this script."
        )

print(f"Loading CelebA attributes from '{CELEBA_ATTR_PATH}' …")
df = load_celeba_attributes(CELEBA_ATTR_PATH)

print(f"Dataset shape  : {df.shape[0]:,} samples × {df.shape[1]} attributes")
print(f"Target         : {TARGET_ATTR!r}  (positive class = {df[TARGET_ATTR].mean()*100:.1f}%)")
print(f"\nAttribute list :\n{list(df.columns)}\n")

# Separate features (X) from the target label (y).
# X contains the 39 remaining binary attributes as a NumPy array (shape: N × 39).
# y is the binary Smiling label (1 = smiling, 0 = not smiling).
# Storing feature names separately lets us label axes in the importance plot later.
y = df[TARGET_ATTR].values
X = df.drop(columns=[TARGET_ATTR]).values
feature_names = [c for c in df.columns if c != TARGET_ATTR]

# ── Three-way stratified split: 60 % train / 20 % validation / 20 % test ────
# Why three splits?
#   • Training set   — used exclusively to fit model parameters (tree structure, weights).
#   • Validation set — used to select hyperparameters (best depth, best n_estimators)
#                      without ever touching the test set; prevents optimistic bias.
#   • Test set       — touched once at the very end to produce unbiased metric estimates.
#
# stratify=y ensures each split preserves the original ~48 % positive rate.
# Two-step approach: first cut off 20 % as test; then split the remaining 80 %
# into 75 % train and 25 % validation (which equals 60 % / 20 % of the total).
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
# A Decision Tree partitions the feature space into axis-aligned rectangles by
# recursively choosing the split that maximises Gini impurity reduction at each
# node.  Gini impurity for a node is defined as:
#   G = 1 − Σ pᵢ²
# where pᵢ is the proportion of samples belonging to class i.
# A pure node (all one class) has G = 0; a 50/50 mix gives G = 0.5.
#
# The key hyperparameter is max_depth:
#   • Too shallow (depth 1–2) → underfitting: the model is too simple to
#     capture the relevant decision boundaries (high bias).
#   • Too deep (depth 10+)   → overfitting: the model memorises training
#     noise rather than general patterns (high variance).
#   We grid-search depths 1–20 and pick the one with the best validation accuracy.
print("\n" + "=" * 60)
print("SECTION 3 — Decision Tree (Baseline)")
print("=" * 60)

depths = list(range(1, 21))
dt_train_acc, dt_val_acc = [], []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    dt_train_acc.append(accuracy_score(y_train, dt.predict(X_train)))
    dt_val_acc.append(accuracy_score(y_val,   dt.predict(X_val)))

# The depth that maximises validation accuracy is our chosen hyperparameter.
# Using argmax on the validation curve (not training curve!) avoids selecting
# an overfit model that scores perfectly on training data.
best_depth = depths[np.argmax(dt_val_acc)]
print(f"Best max_depth on validation set: {best_depth}  "
      f"(val_acc = {max(dt_val_acc):.4f})")

# Re-train on the combined train+val set with the best depth so the final
# model sees as much labelled data as possible before test evaluation.
dt_best = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_best.fit(X_trainval, y_trainval)
# predict_proba returns [P(class=0), P(class=1)] per sample; we keep column 1
# (probability of "smiling") for ROC curve computation.
dt_proba = dt_best.predict_proba(X_test)[:, 1]
dt_pred  = dt_best.predict(X_test)

# ── Figure 1: DT depth vs accuracy ─────────────────────────────────────────
# This plot is the visual proof of the bias-variance trade-off:
# • Training accuracy (blue) keeps rising — deeper trees fit training data better.
# • Validation accuracy (orange) peaks then drops — deeper trees overfit.
# The vertical dashed line marks the sweet spot chosen automatically above.
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
# Random Forest is a *bagging* ensemble: it trains T independent decision trees,
# each on a different bootstrap sample of the training data, and at each split
# considers only a random subset of √p features (p = total features).
#
# Why does this help?
#   • Averaging T independent estimators reduces variance by a factor of 1/T
#     (assuming zero correlation between trees).
#   • The feature subsampling decorrelates the trees: even when one feature is
#     very strong, different trees use different feature subsets, so they make
#     different errors that cancel out when averaged.
#
# The main hyperparameter is n_estimators (number of trees).
#   • Too few trees → noisy predictions (high variance between runs).
#   • More trees always helps or stays neutral — RF does not overfit with
#     more trees; it just stops improving after a point.
#   We test n_estimators ∈ {10, 25, 50, 100, 200, 300} and pick the best on validation.
print("\n" + "=" * 60)
print("SECTION 4 — Random Forest (Bagging)")
print("=" * 60)

n_est_options = [10, 25, 50, 100, 200, 300]
rf_val_acc = []

for n in n_est_options:
    # n_jobs=-1 uses all available CPU cores to parallelise tree construction
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_val_acc.append(accuracy_score(y_val, rf.predict(X_val)))

best_n_rf = n_est_options[np.argmax(rf_val_acc)]
print(f"Best n_estimators on validation set: {best_n_rf}  "
      f"(val_acc = {max(rf_val_acc):.4f})")

# Re-train the best RF on the full train+val set before evaluating on the test set
rf_best = RandomForestClassifier(n_estimators=best_n_rf, random_state=42, n_jobs=-1)
rf_best.fit(X_trainval, y_trainval)
rf_proba = rf_best.predict_proba(X_test)[:, 1]   # probability of smiling
rf_pred  = rf_best.predict(X_test)

# ── Figure 2: RF n_estimators vs accuracy ──────────────────────────────────
# The curve is expected to rise steeply for small T and plateau for large T.
# Once trees are sufficiently decorrelated, the marginal benefit of adding
# one more tree becomes negligible.
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
# AdaBoost (Adaptive Boosting) is a *sequential* ensemble method.  Unlike
# Random Forest — where all trees are trained independently in parallel —
# AdaBoost trains one weak learner at a time and adjusts sample weights
# after each round to make the next learner focus on previously misclassified
# samples.
#
# Algorithm sketch (Freund & Schapire, 1997):
#   1. Initialise uniform sample weights: wᵢ = 1/N for all i.
#   2. For t = 1, 2, …, T:
#       a. Fit a weak learner hₜ on the weighted training set.
#       b. Compute weighted error: εₜ = Σᵢ wᵢ · 1[hₜ(xᵢ) ≠ yᵢ]
#       c. Compute learner weight:  αₜ = ½ ln((1 − εₜ) / εₜ)
#          → High αₜ for low-error stumps (they contribute more to final vote).
#       d. Update sample weights:
#            wᵢ ← wᵢ · exp(−αₜ yᵢ hₜ(xᵢ))   then renormalise so Σwᵢ = 1.
#          → Misclassified samples get higher weight; correctly classified
#            samples get lower weight.
#   3. Final prediction: H(x) = sign( Σₜ αₜ hₜ(x) )
#
# Key properties:
#   • Reduces bias (the ensemble can represent complex decision boundaries
#     by combining many stumps).
#   • More sensitive to label noise than RF (noisy labels get repeatedly
#     up-weighted, leading to overfitting on noise).
#
# Hyperparameter choices:
#   • max_depth=1  — "decision stump" base learner: a single binary split.
#     Stumps are weak enough to satisfy the theoretical guarantee that each
#     round beats random chance (εₜ < 0.5).
#   • n_estimators=200 — enough rounds to reduce bias substantially on this
#     dataset; overfitting is limited because each stump has very low variance.
#   • learning_rate=0.5 — shrinks each αₜ by 0.5 before adding to the ensemble,
#     trading convergence speed for better generalisation (similar to the
#     step-size in gradient descent).
print("\n" + "=" * 60)
print("SECTION 5 — AdaBoost (Boosting)")
print("=" * 60)

# Weak learner: decision stump (depth=1 tree — splits on exactly one feature)
stump = DecisionTreeClassifier(max_depth=1)
ada = AdaBoostClassifier(
    estimator=stump,
    n_estimators=200,
    learning_rate=0.5,
    random_state=42,
)
# Train directly on train+val (no separate validation needed for hyperparameter
# selection here; we use the fixed choices justified above)
ada.fit(X_trainval, y_trainval)
ada_proba = ada.predict_proba(X_test)[:, 1]   # probability of smiling
ada_pred  = ada.predict(X_test)
print(f"AdaBoost trained  (n_estimators=200, lr=0.5, base=DecisionStump)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
# We report five complementary metrics for each model:
#   Accuracy  = (TP + TN) / N           — overall fraction correct
#   Precision = TP / (TP + FP)          — "when we predict smile, are we right?"
#   Recall    = TP / (TP + FN)          — "of all actual smiles, how many found?"
#   F1        = 2·(P·R)/(P+R)           — harmonic mean of precision & recall
#   ROC-AUC   = area under the ROC curve — ranking quality, threshold-independent
#
# Why multiple metrics?
#   A model that predicts "not smiling" 100 % of the time would achieve ~52 %
#   accuracy (since ~48 % are positive).  Precision, recall, and AUC expose
#   such a degenerate model immediately.  Using all five gives a complete picture.
print("\n" + "=" * 60)
print("SECTION 6 — Evaluation on Test Set")
print("=" * 60)


def metrics(y_true, y_pred, y_proba, name):
    """Compute and return a dict of evaluation metrics for one model.

    Parameters
    ----------
    y_true  : array of true binary labels (0 or 1)
    y_pred  : array of predicted binary labels
    y_proba : array of predicted probabilities for the positive class
    name    : human-readable model name for display

    Returns
    -------
    dict with keys: Model, Accuracy, Precision, Recall, F1, ROC-AUC
    """
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
# ROC (Receiver Operating Characteristic) curve: for each decision threshold θ,
# plot TPR = TP/(TP+FN) vs FPR = FP/(FP+TN).
# Sweeping θ from 1 → 0 traces from (0,0) to (1,1).
# AUC (Area Under Curve) summarises the curve in one number:
#   • AUC = 1.0 → perfect classifier
#   • AUC = 0.5 → no better than random guessing (diagonal line)
# A higher AUC means the model assigns higher probabilities to true positives
# than to true negatives, regardless of the chosen threshold.
fig, ax = plt.subplots(figsize=(7, 6))
for (name, proba), color in zip(
    [("Decision Tree", dt_proba), ("Random Forest", rf_proba), ("AdaBoost", ada_proba)],
    COLORS,
):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=color, lw=2)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")   # baseline diagonal
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curves — Predicting '{TARGET_ATTR}' (CelebA)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("fig3_roc_curves.png", dpi=150)
plt.close(fig)
print("\nSaved fig3_roc_curves.png")

# ── Figure 4: Feature Importance (Random Forest) ────────────────────────────
# sklearn's feature_importances_ attribute stores the Mean Decrease in Impurity
# (MDI) averaged over all trees and all nodes in which each feature was used.
# Higher MDI → that feature contributed more to reducing node impurity (Gini)
# across the forest.
# We display the top 15 features; the full ranking is stored in `importances`.
importances = rf_best.feature_importances_
sorted_idx  = np.argsort(importances)[-15:]    # indices of top-15 features

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(
    [feature_names[i] for i in sorted_idx],   # feature labels on y-axis
    importances[sorted_idx],                   # importance values on x-axis
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
# Grouped bar chart comparing all five metrics side-by-side for each model.
# Numeric labels are placed above each bar for easy comparison.
fig, ax = plt.subplots(figsize=(10, 5))
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
x = np.arange(len(metrics_to_plot))
width = 0.25   # bar width; three models × 0.25 = 0.75 of each group slot

for i, (model, color) in enumerate(zip(results_df.index, COLORS)):
    vals = results_df.loc[model, metrics_to_plot].values.astype(float)
    bars = ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
    # Annotate each bar with its numeric value for readability
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
# This section summarises each model's strengths, weaknesses, and performance,
# and explains *why* the results came out the way they did.
print("\n" + "=" * 60)
print("SECTION 7 — Discussion")
print("=" * 60)

# Identify the best model by F1 score (balances precision and recall)
best_model = results_df["F1"].idxmax()
print(f"""
Decision Tree (Baseline)
  - Simple, interpretable, but prone to overfitting (high variance).
  - Depth tuning reveals the bias–variance trade-off clearly (Fig. 1):
    too shallow → underfits; too deep → memorises noise.
  - Accuracy: {results_df.loc['Decision Tree','Accuracy']:.4f}

Random Forest (Bagging)
  - Trains many diverse trees on bootstrap samples and averages votes.
  - Diversity comes from (a) different data subsets and (b) random feature
    selection at each split, which decorrelates the trees.
  - Reduces variance without significantly increasing bias.
  - Accuracy: {results_df.loc['Random Forest','Accuracy']:.4f}
  - Most informative features: {', '.join([feature_names[i] for i in sorted_idx[-3:]][::-1])}.

AdaBoost (Boosting)
  - Trains weak learners (stumps) sequentially; each round up-weights
    misclassified samples so the next stump corrects prior errors.
  - Reduces bias; can overfit on noisy data if n_estimators is too large
    or the labels contain significant noise.
  - Accuracy: {results_df.loc['AdaBoost','Accuracy']:.4f}

Best model by F1: {best_model}
  → Ensemble methods outperform the single Decision Tree, demonstrating
    that combining learners improves both accuracy and robustness.
  → Bagging (RF) and boosting (AdaBoost) address *different* sources of
    error: RF lowers variance by averaging; AdaBoost lowers bias by
    iterative error correction.  Both strategies are valid but suited to
    different problem characteristics (noisy data favours RF; low-bias
    tasks favour boosting).
""")

print("=" * 60)
print("All figures saved. Include them + this output in your PDF submission.")
print("=" * 60)
