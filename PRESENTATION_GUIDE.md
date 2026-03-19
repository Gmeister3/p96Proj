# Presentation Guide — Project 3: Ensemble Learning

> **Why this project is the easiest to present**
> The entire story fits a single clear arc that anyone can follow:
> *"One tree makes mistakes → many trees fix each other → a smarter trainer focuses on hard cases."*
> Every slide has a picture. No math-heavy derivations required.

---

## Suggested Slide Structure (~10–12 minutes)

### Slide 1 — Title & Motivation (1 min)
**Say:** "We're trying to automatically detect whether someone in a photo is smiling —
using only other facial feature labels, no raw pixels."
- Show: task name, dataset name (CelebA), target attribute (Smiling)
- One-liner on why this matters: face recognition, emotion detection, content moderation

---

### Slide 2 — Dataset Overview (1 min)
**Say:** "CelebA has 202,000 celebrity images each tagged with 40 binary facial attributes.
We drop the Smiling label and use the remaining 39 to predict it."
- Show: list of attribute names, class balance (~48% smiling), train/val/test split

---

### Slide 3 — Baseline: Decision Tree (2 min)
**Say:** "We start with the simplest tree-based classifier. A single tree recursively
splits the data by the most informative attribute at each node."
- Show: **Fig 1 (depth vs accuracy)** — point out the sweet spot at depth 3
- Key point: "shallow = underfit (high bias), deep = overfit (high variance)
  — this is the bias–variance trade-off in action"

---

### Slide 4 — Bagging: Random Forest (2 min)
**Say:** "Random Forest grows many trees, each on a random bootstrap sample
and a random subset of features. We take a majority vote at the end."
- Show: **Fig 2 (n_estimators vs accuracy)** — accuracy stabilises after ~100 trees
- Key point: "many uncorrelated trees average out each other's random errors → lower variance"

---

### Slide 5 — Boosting: AdaBoost (2 min)
**Say:** "AdaBoost trains weak learners (stumps, depth=1) one by one.
After each round it *up-weights* the samples that were misclassified,
so the next stump focuses on the hard cases."
- Diagram (draw on whiteboard or add a simple arrow chain): stump₁ → reweight → stump₂ → reweight → …
- Key point: "bagging reduces variance; boosting reduces bias"

---

### Slide 6 — Results: ROC Curves (1.5 min)
**Say:** "The ROC curve shows the trade-off between catching real smiles (recall)
and avoiding false positives (specificity) across every decision threshold."
- Show: **Fig 3 (ROC curves)** — ensemble methods sit above the DT baseline
- Point out AUC values in the legend

---

### Slide 7 — Results: Feature Importance (1 min)
**Say:** "Random Forest tells us which attributes were most useful.
The top predictors — Mouth_Slightly_Open, High_Cheekbones, Attractive — align with
what we'd intuitively expect for a smiling face."
- Show: **Fig 4 (feature importance bar chart)**

---

### Slide 8 — Results: Metric Comparison (1.5 min)
**Say:** "Across all five metrics, ensembles match or beat the single tree."
- Show: **Fig 5 (grouped bar chart)** — walk through accuracy, precision, recall, F1, AUC
- Brief definition of each metric if needed: "Precision = of all faces we called smiling,
  how many really were? Recall = of all actually smiling faces, how many did we catch?"

---

### Slide 9 — Discussion & Conclusion (1 min)
**Say:** "Ensemble methods reduce different sources of error.
Bagging (RF) cuts variance. Boosting (AdaBoost) cuts bias.
Both outperform a single tree, confirming the well-known result that combining
learners produces more robust classifiers."
- One sentence on limitations: "Results here use attribute labels only, not raw pixels.
  A CNN operating directly on images would push accuracy much higher."

---

## Quick-Reference Talking Points

| Concept | One-sentence explanation |
|---|---|
| Decision Tree | Split data greedily on the most informative feature at each node |
| Bias–Variance | Simple models underfit (high bias); complex models overfit (high variance) |
| Bagging | Train many models on random subsets; average → lower variance |
| Boosting | Train models sequentially; focus on mistakes → lower bias |
| Random Forest | Bagging + random feature subsets → decorrelated trees |
| AdaBoost | Boosting with exponential loss; each weak learner fixes prior errors |
| Accuracy | (TP + TN) / total — misleading if classes are imbalanced |
| Precision | TP / (TP + FP) — "of predictions made, how many were right?" |
| Recall | TP / (TP + FN) — "of real positives, how many did we catch?" |
| F1 | Harmonic mean of precision & recall |
| ROC-AUC | Probability that a random positive ranks above a random negative |

---

## Demo Tips

- If asked *"why not just use more depth for the decision tree?"*:
  > "Fig 1 shows validation accuracy dropping after depth 3 — the tree memorises
  > training noise instead of learning general patterns."

- If asked *"why doesn't Random Forest do better than a single tree here?"*:
  > "Our synthetic data has low feature correlation and the key signal is concentrated in
  > only ~3 attributes.  With real CelebA images (pixel features), RF would pull further ahead."

- If asked *"how is AdaBoost different from Gradient Boosting?"*:
  > "AdaBoost re-weights samples and uses exponential loss.  Gradient Boosting frames each
  > round as fitting the residuals of a differentiable loss function — more general and
  > usually more accurate, but harder to tune."

---

## Submission Checklist

- [ ] `project3_ensemble_learning.py` — full Python source, well commented
- [ ] All five `fig*.png` outputs embedded in the PDF
- [ ] Printed console output (metrics table + discussion) in the PDF
- [ ] Names and student numbers on the first page
- [ ] Single PDF uploaded to Brightspace before **March 25, 2026 (11:59 PM)**
