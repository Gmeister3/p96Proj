# Presentation Guide — Project 3: Ensemble Learning

> **Why this project is the easiest to present**
> The entire story fits a single clear arc that anyone can follow:
> *"One tree makes mistakes → many trees fix each other → a smarter trainer focuses on hard cases."*
> Every slide has a picture. No math-heavy derivations required.
>
> **Narrative thread to keep in mind throughout:**
> Start with *why* a single decision tree is fragile, then show *two* fundamentally different
> fixes — averaging (bagging) vs. sequential error-correction (boosting) — and let the figures
> prove which approach wins on this task.

---

## Suggested Slide Structure (~10–12 minutes)

### Slide 1 — Title & Motivation (1 min)
**Say:** "We're trying to automatically detect whether someone in a photo is smiling —
using only other facial feature labels, no raw pixels."
- Show: task name, dataset name (CelebA), target attribute (Smiling)
- One-liner on why this matters: face recognition, emotion detection, content moderation
- **Deeper note:** Emphasise that this is a *weakly supervised* proxy task — we are predicting
  one binary label from 39 other binary labels, which means every feature is already a
  human-provided semantic signal (e.g. "wearing lipstick", "high cheekbones").  This is
  intentionally simpler than raw pixels so we can focus on the *classifier* design, not
  the feature engineering.
- If time allows: mention that CelebA is one of the most widely used benchmarks in computer
  vision fairness research because of its scale (202 K images) and diverse celebrity pool.

---

### Slide 2 — Dataset Overview (1 min)
**Say:** "CelebA has 202,000 celebrity images each tagged with 40 binary facial attributes.
We drop the Smiling label and use the remaining 39 to predict it."
- Show: list of attribute names, class balance (~48% smiling), train/val/test split
- **Deeper note:** The ~48 % positive rate is almost perfectly balanced, which means plain
  accuracy is a meaningful metric here (no need for class-weighted scoring).  Point this out
  explicitly — in real-world problems class imbalance often forces us to optimise F1 or AUC
  instead of accuracy.
- Explain the **three-way split (60 / 20 / 20)**:
  - *Training set* — used to fit model weights / tree structure.
  - *Validation set* — used to pick hyperparameters (depth, n_estimators) without touching
    test data.  This prevents optimistic bias.
  - *Test set* — held out until the very end; gives an unbiased estimate of real-world
    performance.  **Never** tune on the test set — that would invalidate the evaluation.
- Stratified splitting (`stratify=y`) is used so each split preserves the 48 % positive
  rate; mention this if asked.

---

### Slide 3 — Baseline: Decision Tree (2 min)
**Say:** "We start with the simplest tree-based classifier. A single tree recursively
splits the data by the most informative attribute at each node."
- Show: **Fig 1 (depth vs accuracy)** — point out the sweet spot at depth 3
- Key point: "shallow = underfit (high bias), deep = overfit (high variance)
  — this is the bias–variance trade-off in action"
- **Deeper note on how splits work:** At each internal node the algorithm evaluates every
  remaining feature and every possible threshold (here: 0 vs 1 for binary attributes).
  It picks the split that maximises *Gini impurity reduction* (or equivalently, information
  gain).  Gini impurity for a node is 1 − Σpᵢ², where pᵢ is the fraction of samples
  belonging to class i.  A pure node (all one class) has Gini = 0; a 50/50 split has
  Gini = 0.5.
- **Deeper note on Fig 1:** The training curve keeps rising as depth increases because the
  tree can memorise every training sample given enough nodes.  The validation curve peaks
  around depth 3 then falls — the tree starts fitting noise rather than signal.  This
  *gap* between training and validation accuracy is the definition of overfitting.
- Mention `random_state=42` is set throughout for reproducibility.

---

### Slide 4 — Bagging: Random Forest (2 min)
**Say:** "Random Forest grows many trees, each on a random bootstrap sample
and a random subset of features. We take a majority vote at the end."
- Show: **Fig 2 (n_estimators vs accuracy)** — accuracy stabilises after ~100 trees
- Key point: "many uncorrelated trees average out each other's random errors → lower variance"
- **Deeper note on bootstrap sampling:** Each tree is trained on a *bootstrap sample* —
  a random draw of N samples *with replacement* from the training set.  On average ~63 %
  of unique samples appear in each bootstrap sample; the remaining ~37 % are the "out-of-bag"
  (OOB) samples, which can be used as a free internal validation set.
- **Deeper note on feature randomness:** At each split, Random Forest only considers a
  random subset of `√p` features (where p = total features).  This *decorrelates* the
  trees — if one strong feature would dominate every tree, the trees would all be nearly
  identical and averaging them would not help much.  By blocking that feature from some
  splits, weaker but complementary features get a chance to contribute.
- **Deeper note on Fig 2:** Accuracy improves quickly with the first few trees (high
  marginal gain) and flattens after ~100 because each additional tree adds less new
  information.  Adding more trees never *hurts* — it just stops helping.  The validation
  sweet spot (best_n) is recorded and the final model is re-trained on train + val
  combined to maximise data usage before test evaluation.

---

### Slide 5 — Boosting: AdaBoost (2 min)
**Say:** "AdaBoost trains weak learners (stumps, depth=1) one by one.
After each round it *up-weights* the samples that were misclassified,
so the next stump focuses on the hard cases."
- Diagram (draw on whiteboard or add a simple arrow chain): stump₁ → reweight → stump₂ → reweight → …
- Key point: "bagging reduces variance; boosting reduces bias"
- **Deeper note on sample weights:** Before round t=1 all N samples have equal weight 1/N.
  After each round, a stump's *voting weight* αₜ = ½ ln((1−εₜ)/εₜ) is computed, where
  εₜ is the weighted error rate.  Stumps with lower error get larger αₜ and contribute
  more to the final vote.  Sample weights are then multiplied by exp(αₜ) for correctly
  classified samples and exp(−αₜ) for misclassified ones (then renormalised).  This forces
  subsequent stumps to attend to the samples the ensemble currently struggles with.
- **Deeper note on why stumps?** A stump (max_depth=1) is a *weak learner* — it barely
  beats random guessing (AUC ≈ 0.55–0.65).  AdaBoost's theoretical guarantee is that as
  long as each weak learner does *slightly* better than chance, the combined model's
  training error decreases exponentially with the number of rounds.
- **Hyperparameter note:** `learning_rate=0.5` shrinks each stump's contribution, trading
  speed of convergence for better generalisation (similar to shrinkage in gradient boosting).
  `n_estimators=200` gives the ensemble enough rounds to reduce bias; overfitting is
  less of a concern here because the base learner is so shallow.
- **Bagging vs. Boosting summary:**
  | | Bagging (RF) | Boosting (AdaBoost) |
  |---|---|---|
  | Trees built | In parallel | Sequentially |
  | Main goal | Reduce variance | Reduce bias |
  | Sample strategy | Bootstrap (random) | Reweight hard samples |
  | Sensitive to noise | No | Yes (noisy labels inflate weights) |

---

### Slide 6 — Results: ROC Curves (1.5 min)
**Say:** "The ROC curve shows the trade-off between catching real smiles (recall)
and avoiding false positives (specificity) across every decision threshold."
- Show: **Fig 3 (ROC curves)** — ensemble methods sit above the DT baseline
- Point out AUC values in the legend
- **Deeper note on how ROC curves are constructed:** For each possible decision threshold
  θ ∈ [0, 1], we classify a sample as positive if predicted probability ≥ θ.  Sweeping θ
  from 1 → 0 traces a curve from (0,0) to (1,1) in (FPR, TPR) space.  A perfect model
  goes straight up then right (AUC = 1.0); a random coin flip lies on the diagonal
  (AUC = 0.5).
- **Why AUC is useful here:** AUC is threshold-independent and class-imbalance-robust.
  It equals the probability that a randomly chosen positive sample is ranked above a
  randomly chosen negative sample by the model.  Even if accuracy values are similar,
  a higher AUC means the model has better *ranking* quality.
- **Reading Fig 3:** Ensemble curves bulge further toward the top-left corner than the DT
  curve.  The gap is largest at low FPR values (strict operating point), which is exactly
  where you'd want to be in a real content-moderation application.

---

### Slide 7 — Results: Feature Importance (1 min)
**Say:** "Random Forest tells us which attributes were most useful.
The top predictors — Mouth_Slightly_Open, High_Cheekbones, Attractive — align with
what we'd intuitively expect for a smiling face."
- Show: **Fig 4 (feature importance bar chart)**
- **Deeper note on what importance measures:** The values shown are *Mean Decrease in
  Impurity* (MDI) — averaged over all trees and all nodes where that feature was used
  to split.  A feature with high MDI was frequently chosen and reduced node impurity
  substantially each time.  MDI can overestimate importance for high-cardinality features,
  but since all our features are binary (low cardinality), this bias is minimal here.
- **Intuition check:** "Mouth_Slightly_Open" scoring at the top makes anatomical sense —
  smiling typically exposes teeth and parts the lips.  "High_Cheekbones" and "Attractive"
  correlate because smiling raises the cheeks and is considered an attractive expression.
  This sanity check of feature importance against domain knowledge builds confidence that
  the model has learned something real rather than spurious correlations.
- If challenged: mention that MDI importance is *not* the same as causal importance or
  the importance you'd get from permutation tests or SHAP values.

---

### Slide 8 — Results: Metric Comparison (1.5 min)
**Say:** "Across all five metrics, ensembles match or beat the single tree."
- Show: **Fig 5 (grouped bar chart)** — walk through accuracy, precision, recall, F1, AUC
- Brief definition of each metric if needed: "Precision = of all faces we called smiling,
  how many really were? Recall = of all actually smiling faces, how many did we catch?"
- **Deeper note — what each metric reveals:**
  - *Accuracy* (TP+TN)/N — overall correctness; only reliable when classes are balanced
    (which they roughly are here at ~48 %).
  - *Precision* TP/(TP+FP) — "when the model says smile, how often is it right?" —
    important when false positives are costly (e.g. wrongly flagging a neutral face).
  - *Recall* TP/(TP+FN) — "of all actual smiles, how many did the model find?" —
    important when false negatives are costly (e.g. missing a smile in an ID photo check).
  - *F1* = 2·(P·R)/(P+R) — harmonic mean; punishes models that sacrifice one of P/R
    to inflate the other.  More informative than accuracy alone when class balance is
    imperfect.
  - *ROC-AUC* — ranking quality independent of threshold; the most robust single
    number for comparing classifiers.
- **What to say if the numbers look similar:** "The differences appear small in absolute
  terms, but remember: with 40,000 test samples even a 0.5 % accuracy gain represents
  ~200 extra correct predictions.  In production at scale, that matters."

---

### Slide 9 — Discussion & Conclusion (1 min)
**Say:** "Ensemble methods reduce different sources of error.
Bagging (RF) cuts variance. Boosting (AdaBoost) cuts bias.
Both outperform a single tree, confirming the well-known result that combining
learners produces more robust classifiers."
- One sentence on limitations: "Results here use attribute labels only, not raw pixels.
  A CNN operating directly on images would push accuracy much higher."
- **Deeper note on limitations and future work:**
  - *Label noise:* Human annotators labelling 202 K images will make mistakes; boosting
    is particularly sensitive to noisy labels because it keeps up-weighting mislabelled
    samples.
  - *Feature engineering ceiling:* Binary attribute labels compress a rich image into 39
    bits.  A convolutional network operating on raw pixels could capture subtle texture
    and shape cues invisible to our feature set.
  - *Fairness considerations:* CelebA is celebrity-skewed (lighter skin tones, specific
    beauty standards).  A model trained here may not generalise to diverse populations.
    This is an active research area in algorithmic fairness.
  - *Next steps:* Gradient Boosted Trees (XGBoost, LightGBM) often outperform AdaBoost
    in practice; a grid search over learning rate and tree depth would likely improve
    results further.

---

## Quick-Reference Talking Points

| Concept | One-sentence explanation | Deeper detail |
|---|---|---|
| Decision Tree | Split data greedily on the most informative feature at each node | Uses Gini impurity or information gain; fully grown tree can memorise any training set |
| Gini Impurity | Measures node purity: 1 − Σpᵢ² | Zero for a pure node; 0.5 for a perfectly mixed binary node |
| Bias–Variance | Simple models underfit (high bias); complex models overfit (high variance) | Total expected error = bias² + variance + irreducible noise |
| Bootstrap Sample | Random sample of N items drawn with replacement | ~63 % unique items appear; remaining ~37 % are out-of-bag (OOB) |
| Bagging | Train many models on random subsets; average → lower variance | Works because averaging independent estimators reduces variance by factor 1/T |
| Boosting | Train models sequentially; focus on mistakes → lower bias | Weak learner guarantee: each round must only beat random chance (AUC > 0.5) |
| Random Forest | Bagging + random feature subsets → decorrelated trees | Feature subset size √p at each split prevents any single strong feature from dominating |
| AdaBoost | Boosting with exponential loss; each weak learner fixes prior errors | Stump vote weight αₜ = ½ ln((1−εₜ)/εₜ); larger for lower-error stumps |
| Accuracy | (TP + TN) / total — misleading if classes are imbalanced | Can be gamed by predicting majority class always; use F1/AUC instead when skewed |
| Precision | TP / (TP + FP) — "of predictions made, how many were right?" | High precision = few false alarms |
| Recall | TP / (TP + FN) — "of real positives, how many did we catch?" | High recall = few missed detections |
| F1 | Harmonic mean of precision & recall | Harmonic mean penalises imbalance between P and R more than arithmetic mean |
| ROC-AUC | Probability that a random positive ranks above a random negative | Equivalent to the area under the ROC curve; 1.0 = perfect, 0.5 = random |
| MDI Feature Importance | Average impurity reduction across all splits using a feature | Can be biased for high-cardinality features; consider permutation importance as an alternative |

---

## Demo Tips

- If asked *"why not just use more depth for the decision tree?"*:
  > "Fig 1 shows validation accuracy dropping after depth 3 — the tree memorises
  > training noise instead of learning general patterns.  At depth 20 the training
  > accuracy is near 100 % but validation accuracy is lower than depth 3, which is
  > a textbook illustration of overfitting."

- If asked *"why doesn't Random Forest do better than a single tree here?"*:
  > "Our attribute features are already highly processed binary labels with a clear
  > dominant signal (Mouth_Slightly_Open).  With real image pixels or continuous features
  > RF would pull much further ahead, because there the diversity of tree views matters
  > more.  Even here RF's AUC and F1 are measurably above the single tree."

- If asked *"how is AdaBoost different from Gradient Boosting?"*:
  > "AdaBoost re-weights samples and uses exponential loss with a fixed base learner.
  > Gradient Boosting (XGBoost, LightGBM) frames each round as fitting the *gradient*
  > of a differentiable loss function — this generalises to any loss (log-loss, MSE,
  > quantile) and usually gives better results.  The cost is more hyperparameters to tune."

- If asked *"why use a stump (depth=1) as the weak learner for AdaBoost?"*:
  > "A stump makes a single binary decision based on one feature.  It is weak enough
  > that the theoretical guarantees hold (each round slightly beats random chance), yet
  > computationally cheap to train.  Deeper base learners tend to overfit with boosting
  > because the algorithm already combines many of them."

- If asked *"what is the learning_rate parameter in AdaBoost?"*:
  > "It shrinks each stump's contribution to the ensemble by a factor lr before adding it.
  > Lower lr means you need more stumps (higher n_estimators) to reach the same training
  > error, but the final model generalises better — similar to step-size shrinkage in
  > gradient descent."

- If asked *"what does 'stratify=y' mean in the data split?"*:
  > "It ensures each split contains roughly the same fraction of positive samples as the
  > full dataset (~48 % smiling).  Without stratification a random split could produce a
  > test set with 60 % positives purely by chance, making metrics non-comparable."

- If asked *"could we use cross-validation instead of a validation set?"*:
  > "Yes — k-fold cross-validation would give a lower-variance estimate of generalisation
  > error.  We used a single hold-out split here for speed given the 202 K-sample dataset,
  > but on smaller datasets k-fold is strongly preferred."

---

## Common Misconceptions to Pre-empt

| Misconception | Correction |
|---|---|
| "More trees always overfit" | Adding trees to RF never increases variance; worst case they add nothing |
| "AdaBoost is the same as Random Forest" | RF = parallel + averaging (bagging); AdaBoost = sequential + reweighting (boosting) |
| "Accuracy is the best metric" | Only true for balanced classes; use AUC or F1 when classes are skewed |
| "Feature importance means causation" | MDI measures predictive correlation, not causal influence |
| "Deeper trees learn better" | After the sweet spot, deeper trees overfit — see Fig 1 |
| "Ensembles always beat single models" | Ensembles reduce *reducible* error; if the base learner has already saturated signal, gains are small |

---

## Submission Checklist

- [ ] `project3_ensemble_learning.py` — full Python source, well commented
- [ ] All five `fig*.png` outputs embedded in the PDF
- [ ] Printed console output (metrics table + discussion) in the PDF
- [ ] Names and student numbers on the first page
- [ ] Single PDF uploaded to Brightspace before **March 25, 2026 (11:59 PM)**
