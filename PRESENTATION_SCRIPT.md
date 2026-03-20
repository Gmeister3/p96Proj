# Presentation Script — Project 3: Ensemble Learning
### COSC 3P96 — Machine Learning | CelebA Smiling Classifier

> **How to use this script**
> This is a full word-for-word spoken script for a ~10–12 minute presentation.
> Text in *italics* are stage directions (slide changes, pointing at figures, etc.).
> Text in **[brackets]** are optional expansions — include them if time allows or if
> you are asked a question mid-slide.
> Practice the script aloud at least twice so it feels natural, not read.

---

## Slide 1 — Title & Motivation (~1 minute)

*[Display title slide: "Project 3: Ensemble Learning for Robust Facial Attribute
Classification — CelebA Dataset"]*

---

"Good [morning / afternoon], everyone.

For Project 3, our task was to build a classifier that can automatically
detect whether a person in a photo is smiling — using *only* other facial
attribute labels from the CelebA dataset, no raw pixel data at all.

CelebA is one of the most widely used datasets in computer-vision research.
It contains around 202,000 celebrity photos, and every photo has been
human-labelled with 40 binary facial attributes — things like
'wearing lipstick', 'has high cheekbones', or 'has a beard'.

Our target attribute is **Smiling**.
We drop that label, and then we use the remaining 39 attributes as
input features to predict whether or not the person is smiling.

Why does this matter?
Detecting smiles automatically is useful in face recognition systems,
emotion detection in human-computer interaction, and content moderation
on social media platforms.

**[** What makes this setup particularly interesting is that every single
feature is already a high-level human-provided semantic signal.
That lets us focus entirely on *classifier design* — comparing different
tree-based models — rather than worrying about feature engineering. **]**

So let's walk through what we did."

---

## Slide 2 — Dataset Overview (~1 minute)

*[Display slide showing attribute list, class balance bar chart, and
train/val/test split diagram]*

---

"Before we train anything, let's look at the data.

CelebA has 202,000 images, each tagged with 40 binary attributes.
After dropping 'Smiling' we have **39 input features**, all binary — either
+1 or −1 in the raw file, which we convert to 1 and 0.

Crucially, the dataset is **almost perfectly balanced** —
about 48% of the photos are labelled as smiling.
That's important because it means plain accuracy is actually a meaningful
metric here. We don't need to worry about class-weighted scoring or
artificially rebalancing the dataset.

We split the data into **three parts using a 60 / 20 / 20 ratio**:

- The **training set** is what we actually fit our models on.
- The **validation set** is used to tune hyperparameters — things like
  tree depth and number of estimators — without ever touching test data.
  This prevents us from accidentally over-fitting our choices to the test set.
- The **test set** is held out entirely until the very end, so our final
  accuracy numbers represent genuinely unseen data.

We use **stratified splitting** throughout, which ensures each split
preserves that ~48% positive rate — so none of the three sets gets an
unusually high or low proportion of smiling faces by chance."

---

## Slide 3 — Baseline: Decision Tree (~2 minutes)

*[Display slide with Fig 1 — depth vs accuracy curves for training and validation]*

---

"We start with the simplest model: a **single decision tree**.

A decision tree recursively splits the training data.
At each internal node, it evaluates every possible feature and every
possible threshold and picks the split that most reduces *Gini impurity* —
a measure of how mixed the class labels are in a node.
Gini impurity is defined as 1 minus the sum of squared class proportions.
A perfectly pure node — all one class — has a Gini of zero.
A perfectly mixed 50-50 node has a Gini of 0.5.

The key hyperparameter is **maximum depth** — how many levels of splits
the tree is allowed to make.

*[Point at Fig 1]*

Fig 1 shows what happens as we vary depth from 1 to 20.
The **blue line** is training accuracy — it keeps climbing as the tree
gets deeper because a fully grown tree can simply memorise the
training set.
The **orange line** is validation accuracy — it peaks around **depth 3**
and then starts to fall.

That gap between the two curves is the textbook definition of **overfitting**:
the tree is learning noise instead of signal.

This is the **bias–variance trade-off** in action.
A very shallow tree has high bias — it underfits, making systematic errors.
A very deep tree has high variance — it overfits, being too sensitive to
the specific training samples.

The sweet spot at depth 3 is what we use as our Decision Tree baseline.

**[** We set `random_state=42` everywhere for reproducibility —
so every run of the script produces identical results. **]**"

---

## Slide 4 — Bagging: Random Forest (~2 minutes)

*[Display slide with Fig 2 — n_estimators vs accuracy]*

---

"Now let's fix the single tree's variance problem with **Random Forest**.

The idea behind Random Forest is elegantly simple:
instead of one tree, grow **many trees**, and take a majority vote.

There are two sources of randomness that make this work:

First, each tree is trained on a **bootstrap sample** — a random draw of
N training samples *with replacement*.
On average, about 63% of unique training samples appear in any one
bootstrap sample.
The rest — the so-called 'out-of-bag' samples — could be used as a
free internal validation set.

Second, at each split, the tree only considers a **random subset of features**
— specifically the square root of the total number of features.
This *decorrelates* the trees.
If we let every tree see every feature at every split, they would all
focus on the same strongest signal and end up nearly identical —
averaging them would barely help.
By blocking some features from some splits, weaker but complementary
features get to contribute, and the trees become genuinely diverse.

*[Point at Fig 2]*

Fig 2 shows Random Forest validation accuracy as we increase the number
of trees from 1 to 200.
Accuracy jumps up quickly with the first few trees, then **levels off
around 100 trees**.
This is typical — each additional tree contributes less new information.
Importantly, adding more trees **never hurts** — it just stops helping.

We record the best number of trees from validation, then retrain on the
combined training-plus-validation set before final test evaluation, to
make maximum use of available data.

The key insight of bagging:
**averaging many uncorrelated estimators reduces variance** by approximately
a factor of 1 over T, where T is the number of trees."

---

## Slide 5 — Boosting: AdaBoost (~2 minutes)

*[Display slide with a diagram of sequential stump training and reweighting]*

---

"Our third model takes a completely different approach: **AdaBoost**.

Instead of building trees in parallel and averaging them,
AdaBoost builds **weak learners sequentially**.
Each new learner focuses on the mistakes of the ones before it.

Specifically, we use **decision stumps** — trees of maximum depth 1,
which make a single binary decision based on a single feature.
A stump is a *weak learner*: it barely beats random guessing.
But AdaBoost's theoretical guarantee is that as long as each weak learner
does *slightly* better than chance, combining enough of them drives the
ensemble's training error down exponentially.

Here's how the reweighting works:
- Round 1: all N samples have equal weight 1/N.
- After fitting a stump, we compute its weighted error rate — call it epsilon.
- The stump's voting weight, alpha, is ½ × ln((1 − epsilon) / epsilon).
  A lower error rate gives a larger alpha — that stump gets more say
  in the final vote.
- Then we multiply the weights of **misclassified** samples by a large
  factor and **renormalise**, so the next stump is forced to attend to
  the cases the ensemble currently gets wrong.

We use **200 stumps** and a **learning rate of 0.5**,
which shrinks each stump's contribution and trades speed of convergence
for better generalisation — similar to step-size shrinkage in gradient descent.

The contrast between the two ensemble strategies is worth summarising:

| | Random Forest | AdaBoost |
|---|---|---|
| Trees built | In parallel | Sequentially |
| Main benefit | Reduces **variance** | Reduces **bias** |
| Sample strategy | Bootstrap (random) | Reweight hard samples |
| Sensitive to noise | No | Yes |

A single noisy label inflates its weight in AdaBoost round after round,
which is one of its known weaknesses.
Random Forest is not affected because each noisy sample only shows up
in some bootstrap samples."

---

## Slide 6 — Results: ROC Curves (~1.5 minutes)

*[Display slide with Fig 3 — ROC curves for all three models, with AUC in legend]*

---

"Let's look at the results.

*[Point at Fig 3]*

This is the **ROC curve** — Receiver Operating Characteristic.
For each possible decision threshold, we ask:
'What fraction of true smiles do we correctly detect — the True Positive
Rate — and what fraction of non-smiles do we wrongly flag — the False
Positive Rate?'

Sweeping the threshold from 1 down to 0 traces a curve from
the bottom-left to the top-right of this plot.
A **perfect model** goes straight up then right — hugging the top-left corner.
A **random coin flip** lies on the diagonal.

The area under the curve — **AUC** — is a single number that summarises
ranking quality.
It equals the probability that a randomly chosen smiling face is ranked
above a randomly chosen non-smiling face by the model.

*[Trace the curves with a pointer]*

You can see that **both ensemble methods sit above the Decision Tree baseline**,
especially at low False Positive Rates — the region that matters most in
strict real-world operating conditions like content moderation,
where we really do not want to incorrectly flag neutral faces."

---

## Slide 7 — Results: Feature Importance (~1 minute)

*[Display slide with Fig 4 — horizontal bar chart of top-15 RF features]*

---

"One of the nice properties of Random Forest is that it gives us
**feature importance scores** automatically.

*[Point at Fig 4]*

The values shown are *Mean Decrease in Impurity* —
averaged across all trees and all nodes where that feature was used to split.
A high value means the feature was frequently chosen and reduced
node impurity substantially each time it was used.

The top predictor is **Mouth_Slightly_Open**, followed by
**High_Cheekbones** and **Attractive**.

This passes an intuitive sanity check:
- Smiling typically parts the lips and exposes the teeth — hence
  'Mouth_Slightly_Open' at the top.
- Smiling raises the cheeks — hence 'High_Cheekbones'.
- Smiling is considered an attractive expression — hence 'Attractive'.

The fact that the model has latched onto anatomically and socially
meaningful features, rather than arbitrary correlations, gives us
confidence it has learned something real.

**[** A note of caution: MDI importance measures *predictive correlation*,
not causation.
If you want a more robust importance measure, you'd use permutation
importance or SHAP values, which are threshold-independent.
MDI can over-estimate importance for high-cardinality features, but since
all our features are binary, that bias is minimal here. **]**"

---

## Slide 8 — Results: Metric Comparison (~1.5 minutes)

*[Display slide with Fig 5 — grouped bar chart: Accuracy / Precision / Recall / F1 / AUC]*

---

"Finally, let's compare all three models across five metrics.

*[Point at Fig 5]*

We have three groups — Decision Tree in **blue**, Random Forest in
**orange**, AdaBoost in **green** — and five metrics across the x-axis.

Let me quickly define each:

- **Accuracy** — overall fraction correct. Reliable here because our
  classes are balanced at ~48%.
- **Precision** — of all the faces we labelled as smiling, how many
  really were? High precision means few false alarms.
- **Recall** — of all the faces that actually are smiling, how many
  did we catch? High recall means few missed detections.
- **F1** — the harmonic mean of precision and recall. It punishes
  models that sacrifice one to inflate the other.
- **AUC** — ranking quality, independent of any specific threshold.
  The most robust single number for comparing classifiers.

Across all five metrics, **both ensemble methods match or beat the
single Decision Tree baseline**.

**[** You might notice the differences look small in absolute numbers.
But keep in mind: our test set has around 40,000 samples.
Even a 0.5% accuracy gain represents approximately 200 extra correct
predictions.
In a production system running at that scale, those gains matter. **]**"

---

## Slide 9 — Discussion & Conclusion (~1 minute)

*[Display final slide: summary bullet points]*

---

"To wrap up:

We compared three tree-based classifiers on a real-world binary classification
task — predicting smiling from 39 binary facial attributes in CelebA.

Our results confirm the well-established principle that **combining learners
produces more robust classifiers** than any single model:

- **Random Forest** reduced variance by averaging many decorrelated trees.
- **AdaBoost** reduced bias by sequentially correcting errors.
- **Both outperformed the single Decision Tree** across all five metrics.

One important limitation to acknowledge:
we used attribute labels only — not raw image pixels.
A convolutional neural network operating directly on the photographs
would push accuracy considerably higher, because it can exploit
subtle texture and shape cues that our 39 binary labels simply cannot capture.

**[** Other limitations include:
potential label noise from human annotators — which hits AdaBoost
particularly hard — and fairness concerns, since CelebA skews toward
lighter skin tones and certain beauty standards, which may limit
generalisation to diverse populations. **]**

That concludes our presentation.
We're happy to take any questions."

---

## Anticipated Q&A Responses

### "Why doesn't Random Forest do much better than a single tree here?"

"Our features are already highly processed binary semantic labels,
and there is one very strong signal — Mouth_Slightly_Open.
When one feature dominates, the benefit of Random Forest's feature
randomness is smaller.
With raw image pixels or continuous features, RF would pull much further
ahead because tree diversity matters more.
Even here, RF's AUC and F1 are measurably above the single tree."

---

### "Why doesn't AdaBoost do better than Random Forest?"

"AdaBoost shines when the dominant weakness is bias — when the model is
too simple to capture the signal.
Here, even a shallow tree captures the main signal well.
Also, AdaBoost with stumps can be sensitive to the label noise inherent
in human-labelled data at 202K scale, which limits its advantage."

---

### "How is AdaBoost different from Gradient Boosting (XGBoost, LightGBM)?"

"AdaBoost re-weights samples and uses an exponential loss function.
Gradient Boosting frames each round as fitting the *gradient* of any
differentiable loss function — log-loss, mean squared error, quantile loss.
That generalisation usually gives better results but introduces more
hyperparameters to tune.
For this project the simpler AdaBoost formulation was sufficient."

---

### "Why did you use a validation set instead of cross-validation?"

"With 202,000 samples, a single hold-out split gives a stable estimate
of generalisation error at a fraction of the compute cost.
On a smaller dataset — say, a few thousand samples — k-fold
cross-validation is strongly preferred because it uses all data for
both training and evaluation in rotation."

---

### "What does stratify=y do?"

"It ensures each of the three splits — train, validation, test —
contains roughly the same 48% positive rate as the full dataset.
Without stratification, a random split could produce a test set with,
say, 60% positives purely by chance, which would make metrics
non-comparable across experiments."

---

### "Could you improve results further?"

"Yes. Gradient Boosted Trees — XGBoost or LightGBM — typically outperform
plain AdaBoost on tabular data.
A grid search over learning rate, tree depth, and number of estimators
would likely push performance further.
And of course, moving to raw image features with a CNN would be a much
bigger jump."

---

*End of script.*
