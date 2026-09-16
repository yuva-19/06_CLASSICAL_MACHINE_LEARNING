# AI/ML Lifecycle — Part 4

## Model Evaluation

> **Goal:** Determine whether a trained ML model performs well enough on unseen data and whether it is suitable for the actual business problem.

---

# 1. What is Model Evaluation?

**Model Evaluation** is the process of measuring how well a trained machine learning model performs using appropriate evaluation metrics.

The evaluation process helps us answer:

* Is the model making accurate predictions?
* Does it generalize to unseen data?
* Is it overfitting?
* Is it suitable for the business requirement?
* Which model performs better?
* Can the model be deployed to production?

### Important

The evaluation metric depends on the **type of ML problem**.

| Problem Type    | Common Evaluation Metrics                                       |
| --------------- | --------------------------------------------------------------- |
| Classification  | Accuracy, Precision, Recall, F1-Score, ROC-AUC, Log Loss        |
| Regression      | MAE, MSE, RMSE, R², Adjusted R², MAPE                           |
| Clustering      | Rand Index, Mutual Information                                  |
| NLP             | BLEU Score                                                      |
| Computer Vision | Task-specific metrics such as classification accuracy, IoU, mAP |

---

# 2. Classification Evaluation

Classification predicts a **category/class**.

Examples:

```text
Spam / Not Spam
Fraud / Not Fraud
Disease / No Disease
Cat / Dog
```

The most important foundation for classification metrics is the **Confusion Matrix**.

---

# 3. Confusion Matrix

A confusion matrix compares:

* **Actual values**
* **Predicted values**

For binary classification:

|                     | Predicted Positive | Predicted Negative |
| ------------------- | -----------------: | -----------------: |
| **Actual Positive** |                 TP |                 FN |
| **Actual Negative** |                 FP |                 TN |

### 3.1 True Positive — TP

The model predicted **Positive**, and the actual value was **Positive**.

Example:

```text
Actual: Fraud
Predicted: Fraud
→ TP
```

### 3.2 True Negative — TN

The model predicted **Negative**, and the actual value was **Negative**.

```text
Actual: Not Fraud
Predicted: Not Fraud
→ TN
```

### 3.3 False Positive — FP

The model predicted **Positive**, but the actual value was **Negative**.

```text
Actual: Not Fraud
Predicted: Fraud
→ FP
```

Also called a **Type I Error**.

### 3.4 False Negative — FN

The model predicted **Negative**, but the actual value was **Positive**.

```text
Actual: Fraud
Predicted: Not Fraud
→ FN
```

Also called a **Type II Error**.

---

# 4. Accuracy

**Accuracy** measures the proportion of all predictions that were correct.

### Formula

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Example

Suppose:

```text
TP = 80
TN = 90
FP = 10
FN = 20
```

Then:

```text
Accuracy = (80 + 90) / (80 + 90 + 10 + 20)
         = 170 / 200
         = 0.85
         = 85%
```

### When is Accuracy Useful?

Accuracy works well when:

* Classes are reasonably balanced.
* False positives and false negatives have similar importance.

### Problem with Accuracy

Accuracy can be misleading for **imbalanced datasets**.

Example:

```text
990 non-fraud transactions
10 fraud transactions
```

A model that predicts **every transaction as non-fraud** gets:

```text
Accuracy = 990 / 1000 = 99%
```

But the model detects **zero fraud cases**.

Therefore, accuracy alone is not enough for many real-world problems.

---

# 5. Precision

**Precision** answers:

> "Of all the instances the model predicted as positive, how many were actually positive?"

### Formula

```text
Precision = TP / (TP + FP)
```

### Example

If:

```text
TP = 80
FP = 20
```

Then:

```text
Precision = 80 / (80 + 20)
          = 0.80
          = 80%
```

### High Precision Means

The model produces fewer **false positives**.

### Example

For spam detection:

```text
Predicted spam = 100 emails
Actually spam = 90
```

High precision means most emails classified as spam really are spam.

### When is Precision Important?

Use precision when **false positives are costly**.

Examples:

* Spam filtering
* Fraud investigation
* Content moderation
* Medical alerts where unnecessary alarms are costly

---

# 6. Recall

**Recall** answers:

> "Of all the actual positive instances, how many did the model successfully identify?"

### Formula

```text
Recall = TP / (TP + FN)
```

### Example

If:

```text
TP = 80
FN = 20
```

Then:

```text
Recall = 80 / (80 + 20)
       = 0.80
       = 80%
```

### High Recall Means

The model produces fewer **false negatives**.

### When is Recall Important?

Recall is especially important when missing a positive case is dangerous or expensive.

Examples:

* Disease detection
* Fraud detection
* Security threat detection
* Defect detection

For example, in disease detection:

```text
Actual disease → Model says healthy
```

This is a **False Negative**, which could be much more serious than a false positive.

---

# 7. Precision vs Recall

This is one of the **most important concepts for ML interviews**.

| Metric    | Main Question                                 | Focus     |
| --------- | --------------------------------------------- | --------- |
| Precision | Of predicted positives, how many are correct? | Reduce FP |
| Recall    | Of actual positives, how many did we find?    | Reduce FN |

### Easy Memory Trick

```text
Precision → "Predicted Positive → Correct?"

Recall → "Actual Positive → Found?"
```

---

# 8. F1-Score

**F1-Score** is the harmonic mean of **Precision and Recall**.

### Formula

```text
F1 = 2 × (Precision × Recall)
        -----------------------
        (Precision + Recall)
```

### Why Harmonic Mean?

The harmonic mean gives a low score when either precision or recall is low.

For example:

```text
Precision = 0.90
Recall    = 0.10
```

The F1-score will remain relatively low.

This prevents a model from appearing good simply because one metric is high.

### When to Use F1-Score?

F1 is useful when:

* Dataset is imbalanced.
* Both precision and recall are important.
* You want a single metric balancing FP and FN.

---

# 9. ROC-AUC

## ROC

**ROC = Receiver Operating Characteristic**

The ROC curve evaluates a binary classifier across different classification thresholds.

It plots:

```text
True Positive Rate
        vs
False Positive Rate
```

Where:

```text
True Positive Rate = Recall
```

and

```text
False Positive Rate = FP / (FP + TN)
```

---

## AUC

**AUC = Area Under the Curve**

AUC represents how well the model separates positive and negative classes.

Typical interpretation:

|     AUC | Interpretation     |
| ------: | ------------------ |
|     1.0 | Perfect classifier |
| 0.9–1.0 | Excellent          |
| 0.8–0.9 | Good               |
| 0.7–0.8 | Fair               |
|     0.5 | Random performance |
|   < 0.5 | Worse than random  |

### Important

ROC-AUC is particularly useful when you want to evaluate the model's **ranking/separation ability across thresholds**, rather than relying on one fixed threshold.

---

# 10. Log Loss

**Log Loss** measures the quality of predicted probabilities.

It penalizes predictions that are:

* Incorrect
* Overconfident

Example:

```text
Actual class = 1

Prediction A = 0.90
Prediction B = 0.60
Prediction C = 0.01
```

Prediction C receives a very large penalty because the model was **extremely confident in the wrong answer**.

### Important

For Log Loss:

```text
Lower = Better
```

Unlike:

```text
Accuracy, Precision, Recall, F1
```

where generally:

```text
Higher = Better
```

---

# 11. Regression Evaluation

Regression predicts a **continuous numerical value**.

Examples:

```text
House price
Taxi fare
Temperature
Sales revenue
Demand
Salary
```

Common regression metrics include:

```text
MAE
MSE
RMSE
R²
Adjusted R²
MAPE
```

---

# 12. MAE — Mean Absolute Error

MAE measures the average absolute difference between actual and predicted values.

### Formula

```text
MAE = (1/n) Σ |yi - ŷi|
```

Where:

```text
yi  = actual value
ŷi  = predicted value
n   = number of observations
```

### Example

Actual:

```text
[100, 200, 300]
```

Predicted:

```text
[110, 180, 320]
```

Errors:

```text
10, 20, 20
```

Therefore:

```text
MAE = (10 + 20 + 20) / 3
    = 16.67
```

### Interpretation

The model's predictions are off by approximately **16.67 units on average**.

### Important

```text
Lower MAE = Better
```

MAE is relatively easy to interpret because it uses the **same unit as the target variable**.

---

# 13. MSE — Mean Squared Error

MSE calculates the average of squared errors.

### Formula

```text
MSE = (1/n) Σ (yi - ŷi)²
```

Because errors are squared, **large errors receive much more penalty**.

### Important

```text
Lower MSE = Better
```

### Example

Errors:

```text
2, 3, 10
```

Squared errors:

```text
4, 9, 100
```

The error of `10` has a much greater impact.

### MAE vs MSE

```text
MAE → Treats errors more linearly
MSE → Penalizes large errors heavily
```

---

# 14. RMSE — Root Mean Squared Error

RMSE is the square root of MSE.

### Formula

```text
RMSE = √[(1/n) Σ (yi - ŷi)²]
```

### Important

```text
Lower RMSE = Better
```

RMSE is useful because the result is expressed in the **same unit as the target variable**.

### MAE vs RMSE

| Metric | Behavior                                |
| ------ | --------------------------------------- |
| MAE    | Less sensitive to outliers              |
| RMSE   | More sensitive to large errors/outliers |

---

# 15. R² — R-Squared

**R²** measures how much of the variance in the target variable is explained by the model.

### Formula

```text
R² = 1 - SSR / SST
```

Where:

```text
SSR = Σ(yi - ŷi)²
SST = Σ(yi - ȳ)²
```

### General Interpretation

```text
R² = 1
```

Means perfect prediction.

```text
R² = 0
```

Means the model performs roughly like predicting the mean of the target, in the standard interpretation.

R² can also be **negative** on unseen data when the model performs worse than that baseline.

### Example

```text
R² = 0.80
```

This generally means the model explains about **80% of the variance** in the target for that evaluation dataset.

### Important

Higher R² is generally better, but **R² alone should not be used to judge a regression model**.

---

# 16. Adjusted R²

Adjusted R² is a modified version of R² that accounts for the **number of predictors/features** in the model.

Regular R² can increase when additional features are added, even if those features provide little useful information.

Adjusted R² introduces a penalty for unnecessary predictors.

### Important

Use Adjusted R² when comparing regression models with **different numbers of features**.

---

# 17. MAPE — Mean Absolute Percentage Error

MAPE measures the average absolute error as a percentage of the actual value.

### Formula

```text
MAPE = (100/n) Σ |(yi - ŷi) / yi|
```

### Example

If:

```text
Actual = 100
Predicted = 90
```

Then:

```text
Percentage error = |100 - 90| / 100 × 100
                 = 10%
```

### Important

```text
Lower MAPE = Better
```

### Limitation

MAPE can behave poorly when actual values are **zero or very close to zero**.

---

# 18. Regression Metrics — Quick Comparison

| Metric      | Measures                                | Outlier Sensitivity          | Better |
| ----------- | --------------------------------------- | ---------------------------- | ------ |
| MAE         | Average absolute error                  | Lower                        | Lower  |
| MSE         | Average squared error                   | High                         | Lower  |
| RMSE        | Square root of MSE                      | High                         | Lower  |
| R²          | Explained variance                      | —                            | Higher |
| Adjusted R² | Explained variance with feature penalty | —                            | Higher |
| MAPE        | Percentage error                        | Can be problematic near zero | Lower  |

---

# 19. Unsupervised Model Evaluation

Unsupervised learning works with **unlabeled data**, so evaluation is different from supervised learning.

The provided material mentions:

### Rand Index

Measures similarity between:

* Clustering assignments produced by the model
* A reference/ground-truth partition, when such reference labels are available

Higher values generally indicate greater agreement.

### Mutual Information

Measures the amount of information shared between two variables or partitions.

In clustering evaluation, it can measure how much information the predicted clusters share with known/reference labels.

---

# 20. Other Evaluation Metrics

Some ML applications require specialized metrics.

### Computer Vision

Examples:

```text
IoU
mAP
Classification Accuracy
Precision
Recall
```

### NLP

The slides mention:

```text
BLEU Score
```

BLEU is commonly used to evaluate machine-generated text against reference text, especially in tasks such as machine translation.

---

# 21. Choosing the Right Evaluation Metric

Do **not** blindly choose accuracy.

Choose the metric according to the **business problem**.

### Example 1 — Disease Detection

Priority:

```text
Recall
```

Reason:

Missing an actual disease case (**FN**) can be very costly.

---

### Example 2 — Spam Detection

Precision can be very important.

Reason:

You don't want legitimate emails incorrectly classified as spam.

---

### Example 3 — Fraud Detection

Often:

```text
Precision + Recall
```

or:

```text
F1 / PR-AUC / other business-specific metrics
```

may be more informative than accuracy because fraud datasets are usually highly imbalanced.

---

### Example 4 — House Price Prediction

Use:

```text
MAE
RMSE
R²
```

depending on the business objective.

---

# 22. Model Evaluation in the ML Lifecycle

Evaluation is **not the end of model development**.

A typical flow is:

```text
Train Model
     ↓
Evaluate on Validation Data
     ↓
Analyze Errors
     ↓
Improve Features / Hyperparameters / Model
     ↓
Evaluate Again
     ↓
Final Test Evaluation
     ↓
Deploy
```

The process can be iterative.

---

# 23. Critical Concept: Validation vs Test Set

During model development:

```text
Training Set
    ↓
Model learns
```

```text
Validation Set
    ↓
Model selection + hyperparameter tuning
```

```text
Test Set
    ↓
Final unbiased performance estimate
```

### Golden Rule

**Do not repeatedly tune your model using the test set.**

Otherwise, information from the test set can indirectly influence model decisions, making the final evaluation less trustworthy.

---

# 24. Interview Quick Revision

### Classification

```text
Accuracy  → Overall correctness
Precision → Correctness of positive predictions
Recall    → Ability to find actual positives
F1        → Balance between precision and recall
ROC-AUC   → Class separation across thresholds
Log Loss  → Quality of predicted probabilities
```

### Regression

```text
MAE          → Average absolute error
MSE          → Squared error; penalizes large errors
RMSE         → Error in target's units; sensitive to large errors
R²           → Explained variance
Adjusted R²  → R² adjusted for number of features
MAPE         → Percentage error
```

### Most Important Formulas

```text
Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

Accuracy = (TP + TN) / (TP + TN + FP + FN)

F1 = 2 × Precision × Recall
         ---------------------
         Precision + Recall

MAE = (1/n) Σ |yi - ŷi|

MSE = (1/n) Σ (yi - ŷi)²

RMSE = √MSE

R² = 1 - SSR/SST
```

---

