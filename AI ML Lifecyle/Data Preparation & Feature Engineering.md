# AI/ML Lifecycle — Part 2

## Data Preparation & Feature Engineering

Data preparation is one of the most important stages of the ML lifecycle. Real-world data is rarely ready to be directly given to a machine learning algorithm.

```text
Raw Data
   ↓
Data Cleaning
   ↓
Data Transformation
   ↓
Feature Engineering
   ↓
Feature Selection
   ↓
Data Splitting
   ↓
Prepared Dataset
```

---

# 1. Data Preparation

**Data Preparation** is the process of converting raw collected data into a clean, consistent, and model-ready format.

Raw data may contain:

* Missing values
* Duplicate records
* Incorrect values
* Outliers
* Different units
* Categorical variables
* Different scales
* Irrelevant features
* Inconsistent formats

### Example

Raw data:

```text
age     salary       city       experience
25      40000        Chennai    2
NULL    55000        Chennai    4
30      60000        chennai    5
25      40000        Chennai    2
```

Problems:

* Missing `age`
* `Chennai` vs `chennai`
* Duplicate row
* Potential formatting inconsistencies

These issues should be addressed before model training.

---

# 2. Data Cleaning

**Data Cleaning** is the process of detecting and correcting problems in the dataset.

Common operations include:

1. Handling missing values
2. Removing duplicates
3. Correcting inconsistent values
4. Handling outliers
5. Fixing incorrect data types
6. Standardizing formats

---

## 2.1 Missing Values

A missing value occurs when an observation does not contain a value for a particular feature.

Example:

```text
Age
---
25
30
NaN
28
```

Common approaches:

### Remove Rows

Useful when only a small number of records are missing values.

```python
df.dropna()
```

### Fill with Mean

```text
Missing value → Average of available values
```

Useful for some numerical features when the distribution is reasonably suitable.

### Fill with Median

```text
Missing value → Median
```

Often more robust than the mean when the data contains outliers.

### Fill with Mode

Used commonly for categorical variables.

```text
Missing city → Most frequent city
```

### Model-Based Imputation

A separate model can be used to estimate missing values based on other features.

---

# 3. Duplicate Data

Duplicate records occur when the same observation appears multiple times.

Example:

```text
ID   Age   Salary
1    25    40000
1    25    40000
```

Duplicates can:

* Distort the dataset
* Give excessive importance to certain observations
* Affect model performance

Pandas:

```python
df.drop_duplicates()
```

---

# 4. Handling Outliers

An **outlier** is an observation that is unusually far from the typical values in a dataset.

Example:

```text
Salaries:
30,000
35,000
40,000
42,000
45,000
5,00,00,000
```

The extremely large value may be an outlier.

Common methods for identifying outliers:

* Box plots
* IQR method
* Z-score
* Statistical analysis
* Domain knowledge

### Important

**Do not automatically remove every outlier.**

An outlier may represent:

* Data-entry error
* Rare legitimate event
* Fraud
* Important business case

The decision should depend on the problem and domain.

---

# 5. Data Transformation

Data often needs to be transformed before being provided to a model.

Common transformations include:

* Scaling
* Normalization
* Standardization
* Encoding
* Log transformation
* Date/time transformation

---

# 6. Feature Engineering

**Feature Engineering** is the process of creating, transforming, or deriving useful features from existing data to help a machine learning model learn better patterns.

It can involve:

```text
Existing Features
       ↓
Transformation / Combination
       ↓
New Features
```

Feature engineering is particularly important in traditional ML because the quality of features can strongly affect model performance.

---

# 7. Creating New Features

New features can be derived from existing features.

### Example: Customer Data

Suppose we have:

```text
date_of_birth
```

Instead of directly using the date, we can create:

```text
age
```

### Example: Taxi Data

Suppose we have:

```text
pickup_datetime
dropoff_datetime
```

We can create:

```text
trip_duration
```

Similarly:

```text
pickup_datetime
        ↓
hour
day_of_week
month
is_weekend
```

These derived features may contain information that is easier for the model to use.

---

# 8. Interaction Features

An **interaction feature** represents the combined effect of two or more features.

Example:

```text
price × quantity = total_value
```

For a house-price model:

```text
area × number_of_rooms
```

can potentially represent an interaction between house size and room count.

Interaction features can help some models capture relationships between variables.

---

# 9. Feature Transformation

Sometimes the original representation of a feature is not ideal.

### Example

Income:

```text
10,000
20,000
50,000
1,00,000
10,00,000
```

The values may have a highly skewed distribution.

A logarithmic transformation can sometimes make the distribution easier for certain models to work with:

```text
income → log(income)
```

Feature transformations should be chosen based on the data and model.

---

# 10. Categorical Encoding

Machine learning algorithms generally require numerical representations of categorical variables.

Example:

```text
City
----
Chennai
Mumbai
Delhi
```

We can encode them using techniques such as **One-Hot Encoding**:

```text
City_Chennai    City_Mumbai    City_Delhi
      1              0              0
      0              1              0
      0              0              1
```

Other encoding techniques include:

* Ordinal encoding
* Target encoding
* Frequency encoding

The appropriate method depends on the nature of the categorical variable and the model.

---

# 11. Feature Scaling

Features can have very different numerical ranges.

Example:

```text
Age       → 18–80
Salary    → 20,000–20,00,000
Distance  → 1–100
```

Some algorithms can be affected by these differences in scale.

Two important techniques are:

## Standardization

Transforms values so that the feature approximately has:

```text
Mean = 0
Standard Deviation = 1
```

Formula:

```text
z = (x - μ) / σ
```

Commonly implemented using:

```python
from sklearn.preprocessing import StandardScaler
```

---

## Normalization

Often scales values to a fixed range, commonly:

```text
0 to 1
```

One common formula:

```text
x' = (x - min(x)) / (max(x) - min(x))
```

Commonly implemented using:

```python
from sklearn.preprocessing import MinMaxScaler
```

---

# 12. Feature Selection

**Feature Selection** means selecting the most useful features for model training and removing irrelevant or redundant ones.

Example:

```text
100 available features
        ↓
Select 25 useful features
        ↓
Train model
```

### Why Feature Selection?

It can:

* Reduce model complexity
* Reduce training time
* Reduce noise
* Improve generalization
* Make models easier to interpret
* Reduce the risk of overfitting

---

# 13. Feature Selection Methods

## Filter Methods

Features are selected using statistical properties of the data.

Examples:

* Correlation
* Chi-square test
* ANOVA
* Mutual information

These methods are generally independent of a specific ML model.

---

## Wrapper Methods

Features are evaluated by repeatedly training a model using different feature subsets.

Examples:

* Recursive Feature Elimination (RFE)
* Forward Selection
* Backward Elimination

They can be computationally expensive because models may need to be trained many times.

---

## Embedded Methods

Feature selection happens during model training.

Examples:

* Lasso Regression
* Decision-tree-based feature importance
* Regularization-based methods

---

# 14. Feature Engineering vs Feature Selection

These are **not the same thing**.

| Feature Engineering                        | Feature Selection                       |
| ------------------------------------------ | --------------------------------------- |
| Creates or transforms features             | Chooses useful existing features        |
| Can increase number of features            | Usually reduces number of features      |
| Example: `trip_duration`                   | Example: remove `customer_id`           |
| Focuses on creating useful representations | Focuses on identifying useful variables |

### Simple Memory Trick

> **Engineering = Create/Transform**
> **Selection = Choose/Remove**

---

# 15. Data Splitting

After preparing the data, it needs to be divided into different subsets.

The three common subsets are:

```text
Dataset
   │
   ├── Training Set
   │
   ├── Validation Set
   │
   └── Test Set
```

---

# 16. Training Set

The **training set** is used to train the model.

The model learns relationships between:

```text
Features → Target
```

Example:

```text
Distance + Time + Location
             ↓
       ML Algorithm
             ↓
        Fare Amount
```

The model parameters are learned from the training data.

---

# 17. Validation Set

The **validation set** is used during model development.

It helps with:

* Hyperparameter tuning
* Model selection
* Comparing different approaches
* Detecting overfitting during development

Example:

```text
Random Forest
Decision Tree
XGBoost
        ↓
Validation Performance
        ↓
Select / Tune Model
```

---

# 18. Test Set

The **test set** is used for the final evaluation of the selected model.

It should represent **unseen data**.

The test set should ideally be kept separate from the model-development process.

### Important Rule

> **Do not repeatedly tune your model based on test-set performance.**

Otherwise, the test set effectively becomes part of the development process and no longer provides an unbiased estimate of generalization.

---

# 19. Typical Data Split

A common split might be:

```text
80% → Training
10% → Validation
10% → Testing
```

Another common approach:

```text
70% → Training
15% → Validation
15% → Testing
```

There is **no universal split ratio**.

The appropriate split depends on:

* Dataset size
* Problem type
* Amount of available data
* Validation strategy
* Whether cross-validation is used

---

# 20. Data Leakage During Preparation ⚠️

One of the most important concepts in preprocessing is:

> **Never allow information from validation/test data to influence preprocessing learned from training data.**

### Wrong Approach

```text
Entire Dataset
      ↓
Fit Scaler
      ↓
Split Dataset
```

The scaler has seen information from the future validation/test data.

### Better Approach

```text
Raw Dataset
     ↓
Split
     ↓
Training Data ──→ Fit preprocessing
                      ↓
Validation/Test ──→ Transform using
                     training-fitted preprocessing
```

For example:

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler learns its parameters **only from training data**.

---

# 21. Preprocessing Pipelines

A **preprocessing pipeline** automates a sequence of preprocessing operations.

Instead of manually performing:

```text
Missing Value Handling
        ↓
Encoding
        ↓
Scaling
        ↓
Feature Selection
        ↓
Model
```

we can create a reproducible pipeline.

### Scikit-learn

```python
from sklearn.pipeline import Pipeline
```

Conceptually:

```text
Raw Data
   ↓
Imputation
   ↓
Encoding
   ↓
Scaling
   ↓
Model
   ↓
Prediction
```

---

# 22. Why Pipelines Are Important

Pipelines provide:

### Consistency

The same preprocessing logic is applied during training and prediction.

### Reproducibility

The workflow can be repeated consistently.

### Reduced Data Leakage

Preprocessing steps can be fitted correctly within the training workflow.

### Easier Deployment

The preprocessing and model can be treated as one workflow.

### Maintainability

Changing or adding preprocessing steps becomes easier.

---

# 23. Important Tools

The slides mention three major Python libraries for data preparation.

## Pandas

Used for:

* Data manipulation
* Data cleaning
* Data analysis
* DataFrame operations
* Handling missing values

```python
import pandas as pd
```

---

## NumPy

Used for:

* Numerical computation
* Arrays
* Mathematical operations
* Efficient numerical processing

```python
import numpy as np
```

---

## Scikit-learn

Provides tools for:

* Preprocessing
* Feature selection
* Data splitting
* Machine learning algorithms
* Model evaluation
* Pipelines

```python
import sklearn
```

---

# 24. Typical Data Preparation Workflow

A practical workflow looks like:

```text
Raw Dataset
     ↓
Understand the Data
     ↓
Clean Data
     ↓
Handle Missing Values
     ↓
Handle Duplicates
     ↓
Handle Outliers
     ↓
Encode Categorical Features
     ↓
Feature Engineering
     ↓
Feature Selection
     ↓
Split Data
     ↓
Fit Preprocessing on Training Data
     ↓
Transform Validation/Test Data
     ↓
Build Pipeline
     ↓
Model Training
```

---

# 25. NYC Taxi Example 🚕

For the NYC taxi project, raw columns might include:

```text
pickup_datetime
dropoff_datetime
passenger_count
pickup_longitude
pickup_latitude
dropoff_longitude
dropoff_latitude
trip_distance
fare_amount
```

Feature engineering could create:

```text
trip_duration
pickup_hour
pickup_day
pickup_month
pickup_day_of_week
is_weekend
```

Then:

```text
Raw Taxi Data
      ↓
Clean invalid records
      ↓
Handle missing values
      ↓
Create useful features
      ↓
Select relevant features
      ↓
Split into train/test
      ↓
Train regression model
```

This is exactly how the concepts from this section connect to an actual ML project.

---

# 26. Common Mistakes ⚠️

### Mistake 1: Scaling before splitting

Can cause data leakage.

### Mistake 2: Using test data for hyperparameter tuning

Makes the final evaluation unreliable.

### Mistake 3: Creating features using future information

This is a form of **data leakage**.

### Mistake 4: Removing every outlier

Some outliers are legitimate and important.

### Mistake 5: Keeping every feature

Irrelevant features can add noise and complexity.

### Mistake 6: Treating preprocessing as a one-time manual task

Production systems need **reproducible preprocessing pipelines**.

---

# 27. Key Takeaways

```text
Data Preparation
    ↓
Clean
    ↓
Transform
    ↓
Engineer Features
    ↓
Select Features
    ↓
Split Data
    ↓
Build Preprocessing Pipeline
```

Remember:

* **Data preparation** converts raw data into model-ready data.
* **Data cleaning** handles missing values, duplicates, inconsistencies, and outliers.
* **Feature engineering** creates or transforms useful features.
* **Feature selection** chooses relevant features.
* **Encoding** converts categorical information into numerical representations.
* **Scaling** puts numerical features on comparable scales when appropriate.
* **Training data** is used to learn model parameters.
* **Validation data** is used during model development and tuning.
* **Test data** is reserved for final evaluation.
* Preprocessing should be **fitted on training data**, then applied to validation/test data.
* **Pipelines** improve consistency, reproducibility, maintainability, and help prevent leakage.
* Pandas, NumPy, and scikit-learn are important tools for this stage.

---

# 28. Interview Quick Revision 🎯

**Q: What is feature engineering?**
Feature engineering is the process of creating, transforming, or deriving useful features from raw data to improve the representation of information for a machine learning model.

**Q: What is feature selection?**
Feature selection is the process of selecting relevant features and removing irrelevant or redundant features.

**Q: Why do we split data into train, validation, and test sets?**
Training data is used to learn the model, validation data is used for model selection and hyperparameter tuning, and test data is reserved for final evaluation on unseen data.

**Q: Why shouldn't we fit a scaler on the entire dataset?**
Because the scaler would learn information from validation/test data, causing data leakage.

**Q: What is a preprocessing pipeline?**
A preprocessing pipeline chains multiple preprocessing steps into a reproducible workflow and can combine them with the model.

**Q: What is the difference between normalization and standardization?**
Normalization commonly scales values to a fixed range such as 0–1, while standardization transforms values based on the mean and standard deviation, typically producing mean 0 and standard deviation 1.

**Q: Should every outlier be removed?**
No. An outlier may be a legitimate observation. Its treatment should depend on the data, domain, and reason for the outlier.

**Q: What is the difference between training, validation, and test data?**

```text
Training   → Learn
Validation → Tune / Select
Testing    → Final Evaluation
```

### ⭐ One-line memory rule

> **Prepare the data without allowing future information to leak into the training process.**
