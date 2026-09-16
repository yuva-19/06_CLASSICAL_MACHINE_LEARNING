# AI/ML Lifecycle — Part 3

## Model Building: Algorithm Selection, Training & Cross-Validation

Once the data has been cleaned, transformed, engineered, and split, the next stage is **Model Building**.

The goal is to find and train a model that can learn useful patterns from the prepared data and generalize well to unseen data.

---

# 1. Model Building

**Model Building** is the process of selecting an appropriate machine learning algorithm, training it on data, tuning it, and producing a model that can make predictions.

A simplified workflow is:

```text
Prepared Data
     ↓
Choose ML Problem
     ↓
Select Algorithm
     ↓
Train Model
     ↓
Validate Model
     ↓
Tune Hyperparameters
     ↓
Compare Models
     ↓
Select Best Model
```

Model building is usually **iterative**.

You rarely choose one algorithm, train it once, and immediately deploy it.

---

# 2. Choosing the Right Algorithm

Algorithm selection depends on several factors:

* Type of ML problem
* Size of dataset
* Number of features
* Data type
* Relationship between features and target
* Interpretability requirements
* Training time
* Prediction latency
* Available computational resources
* Required performance

### Example

If the goal is:

```text
Predict house price
```

This is a **regression** problem.

Possible algorithms:

```text
Linear Regression
Decision Tree Regression
Random Forest Regression
Gradient Boosting
Neural Network Regression
```

If the goal is:

```text
Predict whether an email is spam
```

This is a **classification** problem.

Possible algorithms:

```text
Logistic Regression
Naive Bayes
Decision Tree
Random Forest
SVM
Neural Network
```

---

# 3. Types of Machine Learning

The slide identifies four major categories:

```text
Machine Learning
│
├── Supervised Learning
├── Unsupervised Learning
├── Semi-Supervised Learning
└── Reinforcement Learning
```

The most important categories for traditional ML engineering are **supervised** and **unsupervised learning**.

---

# 4. Supervised Learning

**Supervised learning** is a type of machine learning where the model learns from **labeled data**.

The training data contains:

```text
Input Features + Known Target
```

The model learns a mapping:

```text
X → y
```

where:

* `X` = input features
* `y` = target/label

---

## Example

Suppose we have taxi data:

```text
Distance    Passengers    Time       Fare
------------------------------------------------
5 km        2             10 AM      ₹150
10 km       3             6 PM       ₹300
3 km        1             8 AM       ₹100
```

The model learns:

```text
Distance + Passengers + Time
             ↓
         ML Model
             ↓
        Predicted Fare
```

Because the correct fare is already available during training, this is supervised learning.

---

# 5. Supervised Learning Tasks

The two major supervised learning tasks are:

```text
Supervised Learning
       │
       ├── Classification
       │
       └── Regression
```

---

# 6. Classification

**Classification** predicts a discrete category or class.

Examples:

```text
Spam / Not Spam
Fraud / Not Fraud
Cat / Dog
Pass / Fail
Disease / No Disease
```

The output belongs to one or more predefined classes.

---

## Binary Classification

There are two classes.

Example:

```text
Fraud
Not Fraud
```

Output:

```text
0 or 1
```

---

## Multiclass Classification

There are more than two classes.

Example:

```text
Cat
Dog
Horse
```

The model chooses one class from multiple possibilities.

---

# 7. Popular Classification Algorithms

### Logistic Regression

Despite its name, Logistic Regression is primarily used for **classification**.

It estimates the probability of belonging to a class.

---

### Decision Tree

A tree-like structure that makes decisions using feature-based conditions.

Example:

```text
        Income > ₹50k?
          /      \
        Yes       No
        ↓          ↓
   Low Risk     High Risk
```

Advantages:

* Easy to understand
* Handles nonlinear relationships
* Little preprocessing may be required

---

### Random Forest

An ensemble of multiple decision trees.

```text
Data
 ↓
Tree 1 ──┐
Tree 2 ──┤
Tree 3 ──┤
Tree 4 ──┤ → Combined Prediction
Tree 5 ──┘
```

It generally provides better robustness than a single decision tree.

---

### Support Vector Machine (SVM)

SVM attempts to find a decision boundary that separates classes effectively.

It can be particularly useful for some high-dimensional datasets.

---

### K-Nearest Neighbors (KNN)

KNN makes predictions based on the closest observations in the feature space.

Conceptually:

```text
New Point
    ↓
Find K nearest points
    ↓
Look at their classes
    ↓
Choose majority class
```

---

### Naive Bayes

A probabilistic classification algorithm based on Bayes' theorem with a simplifying conditional-independence assumption.

It is commonly used for tasks such as:

* Text classification
* Spam detection
* Document classification

---

# 8. Regression

**Regression** predicts a continuous numerical value.

Examples:

```text
House Price
Taxi Fare
Temperature
Revenue
Demand
Salary
```

Example:

```text
Distance = 10 km
Passengers = 2
Time = 7 PM

        ↓

Regression Model

        ↓

Predicted Fare = ₹320
```

---

# 9. Popular Regression Algorithms

### Linear Regression

Models the relationship between input variables and a continuous target using a linear function.

Simple form:

```text
y = β₀ + β₁x
```

For multiple features:

```text
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

---

### Decision Tree Regression

Uses decision rules to divide the data into regions and predict numerical values.

---

### Random Forest Regression

Combines predictions from multiple decision trees.

---

### Support Vector Regression (SVR)

An SVM-based approach adapted for regression.

---

### Lasso Regression

Linear regression with **L1 regularization**.

It can shrink some feature coefficients to zero, making it useful for feature selection.

---

### Ridge Regression

Linear regression with **L2 regularization**.

It helps control model complexity and can reduce overfitting.

---

# 10. Unsupervised Learning

**Unsupervised learning** works with **unlabeled data**.

There is no predefined target variable.

```text
Input Data
    ↓
ML Algorithm
    ↓
Discover Hidden Patterns
```

The algorithm tries to discover structure within the data.

---

# 11. Clustering

**Clustering** groups similar observations together.

Example:

Suppose an e-commerce company has customer information:

```text
Customer
├── Spending
├── Purchase Frequency
└── Product Preferences
```

A clustering algorithm may discover:

```text
Cluster 1 → High-value customers
Cluster 2 → Occasional customers
Cluster 3 → Budget customers
```

The groups weren't provided beforehand; the algorithm discovered them.

---

# 12. Popular Clustering Algorithms

### K-Means

Divides observations into `K` clusters.

Conceptually:

```text
Data
 ↓
Choose K
 ↓
Assign points to clusters
 ↓
Update cluster centers
 ↓
Repeat
 ↓
Final clusters
```

---

### DBSCAN

Groups points based on density.

Useful when:

* Clusters have irregular shapes
* Noise/outliers need to be identified

Unlike K-Means, DBSCAN does not require specifying the number of clusters in advance in the same way.

---

### Hierarchical Clustering

Builds a hierarchy of clusters.

It can be represented using a **dendrogram**.

---

### Mean Shift

Identifies regions of high data density and forms clusters around those regions.

---

### Gaussian Mixture Models

Represents data as a mixture of multiple Gaussian distributions.

Unlike hard clustering approaches, GMM can provide **probabilistic membership**.

---

# 13. Reinforcement Learning

**Reinforcement Learning (RL)** is a learning paradigm in which an **agent interacts with an environment** and learns through rewards and penalties.

Basic structure:

```text
        Action
Agent ───────────→ Environment
  ↑                    │
  │                    │
  └── State + Reward ──┘
```

The agent attempts to learn a policy that maximizes cumulative reward.

---

## Example

A game-playing agent:

```text
Agent
 ↓
Makes Move
 ↓
Game Environment
 ↓
Reward / Penalty
 ↓
Learns Better Strategy
```

Common RL concepts include:

* Agent
* Environment
* State
* Action
* Reward
* Policy
* Value function

---

# 14. Semi-Supervised Learning

**Semi-supervised learning** uses a combination of:

```text
Small amount of labeled data
+
Large amount of unlabeled data
```

Example:

```text
10,000 images

500 labeled
9,500 unlabeled
```

Instead of ignoring the unlabeled data, semi-supervised methods can use both sources.

This can be useful when labeling data is expensive.

---

# 15. Model Training

Once an algorithm has been selected, we train the model.

**Training** is the process through which the model learns parameters from training data.

Conceptually:

```text
Training Data
     ↓
Algorithm
     ↓
Model
     ↓
Predictions
     ↓
Calculate Error
     ↓
Update Parameters
     ↓
Repeat
```

The exact learning procedure depends on the algorithm.

---

# 16. Parameters vs Hyperparameters

This distinction is **extremely important** for ML interviews.

## Parameters

Parameters are learned by the model from training data.

Example in Linear Regression:

```text
y = β₀ + β₁x
```

`β₀` and `β₁` are model parameters.

The algorithm learns them during training.

---

## Hyperparameters

Hyperparameters are configuration values chosen **before or during model training**, rather than directly learned as model parameters.

Examples:

```text
Decision Tree:
max_depth

Random Forest:
n_estimators

KNN:
n_neighbors

Neural Network:
learning_rate
batch_size
number_of_layers
```

Hyperparameters are commonly tuned using validation data or cross-validation.

### Memory Trick

> **Parameters = Model learns them**
> **Hyperparameters = We configure/tune them**

---

# 17. Cross-Validation

The slide introduces **Cross-Validation** as a technique for assessing model performance by partitioning data into multiple folds.

Cross-validation is particularly useful when the dataset is not large enough to rely on a single train/validation split.

---

# 18. K-Fold Cross-Validation

In **K-Fold Cross-Validation**, the dataset is divided into `K` approximately equal folds.

Example with `K = 5`:

```text
Dataset

┌────┬────┬────┬────┬────┐
│ F1 │ F2 │ F3 │ F4 │ F5 │
└────┴────┴────┴────┴────┘
```

The model is trained and validated `K` times.

### Round 1

```text
Train: F2 F3 F4 F5
Test : F1
```

### Round 2

```text
Train: F1 F3 F4 F5
Test : F2
```

### Round 3

```text
Train: F1 F2 F4 F5
Test : F3
```

### Round 4

```text
Train: F1 F2 F3 F5
Test : F4
```

### Round 5

```text
Train: F1 F2 F3 F4
Test : F5
```

The final performance is generally summarized using the average of the scores across folds.

---

# 19. Why Cross-Validation?

A single train/validation split can sometimes produce a misleading result depending on which observations happen to be placed in the validation set.

Cross-validation gives the model multiple opportunities to train and validate on different portions of the data.

Benefits:

* Better estimate of model performance
* More efficient use of limited data
* Helps compare models
* Helps tune hyperparameters
* Reduces dependence on one particular split

---

# 20. Leave-One-Out Cross-Validation

**Leave-One-Out Cross-Validation (LOOCV)** is an extreme form of K-Fold Cross-Validation where:

```text
K = Number of observations
```

For `N` observations:

```text
Train → N - 1 observations
Validate → 1 observation
```

This process is repeated `N` times.

Example:

```text
Dataset = 100 observations

Training = 99
Validation = 1

Repeat 100 times
```

### Advantage

Uses almost all available data for training in each iteration.

### Disadvantage

Can be computationally expensive, especially for large datasets.

---

# 21. Overfitting

**Overfitting** occurs when a model learns the training data too closely, including noise or accidental patterns.

Result:

```text
Training Performance → Very Good
Unseen Data Performance → Poor
```

Example:

```text
Training Accuracy = 99%
Validation Accuracy = 72%
```

This is a strong indication of overfitting.

---

# 22. Underfitting

**Underfitting** occurs when the model is too simple to capture the underlying patterns in the data.

Result:

```text
Training Performance → Poor
Validation Performance → Poor
```

Example:

```text
Training Accuracy = 68%
Validation Accuracy = 65%
```

The model isn't learning enough.

---

# 23. Overfitting vs Underfitting

|                        | Underfitting         | Good Fit         | Overfitting  |
| ---------------------- | -------------------- | ---------------- | ------------ |
| Model complexity       | Too low              | Appropriate      | Too high     |
| Training performance   | Poor                 | Good             | Excellent    |
| Validation performance | Poor                 | Good             | Poor         |
| Main issue             | Doesn't learn enough | Generalizes well | Learns noise |

### Memory Trick

```text
Underfitting → Model learns too little
Good fit     → Model learns useful patterns
Overfitting  → Model learns too much noise
```

---

# 24. How to Reduce Overfitting

Common techniques include:

* Collect more high-quality training data
* Feature selection
* Regularization
* Cross-validation
* Early stopping
* Reducing model complexity
* Pruning decision trees
* Dropout for neural networks
* Data augmentation for some applications

The appropriate method depends on the model and problem.

---

# 25. How to Address Underfitting

Possible approaches:

* Use a more expressive model
* Add useful features
* Reduce excessive regularization
* Train for longer where appropriate
* Improve feature engineering

---

# 26. Training Data Size

The slide also emphasizes the importance of training-data size.

Generally:

> More **relevant, representative, high-quality** data can help a model generalize better.

However, more data also means:

* More storage
* More preprocessing
* Longer training
* More computational resources
* Potentially higher infrastructure costs

Therefore, data quantity must be balanced with:

```text
Quality
+
Relevance
+
Representativeness
+
Computational Cost
```

---

# 27. Bias-Variance Perspective

A useful way to understand underfitting and overfitting is through **bias and variance**.

### High Bias

The model makes overly simplistic assumptions.

Usually associated with:

```text
Underfitting
```

### High Variance

The model is highly sensitive to the particular training data.

Usually associated with:

```text
Overfitting
```

Conceptually:

```text
High Bias     → Too Simple
High Variance → Too Complex
```

The goal is to find a model that generalizes well.

---

# 28. Model Selection

In real projects, you may train several candidate models.

Example:

```text
Linear Regression
Random Forest
Gradient Boosting
XGBoost
```

Then compare them using an appropriate validation strategy.

```text
                Validation Score
Linear Regression      0.72
Random Forest          0.84
Gradient Boosting      0.87
XGBoost                0.89
```

The highest score isn't automatically the final choice.

You should also consider:

* Inference latency
* Memory requirements
* Interpretability
* Stability
* Cost
* Deployment constraints
* Business requirements

---

# 29. Practical Model-Building Workflow

A production-oriented workflow looks like:

```text
Prepared Dataset
       ↓
Define Baseline
       ↓
Select Candidate Algorithms
       ↓
Train Models
       ↓
Cross-Validation
       ↓
Tune Hyperparameters
       ↓
Compare Models
       ↓
Check Overfitting / Underfitting
       ↓
Select Candidate Model
       ↓
Final Evaluation
```

---

# 30. Important Libraries

For traditional ML model building, **scikit-learn** is one of the most important Python libraries.

It provides implementations for:

```text
Classification
Regression
Clustering
Preprocessing
Model Selection
Cross-Validation
Hyperparameter Tuning
Metrics
Pipelines
```

Typical imports:

```python
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
```

---

# 31. How This Fits Into the ML Lifecycle

We have now covered:

```text
PART 1
Problem Definition
       ↓
Data Collection

PART 2
Data Preparation
       ↓
Feature Engineering
       ↓
Feature Selection
       ↓
Data Splitting
       ↓
Preprocessing Pipelines

PART 3
Model Building
       ↓
Algorithm Selection
       ↓
Training
       ↓
Cross-Validation
       ↓
Overfitting / Underfitting
```

The next major stage is **Model Evaluation**, where we determine whether the trained model actually performs well enough for the intended task.

---

# 32. Interview Quick Revision 🎯

### Q: What is model building?

Model building is the process of selecting an appropriate ML algorithm, training it on prepared data, tuning it, and selecting a model that generalizes well to unseen data.

### Q: What is supervised learning?

A learning paradigm where a model learns from labeled examples consisting of input features and known target values.

### Q: Classification vs regression?

```text
Classification → Predict category
Regression     → Predict continuous value
```

### Q: What is unsupervised learning?

Learning from unlabeled data to discover patterns or structure, such as clusters.

### Q: What is K-Fold Cross-Validation?

A validation technique that divides data into K folds and repeatedly trains on K−1 folds while validating on the remaining fold.

### Q: What is overfitting?

When a model performs very well on training data but poorly on unseen data because it has learned training-specific patterns/noise.

### Q: What is underfitting?

When a model is too simple to capture the underlying patterns, resulting in poor performance on both training and unseen data.

### Q: Parameters vs hyperparameters?

```text
Parameters     → Learned from training data
Hyperparameters → Configured/tuned during model development
```

### Q: Why is cross-validation useful?

It provides a more robust estimate of model performance by evaluating the model across multiple train/validation splits.

### Q: What is LOOCV?

LOOCV uses one observation for validation and all remaining observations for training, repeating the process once for every observation.

---

# ⭐ Final Mental Model

```text
                 MODEL BUILDING
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   Supervised     Unsupervised   Reinforcement
        │              │              │
   ┌────┴────┐         ↓              ↓
   ↓         ↓     Clustering      Rewards
Classification Regression
   │         │
   ↓         ↓
Algorithms  Algorithms
   │
   ↓
Training
   ↓
Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Check Overfitting / Underfitting
   ↓
Select Model
```

**Core idea:**

> **Model building is not just “train a model.” It is an iterative process of choosing algorithms, training them, validating them, tuning them, checking generalization, and selecting an appropriate model for the real-world problem.**
