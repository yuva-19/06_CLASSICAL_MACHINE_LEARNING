# AI/ML Lifecycle — Part 1

## Lifecycle Overview, Problem Definition & Data Collection

---

# 1. AI/ML Lifecycle

The **AI/ML Lifecycle** is the complete sequence of steps involved in developing, deploying, and maintaining a machine learning system.

A machine learning project does not end when a model is trained. The model must be evaluated, deployed, monitored, and periodically improved.

### Typical AI/ML Lifecycle

```text
Problem Definition
       ↓
Data Collection
       ↓
Data Preparation
       ↓
Model Building
       ↓
Model Evaluation
       ↓
Deployment
       ↓
Monitoring & Maintenance
       ↓
Retraining / Improvement
       ↺
```

The lifecycle is **iterative**, meaning the process can return to an earlier stage when problems or new requirements are discovered.

---

## 1.1 Major Stages

| Stage                           | Main Purpose                                                |
| ------------------------------- | ----------------------------------------------------------- |
| **1. Problem Definition**       | Understand the business problem and define the ML objective |
| **2. Data Collection**          | Gather relevant data from appropriate sources               |
| **3. Data Preparation**         | Clean, transform, and prepare data for ML                   |
| **4. Model Building**           | Select algorithms and train models                          |
| **5. Model Evaluation**         | Measure model performance and generalization                |
| **6. Deployment**               | Make the model available for real-world use                 |
| **7. Monitoring & Maintenance** | Track performance, errors, drift, and system health         |
| **8. Retraining**               | Update the model when data or performance changes           |

---

# 2. Problem Definition

**Problem Definition** is the first and one of the most important stages of an AI/ML project.

It involves converting a real-world/business problem into a clearly defined **machine learning problem**.

A poorly defined problem can lead to a technically good model that solves the wrong problem.

---

## 2.1 Business Problem vs ML Problem

### Business Problem

Describes what the organization actually wants to solve.

**Example:**

> A taxi company wants to estimate the fare of a trip before the customer completes the ride.

### ML Problem

Converts that business requirement into a machine learning task.

> Predict the taxi fare using features such as distance, passenger count, pickup location, drop-off location, and time.

This becomes a **regression problem** because the target is a continuous numerical value.

---

# 3. Define the ML Objective

Before building a model, clearly identify:

### 3.1 Input

What information will be given to the model?

Example:

```text
Trip Distance
Passenger Count
Pickup Location
Drop-off Location
Time
Day of Week
```

### 3.2 Target / Output

What should the model predict?

Example:

```text
Predicted Taxi Fare
```

### 3.3 Features

The input variables used by the model are called **features**.

Example:

```text
distance = 8.5 km
passengers = 2
pickup_hour = 18
```

### 3.4 Target Variable

The value that the model is trying to predict is called the **target variable**, **label**, or **dependent variable**, depending on the context.

Example:

```text
Target = Fare Amount
```

---

# 4. Identify the Type of ML Problem

The problem definition stage should determine what type of machine learning task is required.

## Classification

Used when the output belongs to a category.

Examples:

```text
Spam / Not Spam
Fraud / Not Fraud
Disease / No Disease
Cat / Dog
```

Output:

```text
Discrete category
```

---

## Regression

Used when predicting a continuous numerical value.

Examples:

```text
House Price
Taxi Fare
Temperature
Sales Revenue
```

Output:

```text
Continuous numerical value
```

---

## Clustering

Used to group similar data points when there are **no predefined labels**.

Examples:

```text
Customer Segmentation
Grouping Similar Products
Grouping News Articles
```

Output:

```text
Groups / Clusters
```

---

## Reinforcement Learning

Used when an **agent learns by interacting with an environment** and receiving rewards or penalties.

Examples:

```text
Game-playing agents
Robotics
Autonomous decision-making
```

---

# 5. Define Success Criteria

A model should not simply be described as "good" or "accurate."

The project should define measurable **success criteria**.

For example:

```text
Business Goal:
Reduce fraudulent transactions.

ML Goal:
Detect fraudulent transactions.

Success Criterion:
Achieve at least 90% recall while keeping the false-positive
rate below an acceptable threshold.
```

Different ML problems require different evaluation metrics.

### Examples

| Problem        | Possible Metric                 |
| -------------- | ------------------------------- |
| Classification | Accuracy, Precision, Recall, F1 |
| Regression     | MAE, MSE, RMSE, R²              |
| Ranking        | MAP, NDCG                       |
| NLP generation | BLEU, ROUGE                     |
| Clustering     | Silhouette Score                |

---

# 6. Understand Constraints

Real-world ML projects have constraints beyond model accuracy.

Important constraints include:

* **Time constraints**
* **Budget**
* **Computational resources**
* **Latency requirements**
* **Data availability**
* **Data privacy**
* **Security**
* **Regulatory requirements**
* **Scalability**
* **Interpretability**

### Example

A fraud detection system may need:

```text
Prediction latency < 100 ms
```

A highly accurate model that takes 10 seconds to make a prediction may be unsuitable for real-time fraud detection.

---

# 7. Data Collection

Once the problem has been clearly defined, the next major stage is **Data Collection**.

**Data Collection** is the process of gathering relevant data required to train, validate, and evaluate an ML model.

The quality of the collected data strongly influences the quality of the resulting model.

> **Garbage In → Garbage Out (GIGO)**

If the training data is poor, incomplete, biased, or irrelevant, the model is likely to produce poor predictions.

---

# 8. Sources of Data

Data can come from many different sources.

## 8.1 Databases

Examples:

* MySQL
* PostgreSQL
* MongoDB
* Oracle
* SQL Server

Example:

```text
Customer database
Transaction database
Product database
```

---

## 8.2 APIs

An **API (Application Programming Interface)** allows applications to access data or functionality from another system.

Example:

```text
Weather API
Payment API
Maps API
Stock Market API
```

An ML pipeline can periodically collect data through APIs.

---

## 8.3 Files

Common sources include:

```text
CSV
JSON
Excel
Parquet
XML
Text files
```

Example:

```text
customer_data.csv
transactions.parquet
reviews.json
```

---

## 8.4 Web Data

Data can be collected from websites when permitted.

Examples:

* Product information
* Public datasets
* News articles
* Reviews
* Public statistics

Web scraping should respect the website's terms, robots policies, and applicable laws.

---

## 8.5 Sensors and IoT Devices

Sensors can continuously generate data.

Examples:

```text
Temperature sensors
GPS devices
Smart watches
Industrial machines
Cameras
```

This is particularly important for:

* Predictive maintenance
* Healthcare applications
* Autonomous systems
* Smart manufacturing

---

## 8.6 User-Generated Data

Examples:

```text
Customer reviews
Search queries
Clicks
Ratings
Social media interactions
Purchase history
```

This type of data can be extremely useful for recommendation and personalization systems.

---

# 9. Data Collection Methods

Different methods can be used depending on the problem.

### Manual Collection

Humans collect and enter data.

Example:

```text
Doctors manually labeling medical images.
```

### Automated Collection

Software automatically collects data.

Example:

```text
IoT sensor → Data pipeline → Database
```

### API-Based Collection

```text
API → Python script → Raw data → Storage
```

### Database Extraction

```text
Database → SQL Query → Dataset
```

---

# 10. Data Quality

Before using collected data, its quality should be considered.

Important data-quality dimensions include:

### Accuracy

Does the data correctly represent reality?

### Completeness

Are important values missing?

### Consistency

Are values represented consistently?

Example:

```text
India
IND
India
```

These may represent the same country but are stored differently.

### Relevance

Does the data actually help solve the ML problem?

### Timeliness

Is the data recent enough for the application?

For example, old customer behavior may not accurately represent current behavior.

### Uniqueness

Are duplicate records present?

---

# 11. Data Quantity vs Data Quality

More data does **not automatically mean better data**.

Consider:

```text
Dataset A:
1,000,000 noisy and incorrect records

Dataset B:
100,000 high-quality relevant records
```

Dataset B may produce a better model depending on the problem.

Therefore:

> **Data quality, relevance, representativeness, and correctness are often more important than simply increasing data volume.**

---

# 12. Data Labeling

For many supervised learning problems, collected data needs to be **labeled**.

A label represents the expected output associated with an input.

### Example: Image Classification

```text
Image → Label

Image 1 → Cat
Image 2 → Dog
Image 3 → Cat
```

### Example: Spam Detection

```text
Email → Label

"Win ₹10,000!" → Spam
"Meeting at 5 PM" → Not Spam
```

The labeled dataset can then be used to train a supervised ML model.

---

# 13. Data Bias

Collected data may contain **bias**, which can cause the model to perform poorly or unfairly.

### Example

Suppose a face-recognition model is trained mostly using images from one demographic group.

The model may perform significantly worse on groups that were underrepresented in the training data.

Therefore, collected data should ideally be:

* Representative
* Diverse
* Relevant
* Sufficiently large
* Properly labeled

---

# 14. Data Leakage

**Data leakage** occurs when information that would not legitimately be available at prediction time is accidentally included in the training process.

This can make a model appear extremely accurate during development but perform poorly in production.

### Example

Suppose we want to predict whether a customer will default on a loan.

If the training data contains:

```text
Loan Default Status
```

as an input feature, the model has access to the answer it is supposed to predict.

This creates leakage.

### Key Rule

> **Only use information that would genuinely be available when the prediction is made.**

Data leakage is one of the most important issues to watch for during an ML project.

---

# 15. Example: NYC Taxi Fare Prediction

A simplified lifecycle beginning with problem definition could look like:

```text
Business Problem
       ↓
Predict taxi fare
       ↓
ML Problem
       ↓
Regression
       ↓
Collect historical taxi trip data
       ↓
Identify features
       ↓
Distance
Passenger Count
Pickup Location
Drop-off Location
Time
       ↓
Target
       ↓
Fare Amount
```

Later stages would transform and prepare this data before training and evaluating a regression model.

---

# 16. Key Takeaways

```text
AI/ML Lifecycle
    ↓
Problem Definition
    ↓
Data Collection
    ↓
Data Preparation
    ↓
Model Building
    ↓
Model Evaluation
    ↓
Deployment
    ↓
Monitoring & Maintenance
    ↓
Retraining / Improvement
```

### Remember These

* **Problem Definition** converts a real-world problem into an ML problem.
* Clearly identify **inputs, features, target, and expected output**.
* Determine whether the task is **classification, regression, clustering, or reinforcement learning**.
* Define measurable **success criteria** before building the model.
* Consider real-world constraints such as **latency, cost, scalability, privacy, and interpretability**.
* **Data Collection** gathers the information needed for ML.
* Data can come from **databases, APIs, files, websites, sensors, and user interactions**.
* Good ML systems require **high-quality, relevant, representative data**.
* **Data labeling** is essential for many supervised learning problems.
* Watch carefully for **bias and data leakage**.
* The ML lifecycle is **iterative**, not strictly linear.

---

## Interview Quick Revision 🎯

**Q: What is the ML lifecycle?**
The ML lifecycle is the end-to-end process of defining an ML problem, collecting and preparing data, building and evaluating models, deploying them, and continuously monitoring and maintaining them.

**Q: Why is problem definition important?**
Because a clearly defined problem determines the appropriate ML task, target variable, features, evaluation metrics, and business success criteria.

**Q: What is the difference between classification and regression?**
Classification predicts discrete categories, while regression predicts continuous numerical values.

**Q: What is data leakage?**
Data leakage occurs when information that should not be available during prediction is used during model training, resulting in unrealistically high evaluation performance.

**Q: Does more data always mean a better model?**
No. Data quality, relevance, correctness, and representativeness are also critical.

**Q: Why is data collection important?**
Because the model learns patterns from the collected data, and poor-quality or biased data can directly lead to poor model performance.
