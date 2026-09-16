# AI/ML Lifecycle — Part 5

## Deployment 🚀

> **Goal:** Take a trained and evaluated ML model and make it available for real users, applications, or business systems.

---

# 1. What is Model Deployment?

**Model deployment** is the process of making a trained ML model available in a production environment so that it can generate predictions on new, real-world data.

The overall lifecycle looks like:

```text
Problem Definition
       ↓
Data Collection
       ↓
Data Preparation
       ↓
Feature Engineering
       ↓
Model Building
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Deployment
       ↓
Monitoring
       ↓
Retraining
       ↺
```

### Simple Example

Suppose we build a model that predicts whether a transaction is fraudulent.

During development:

```text
Transaction Data
       ↓
ML Model
       ↓
Fraud / Not Fraud
```

After deployment, a real application can send new transaction information to the model and receive a prediction.

---

# 2. Types of Deployment

The slides focus on two important deployment patterns:

1. **Batch Deployment**
2. **Real-Time Deployment**

The key difference is **when predictions are generated**.

---

# 3. Batch Deployment

In **batch deployment**, the model processes a large amount of data at scheduled intervals.

Instead of making a prediction immediately when each record arrives, data is collected and processed together.

### Flow

```text
Large Dataset
      ↓
Scheduled Job
      ↓
ML Model
      ↓
Predictions
      ↓
Database / Report / Application
```

### Example

A company wants to predict customer churn every night.

At midnight:

```text
Customer data
      ↓
Model
      ↓
Churn predictions
      ↓
Updated customer report
```

There is no requirement for an immediate prediction.

### Suitable For

* Daily reports
* Weekly/monthly forecasting
* Customer segmentation
* Offline recommendation generation
* Large-scale data processing
* Periodic risk scoring

### Advantages

* Efficient for large datasets
* Can process many records together
* Easier to manage computational resources
* Often cheaper than always-on real-time inference

### Disadvantage

Predictions are **not immediately available**.

---

# 4. Real-Time Deployment

In **real-time deployment**, the model generates a prediction immediately after receiving input.

### Flow

```text
User/Application
       ↓
API Request
       ↓
ML Model
       ↓
Prediction
       ↓
API Response
```

### Example — Fraud Detection

A customer makes a payment:

```text
Transaction
     ↓
API
     ↓
Fraud Detection Model
     ↓
Fraud Probability
     ↓
Approve / Reject / Review
```

The prediction needs to happen within a short time.

### Suitable For

* Fraud detection
* Recommendation systems
* Search ranking
* Credit decisions
* Chatbots
* Personalized applications
* Real-time anomaly detection

### Advantages

* Immediate predictions
* Supports interactive applications
* Useful for time-sensitive decisions

### Challenges

Real-time systems must handle:

* Low latency
* High traffic
* Availability
* Scaling
* Failures
* Model versioning

---

# 5. Batch vs Real-Time Deployment

| Feature             | Batch                    | Real-Time                   |
| ------------------- | ------------------------ | --------------------------- |
| Prediction timing   | Scheduled                | Immediate                   |
| Input               | Large batches            | Individual/small requests   |
| Latency requirement | Low                      | High                        |
| Example             | Daily churn prediction   | Fraud detection             |
| Infrastructure      | Usually simpler          | More complex                |
| Cost                | Often lower              | Often higher                |
| Suitable for        | Non-time-sensitive tasks | Time-sensitive applications |

### Easy Memory Trick

```text
Batch → "Predict later in groups"

Real-Time → "Predict now"
```

---

# 6. Model Inference

**Inference** means using a trained model to generate predictions on new data.

Training:

```text
Historical Data
      ↓
Model Training
      ↓
Trained Model
```

Inference:

```text
New Data
      ↓
Trained Model
      ↓
Prediction
```

### Important Distinction

```text
Training → Model learns
Inference → Model predicts
```

This distinction is extremely important in production ML.

---

# 7. Why Deployment Is More Than Saving a Model

A common beginner mistake is thinking:

> "I trained the model and saved `model.pkl`, so deployment is complete."

Not quite. 😄

A production ML system usually needs:

```text
Model
+
Preprocessing
+
Dependencies
+
Application/API
+
Infrastructure
+
Monitoring
+
Logging
+
Versioning
```

For example, if the model was trained after applying:

```text
Missing-value handling
↓
Encoding
↓
Scaling
↓
Feature engineering
↓
Model
```

the **same transformations must be applied correctly to production data**.

This is one reason preprocessing pipelines are important.

---

# 8. Containerization

The slides introduce **Docker** as a deployment technology.

## What is Docker?

Docker allows you to package an application along with its required dependencies into a **container**.

Think of a container as a standardized environment containing what the application needs to run.

For example:

```text
ML Application
├── Python
├── Libraries
├── Preprocessing code
├── Model
├── API code
└── Configuration
```

All of these can be packaged into a Docker image.

---

# 9. Why Docker Is Useful for ML

Without containerization, you may encounter:

```text
"It works on my laptop!" 😭
```

The development environment might have:

```text
Python 3.12
scikit-learn version A
numpy version B
```

while production has:

```text
Python 3.10
scikit-learn version C
numpy version D
```

This can cause compatibility problems.

Docker helps create a consistent environment.

### Basic Concept

```text
Dockerfile
    ↓
Docker Image
    ↓
Docker Container
    ↓
Running ML Application
```

---

# 10. Docker Image vs Container

This distinction is important.

### Docker Image

An **image** is a packaged template containing the application and its dependencies.

Think:

```text
Image = Blueprint / Package
```

### Docker Container

A **container** is a running instance of an image.

Think:

```text
Container = Running instance
```

### Easy Memory Trick

```text
Image → What to run
Container → Running it
```

---

# 11. Benefits of Containerization

Docker provides several useful benefits for ML deployment:

### 1. Consistency

The same environment can be used across:

```text
Development
Testing
Production
```

### 2. Dependency Isolation

Different applications can use different library versions without interfering with each other.

### 3. Portability

Containers can run across different environments that support Docker.

### 4. Scalability

Containers can be replicated when more capacity is required.

### 5. Versioning

Different application/model versions can be packaged separately.

### 6. Rollback

If a new deployment fails:

```text
Version 2 → Problem
      ↓
Rollback
      ↓
Version 1
```

This makes production recovery easier.

---

# 12. APIs for ML Models

A deployed ML model often needs to communicate with other applications.

This is commonly done using an **API**.

**API = Application Programming Interface**

For ML:

```text
Client Application
       ↓
HTTP Request
       ↓
ML API
       ↓
Model
       ↓
Prediction
       ↓
HTTP Response
```

---

# 13. REST API

A **REST API** can expose model predictions through HTTP endpoints.

For example:

```text
POST /predict
```

The client sends input:

```json
{
  "age": 35,
  "income": 50000,
  "credit_score": 720
}
```

The model processes the input and returns something like:

```json
{
  "prediction": "approved",
  "probability": 0.91
}
```

The exact API design depends on the application.

---

# 14. Why Use an API for ML?

An API allows different applications to use the same model.

For example:

```text
                 ┌── Web Application
                 │
                 ├── Mobile Application
                 │
                 ├── Internal Dashboard
                 │
                 └── Other Services
                        ↓
                    ML API
                        ↓
                     Model
```

The model does not need to be duplicated inside every application.

---

# 15. Microservices

The slides also introduce **microservices architecture**.

A **microservice** is a small, independently deployable service responsible for a specific function.

Instead of creating one huge application:

```text
Big Application
├── Authentication
├── Payments
├── Recommendations
├── Fraud Detection
├── Notifications
└── Reporting
```

we can separate functionality:

```text
Authentication Service
Payment Service
Recommendation Service
Fraud Detection Service
Notification Service
```

Each service can potentially be:

* Developed independently
* Deployed independently
* Scaled independently
* Updated independently

---

# 16. ML Model as a Microservice

An ML model can be deployed as its own service.

Example:

```text
                    Application
                         ↓
                  Recommendation API
                         ↓
                 Recommendation Model
```

Another service might handle:

```text
Fraud API
    ↓
Fraud Detection Model
```

This separation can make large systems easier to maintain.

---

# 17. Deployment Architecture — Simple Example

Imagine our **NYC Taxi Fare Prediction** project.

We trained a model using:

```text
pickup location
dropoff location
passenger count
trip distance
time
```

After evaluation, we save the trained model.

Production architecture could look like:

```text
User / Application
        ↓
     REST API
        ↓
  Input Validation
        ↓
Preprocessing Pipeline
        ↓
 Taxi Fare Model
        ↓
   Prediction
        ↓
    API Response
```

For production:

```text
API + Model + Dependencies
            ↓
        Docker Image
            ↓
       Docker Container
            ↓
       Cloud / Server
```

Then monitoring tracks whether everything continues working properly.

---

# 18. Model Versioning

Models change over time.

For example:

```text
Model v1
   ↓
Model v2
   ↓
Model v3
```

Each version may have:

* Different training data
* Different features
* Different algorithms
* Different hyperparameters
* Different performance

You should be able to identify which model version is currently serving predictions.

Example:

```text
Production
    ↓
Model v3
    ↓
Accuracy = 94%
```

If v3 performs badly:

```text
v3
 ↓
Rollback
 ↓
v2
```

---

# 19. Deployment Challenges

Production ML introduces problems that may not appear during experimentation.

### 1. Latency

How long does one prediction take?

```text
Request → Model → Response
```

### 2. Scalability

Can the system handle:

```text
10 requests/sec?
1,000 requests/sec?
100,000 requests/sec?
```

### 3. Reliability

What happens if:

* Model crashes?
* Server goes down?
* Database is unavailable?
* Input is invalid?

### 4. Dependency Management

Different versions of:

```text
Python
NumPy
Pandas
scikit-learn
PyTorch
TensorFlow
```

can create compatibility issues.

### 5. Model Updates

How do you safely move from:

```text
Model v1 → Model v2
```

without breaking production?

---

# 20. Deployment Strategies

You will encounter more advanced deployment strategies later in MLOps.

Important concepts to know:

### Blue-Green Deployment

Maintain two environments:

```text
Blue → Current Production
Green → New Version
```

Test the new version and then switch traffic.

---

### Canary Deployment

Send only a small percentage of traffic to the new model.

Example:

```text
Model v1 → 95% traffic
Model v2 → 5% traffic
```

If v2 works well:

```text
v1 → 70%
v2 → 30%
```

Eventually:

```text
v2 → 100%
```

This reduces deployment risk.

---

# 21. Important Production Concept: Training vs Serving

You should mentally separate these two systems.

### Training Pipeline

```text
Raw Data
   ↓
Cleaning
   ↓
Feature Engineering
   ↓
Training
   ↓
Evaluation
   ↓
Model Artifact
```

### Serving Pipeline

```text
New Input
   ↓
Validation
   ↓
Preprocessing
   ↓
Model
   ↓
Prediction
   ↓
Response
```

The training pipeline happens periodically.

The serving pipeline may run **millions of times** in production.

---

# 22. Key Interview Questions

### Q1. What is model deployment?

Making a trained ML model available in a production environment so it can generate predictions on new data.

### Q2. Batch vs real-time deployment?

```text
Batch → Predictions at scheduled intervals
Real-time → Predictions immediately when requests arrive
```

### Q3. What is Docker?

A containerization platform used to package applications and dependencies into portable, consistent environments.

### Q4. Why Docker for ML?

To ensure consistency, isolate dependencies, simplify deployment, and support scaling/versioning.

### Q5. What is a REST API?

An HTTP-based interface through which applications can communicate with a service, including a deployed ML model.

### Q6. What is a microservice?

A small, independently deployable service responsible for a specific functionality.

### Q7. What is inference?

Using a trained model to make predictions on new data.

---

# 23. Deployment — Final Mental Model 🧠

Remember this:

```text
             TRAINING
                ↓
          Trained Model
                ↓
          Model Evaluation
                ↓
             DEPLOY
                ↓
       ┌────────┴────────┐
       ↓                 ↓
     Batch           Real-Time
       ↓                 ↓
 Scheduled Jobs       API Requests
       ↓                 ↓
       └────────┬────────┘
                ↓
             Prediction
                ↓
           Monitoring
```

And remember the technologies/concepts introduced here:

```text
Deployment
   │
   ├── Batch Deployment
   ├── Real-Time Deployment
   ├── Model Inference
   ├── Docker
   ├── Containerization
   ├── REST APIs
   ├── Microservices
   ├── Model Versioning
   └── Deployment Strategies
```

### ⭐ What you should actually remember for your AI/ML Engineer journey

Don't just memorize definitions. You should be able to explain a complete scenario:

> **"I trained and evaluated a model. I packaged the model and its preprocessing pipeline into a Docker container, exposed inference through a REST API, deployed it as a service, and can scale/version/rollback it when necessary."**

That is the level of understanding we're aiming for. 🚀
