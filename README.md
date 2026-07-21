<<<<<<< HEAD
# Fraud Intelligence System

An end-to-end Explainable AI (XAI) fraud detection platform that combines machine learning, feature engineering, business rules, SHAP explainability, and a local Large Language Model (LLM) to detect suspicious financial transactions and generate human-readable fraud analysis.

---

# Overview

The Fraud Intelligence System is designed for real-time fraud detection in banking and financial services.

Instead of only identifying anomalous transactions, the system explains **why** a transaction is suspicious and recommends the appropriate business action.

The complete pipeline includes:

- Advanced Feature Engineering
- Isolation Forest Anomaly Detection
- Risk Score Normalization
- Business Decision Engine
- SHAP Explainability
- Local LLM Executive Summaries
- FastAPI REST API
- MLflow Experiment Tracking

---

# System Architecture

```text
                ┌────────────────────┐
                │ Incoming Transaction│
                └──────────┬─────────┘
                           │
                           ▼
                 Feature Engineering
                           │
                           ▼
                 Isolation Forest Model
                           │
                           ▼
               Absolute Risk Score (0-1)
                           │
                           ▼
               Business Decision Engine
                           │
                           ▼
                SHAP Explainability
                           │
                           ▼
           Local LLM (Qwen2.5 via Ollama)
                           │
                           ▼
                  FastAPI REST API
```

---

# Project Structure

```
Fraud_System/

│
├── api/
│   ├── app.py
│   ├── routes.py
│   └── schemas.py
│
├── data/
│   └── raw/
│
├── features/
│   ├── preprocess.py
│   └── feature_engineering.py
│
├── inference/
│   ├── predictor.py
│   ├── explain.py
│   ├── decision_engine.py
│   └── pipeline.py
│
├── models/
│
├── training/
│   └── train.py
│
├── evaluation/
│   └── evaluate.py
│
├── utils/
│   └── logger.py
│
└── README.md
```

---

# Dataset

The system uses two datasets.

### Customers

Contains customer profile information including

- Customer ID
- Home Country
- Home City
- Monthly Income
- Account Age
- Card Type

### Transactions

Contains transaction information including

- Transaction Amount
- Merchant
- Country
- City
- Currency
- Transaction Channel
- Timestamp
- Cross Border Indicator

The datasets are merged using:

```
customer_id
```

---

# Feature Engineering

The feature engineering module creates behavioral fraud indicators.

## Time Features

- Hour
- Day of Week
- Weekend Indicator
- Night Transaction Indicator

## Customer Features

- Average Transaction Amount
- Transaction Count
- Rolling 24-hour Transaction Count

## Merchant Features

- Merchant Category Frequency

## Geographic Features

- Cross Border Transaction
- Geographic Inconsistency Indicator

## Financial Features

- Amount to Monthly Income Ratio

These engineered features improve anomaly detection by capturing customer behavior rather than relying solely on raw transaction values.

---

# Data Preprocessing

The preprocessing pipeline is built using Scikit-learn.

### Numerical Features

- Median Imputation
- StandardScaler

### Categorical Features

- Most Frequent Imputation
- OrdinalEncoder
- Unknown category handling

The preprocessing pipeline is saved and reused during inference.

---

# Machine Learning Model

The fraud detector uses an **Isolation Forest**.

## Configuration

| Parameter | Value |
|------------|------:|
| Algorithm | Isolation Forest |
| Trees | 200 |
| Contamination | 0.01 |
| Random State | 42 |
| Parallel Jobs | All CPU Cores |

Isolation Forest is well suited because fraud labels are usually unavailable or extremely imbalanced.

---

# Fraud Risk Score

Isolation Forest produces a raw anomaly score.

Instead of batch normalization, the system applies **absolute normalization** using fixed score boundaries learned during training.

```
Raw Score
     │
     ▼
Clip to Training Range
     │
     ▼
Normalize
     │
     ▼
Risk Score (0–1)
```

Higher values indicate higher fraud risk.

Benefits:

- Stable inference
- Single transaction scoring
- Batch-independent predictions
- Consistent interpretation

---

# Business Decision Engine

The machine learning output is converted into operational decisions.

| Risk Score | Risk Level | Recommended Action |
|------------|------------|--------------------|
| ≥ 0.80 | High | Block Transaction |
| 0.50 – 0.79 | Medium | Manual Review |
| < 0.50 | Low | Approve |

Isolation Forest anomalies are always classified as **High Risk**.

---

# Explainable AI

The system integrates SHAP to explain every prediction.

## SHAP Features

- KernelExplainer
- K-Means summarized background dataset
- Cached explainer
- Feature attribution
- Production-ready optimization

For every transaction the API returns:

- Top contributing features
- Feature impact
- Risk direction

Example

```json
"top_features": [
    {
        "feature": "is_weekend",
        "impact": 0.021149,
        "direction": "Increase Risk"
    },
    {
        "feature": "home_city",
        "impact": 0.018796,
        "direction": "Increase Risk"
    }
]
```

---

# LLM Explainability Layer

The project integrates a locally hosted LLM using **Ollama**.

Model:

```
qwen2.5:3b
```

The LLM translates SHAP feature attributions into executive-level fraud summaries.

Example:

> Transaction flagged as Low risk primarily due to abnormal patterns in weekend activity, customer location, and monthly income characteristics. The transaction remains within acceptable operational thresholds and is recommended for approval.

If the LLM is unavailable, the system automatically falls back to a deterministic rule-based explanation.

---

# SHAP Optimization

The SHAP explainer is built only once during server startup.

```
API Startup
      │
      ▼
Load Background Data
      │
      ▼
Feature Engineering
      │
      ▼
Preprocessing
      │
      ▼
K-Means Background
      │
      ▼
KernelExplainer
      │
      ▼
Cache Explainer
```

This significantly reduces latency for real-time inference.

---

# REST API

## Home

```
GET /
```

Response

```json
{
    "message": "Fraud Intelligence System API"
}
```

---

## Analyze Transaction

```
POST /analyze
```

Example Response

```json
{
    "prediction": "Normal",
    "anomaly_score": 0.016,
    "risk_score": 0.268,
    "risk_level": "Low",
    "recommended_action": "Approve",
    "top_features": [
        {
            "feature": "is_weekend",
            "impact": 0.021149,
            "direction": "Increase Risk"
        },
        {
            "feature": "home_city",
            "impact": 0.018796,
            "direction": "Increase Risk"
        }
    ],
    "explanation": "Transaction flagged as Low risk primarily due to abnormal patterns in weekend activity, customer location, and customer financial profile."
}
```

---

# Model Training

Training is performed using

```
python -m training.train
```

The training pipeline performs

1. Load datasets
2. Feature engineering
3. Data preprocessing
4. Isolation Forest training
5. Model serialization
6. MLflow experiment tracking

Generated artifacts

```
models/

isolation_forest.pkl
scaler.pkl
metadata.json
```

---

# MLflow Experiment Tracking

The project integrates MLflow to track machine learning experiments.

Each run automatically logs:

## Parameters

- Algorithm
- Number of Trees
- Contamination
- Random Seed

## Metrics

- Number of Samples
- Number of Features
- Training Time
- Numerical Features
- Categorical Features

## Artifacts

- Trained Model
- Preprocessing Pipeline
- Metadata

This enables reproducible training and experiment comparison.

---

# Technologies

| Category | Technology |
|------------|------------|
| Language | Python 3 |
| API | FastAPI |
| Machine Learning | Scikit-learn |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Data Processing | Pandas |
| Numerical Computing | NumPy |
| Visualization | Matplotlib |
| Model Persistence | Joblib |
| Local LLM | Ollama |
| LLM Model | Qwen2.5:3B |

---

# Production Features

- Isolation Forest anomaly detection
- Advanced fraud feature engineering
- Geographic inconsistency detection
- Customer behavioral profiling
- Merchant behavior analytics
- Rolling transaction frequency analysis
- Stable 0–1 fraud risk score normalization
- Business decision engine
- SHAP Explainable AI
- K-Means optimized SHAP background sampling
- Cached SHAP explainer
- Local LLM fraud explanation generation
- Rule-based explanation fallback
- FastAPI REST API
- MLflow experiment tracking
- Model persistence
- Comprehensive logging
- Modular production-ready architecture

---

# Future Improvements

- Graph-based fraud detection
- Customer risk profiling
- Real-time streaming with Kafka
- Model registry using MLflow
- Drift detection
- Active learning
- Fraud investigation dashboard
- Role-based authentication
- Docker deployment
- Kubernetes deployment

---

# Author

**Fraud Intelligence System**

An Explainable AI fraud detection platform that combines anomaly detection, explainable machine learning, business decision rules, and local Large Language Models to support real-time fraud investigation in financial institutions.
=======
# Fraud-Intelligence-System
>>>>>>> 55dab63ca4b408cc2339be3a82df909c6c034b80
