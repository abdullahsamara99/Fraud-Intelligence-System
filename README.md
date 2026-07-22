# Fraud Intelligence System

An end-to-end Explainable AI (XAI) fraud detection platform that combines Machine Learning, Feature Engineering, SHAP Explainability, Business Decision Rules, and a Local Large Language Model (LLM) to detect suspicious banking transactions and generate human-readable fraud analysis.

---

# Overview

Financial fraud continues to evolve in sophistication, making traditional rule-based detection systems insufficient for identifying novel attack patterns.

The Fraud Intelligence System addresses this challenge by combining:

- Advanced Feature Engineering
- Isolation Forest Anomaly Detection
- Fraud Risk Score Normalization
- Business Decision Engine
- SHAP Explainable AI
- Local Large Language Model (Qwen2.5 via Ollama)
- FastAPI REST API
- MLflow Experiment Tracking

Unlike traditional fraud detection systems that only classify transactions as fraudulent or legitimate, this platform explains *why* a transaction is considered suspicious and recommends an appropriate business action.

---

# Key Features

- Isolation Forest unsupervised fraud detection
- Customer behavioral profiling
- Geographic inconsistency detection
- Transaction frequency analysis
- Absolute fraud risk normalization (0–1)
- SHAP feature attribution
- Local LLM executive fraud summaries
- Business decision engine
- FastAPI REST API
- MLflow experiment tracking
- Modular production-ready architecture

---

# System Architecture

```text
                 Incoming Transaction
                          │
                          ▼
                 Feature Engineering
                          │
                          ▼
                 Data Preprocessing
                          │
                          ▼
                 Isolation Forest
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
  Raw Anomaly Score              Prediction
          │                               │
          └───────────────┬───────────────┘
                          ▼
              Risk Score Normalization
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
                Executive Fraud Report
                          │
                          ▼
                  FastAPI REST API
```

Fraud_System/

├── api/
│   ├── app.py
│   ├── routes.py
│   └── schemas.py
│
├── config/
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
├── tests/
│   ├── test_decision_engine.py
│   ├── test_feature_engineering.py
│   └── test_predictor.py
│
├── training/
│   ├── train.py
│   └── evaluate.py
│
├── utils/
│   └── logger.py
│
├── requirements.txt
└── README.md

---

# Dataset

The project uses two CSV datasets.

## Customers Dataset

Contains customer profile information including:

- Customer ID
- Account Number
- Home Country
- Home City
- Card Type
- Monthly Income
- Account Age

## Transactions Dataset

Contains transaction details including:

- Transaction Amount
- Currency
- Merchant
- Merchant Category
- Transaction Country
- Transaction City
- Timestamp
- Transaction Type
- Payment Channel
- Cross-border Indicator

The datasets are merged using:

```text
customer_id
```

---


# Feature Engineering

The system creates domain-specific behavioral features that improve anomaly detection beyond raw transaction values.

## Time-Based Features

- Transaction Hour
- Day of Week
- Weekend Indicator
- Night Transaction Indicator

These features help identify suspicious activity occurring during unusual periods.

---

## Customer Behavioral Features

- Customer Average Transaction Amount
- Customer Transaction Count
- Rolling 24-Hour Transaction Count
- Amount-to-Income Ratio

These features model customer spending behavior and detect deviations from historical patterns.

---

## Geographic Features

- Cross-border Transaction Indicator
- Home Country
- Home City

These features identify transactions that occur outside the customer's normal geographic profile.

---

## Merchant Features

- Merchant Category
- Merchant Category Frequency

Merchant behavior is incorporated to detect abnormal purchasing patterns.

---

# Data Preprocessing

Before model training, all features are preprocessed using Scikit-learn pipelines.

## Numerical Features

- Missing value imputation using Median
- StandardScaler normalization

## Categorical Features

- Missing value imputation using Most Frequent
- Ordinal Encoding
- Unknown category handling

The preprocessing pipeline is serialized and reused during inference to guarantee identical feature transformations.

---

# Machine Learning Model

The fraud detector uses an Isolation Forest, an unsupervised anomaly detection algorithm.

Unlike supervised classifiers, Isolation Forest does not require fraud labels, making it suitable for real-world fraud datasets where confirmed fraud cases are rare.

## Model Configuration

| Parameter | Value |
|------------|--------|
| Algorithm | Isolation Forest |
| Trees | 200 |
| Contamination | 0.01 |
| Random State | 42 |
| Parallel Jobs | All CPU Cores |

---

# Fraud Risk Score

Isolation Forest outputs a raw anomaly score through its `decision_function()`.

Because these values are not intuitive for business users, the system converts them into a normalized fraud risk score between 0 and 1.

```
Raw Isolation Forest Score
            │
            ▼
 Clip to Training Score Range
            │
            ▼
 Absolute Score Normalization
            │
            ▼
 Fraud Risk Score (0 – 1)
```

### Advantages

- Stable across API requests
- Independent of batch size
- Suitable for single transaction scoring
- Easy interpretation by analysts

| Risk Score | Interpretation |
|------------|---------------|
| 0.00 | Very Low Risk |
| 0.50 | Medium Risk |
| 1.00 | Very High Risk |

---

# Business Decision Engine

The machine learning output is translated into operational business decisions.

## Decision Rules

| Condition | Risk Level | Recommended Action |
|------------|------------|--------------------|
| Isolation Forest Prediction = -1 OR Risk Score ≥ 0.80 | High | Block Transaction |
| Risk Score ≥ 0.50 | Medium | Manual Review |
| Risk Score < 0.50 | Low | Approve |

This separation between machine learning and business logic makes decision thresholds configurable without retraining the model.

---

# Explainable AI

The platform integrates SHAP (SHapley Additive exPlanations) to explain every prediction.

Instead of simply indicating whether a transaction is anomalous, SHAP identifies the features that contributed most to the prediction.

Each explanation includes:

- Feature Name
- SHAP Impact Score
- Risk Direction
- Ranked Importance

Example:

```json
"top_features": [
    {
        "feature": "transaction_amount",
        "impact": 0.05583,
        "direction": "Increase Risk"
    },
    {
        "feature": "is_weekend",
        "impact": 0.04657,
        "direction": "Increase Risk"
    }
]
```

---

# SHAP Optimization

KernelExplainer is computationally expensive.

To improve inference speed, the project performs several optimizations.

### K-Means Background Summarization

The training data is summarized using SHAP K-Means clustering before building the explainer.

Benefits include:

- Smaller background dataset
- Lower memory usage
- Faster SHAP computation

---

### DenseData Compatibility Fix

Recent SHAP versions wrap K-Means results inside a DenseData object.

To prevent runtime failures, the project extracts the underlying NumPy array before passing it to KernelExplainer.

This resolves the common:

```
DenseData object is not callable
```

error encountered in production deployments.

---

### Cached SHAP Explainer

The SHAP explainer is built only once during application startup.

```
Server Startup
      │
      ▼
Load Background Dataset
      │
      ▼
Feature Engineering
      │
      ▼
Preprocessing
      │
      ▼
K-Means Sampling
      │
      ▼
KernelExplainer
      │
      ▼
Cached for Future Requests
```

Every API request reuses the cached explainer, significantly reducing response time.

---

# LLM Explainability Layer

The system integrates a locally hosted Large Language Model using Ollama.

## Model

```
qwen2.5:3b
```

The LLM receives:

- Risk Level
- Recommended Action
- Top SHAP Features

It then generates a concise executive summary suitable for fraud analysts and AML investigators.

Example:

> Transaction flagged as High risk primarily due to abnormal spending behavior and unusual transaction timing. Immediate investigation is recommended before authorizing the payment.

---

## Rule-Based Fallback

If the Ollama server is unavailable or times out, the system automatically generates a deterministic explanation based on SHAP outputs.

This ensures the API always returns an explanation, even without the LLM.

---

# MLflow Experiment Tracking

Model training is tracked using MLflow.

Each training run automatically logs:

## Parameters

- Algorithm
- Number of Trees
- Contamination
- Random Seed

## Metrics

- Number of Samples
- Number of Features
- Number of Numerical Features
- Number of Categorical Features
- Total Training Time

## Artifacts

- Trained Isolation Forest Model
- Preprocessing Pipeline
- Metadata JSON

MLflow enables reproducible experiments and comparison between model versions.

To launch the MLflow UI:

```bash
mlflow ui
```

The dashboard is available at:

```
http://127.0.0.1:5000
```

---



# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/abdullahsamara99/Fraud-Intelligence-System.git
cd Fraud-Intelligence-System
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install Ollama

Download Ollama from:

https://ollama.com/download

Pull the Qwen model:

```bash
ollama pull qwen2.5:3b
```

Start Ollama:

```bash
ollama serve
```

The Fraud Intelligence System will automatically connect to:

```
http://localhost:11434
```

---

# Training the Model

Train the Isolation Forest model:

```bash
python -m training.train
```

The training pipeline performs:

- Data loading
- Feature engineering
- Data preprocessing
- Isolation Forest training
- Model serialization
- MLflow experiment logging

Generated artifacts:

```
models/
│
├── isolation_forest.pkl
├── scaler.pkl
└── metadata.json
```

---

# Running MLflow

After training the model, launch MLflow:

```bash
mlflow ui
```

Open your browser:

```
http://127.0.0.1:5000
```

MLflow tracks:

- Experiment history
- Parameters
- Metrics
- Models
- Artifacts

---

# Running the API

Start the FastAPI application:

```bash
uvicorn api.app:app --reload
```

Default URL:

```
http://127.0.0.1:8000
```

Interactive Swagger documentation:

```
http://127.0.0.1:8000/docs
```

Alternative ReDoc documentation:

```
http://127.0.0.1:8000/redoc
```

---

# API Endpoint

## POST /analyze

Accepts one transaction and returns:

- Isolation Forest prediction
- Raw anomaly score
- Normalized fraud risk score
- Business decision
- SHAP feature importance
- LLM-generated explanation

---

# Example Request

```json
{
  "transaction_id": "TX100001",
  "customer_id": "C1001",
  "account_number": "ACC0001",
  "timestamp": "2025-07-15 22:45:00",
  "transaction_date": "2025-07-15",
  "transaction_time": "22:45:00",
  "transaction_amount": 9500,
  "currency": "USD",
  "transaction_country": "United States",
  "transaction_country_iso": "US",
  "transaction_city": "New York",
  "is_cross_border": 0,
  "merchant_name": "Apple Store",
  "merchant_category": "Electronics",
  "channel": "POS",
  "card_type": "Credit",
  "transaction_type": "Purchase",
  "home_country": "United States",
  "home_city": "New York",
  "avg_monthly_income": 4500,
  "account_age_days": 850
}
```

---

# Example Response

```json
{
  "prediction": "Anomalous",
  "anomaly_score": -0.0518,
  "risk_score": 0.4036,
  "risk_level": "High",
  "recommended_action": "Block Transaction",
  "top_features": [
    {
      "feature": "transaction_amount",
      "impact": 0.05583,
      "direction": "Increase Risk"
    },
    {
      "feature": "is_weekend",
      "impact": 0.04657,
      "direction": "Increase Risk"
    },
    {
      "feature": "home_city",
      "impact": 0.01927,
      "direction": "Increase Risk"
    },
    {
      "feature": "avg_monthly_income",
      "impact": 0.01879,
      "direction": "Increase Risk"
    },
    {
      "feature": "is_night",
      "impact": 0.01545,
      "direction": "Increase Risk"
    }
  ],
  "explanation": "Transaction flagged as High risk primarily due to abnormal patterns in transaction amount, weekend activity, and customer location. Manual investigation is recommended before approving the transaction."
}
```

---

# Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3 |
| Machine Learning | Scikit-learn |
| Explainability | SHAP |
| Local LLM | Qwen2.5:3b (Ollama) |
| API | FastAPI |
| Validation | Pydantic |
| Data Processing | Pandas |
| Model Serialization | Joblib |
| Experiment Tracking | MLflow |
| Visualization | Matplotlib |

---

# Future Improvements

Potential enhancements include:

- Supervised fraud classification models (XGBoost, LightGBM, CatBoost)
- Ensemble anomaly detection
- Real-time Kafka streaming
- Feature Store integration
- Model Registry with MLflow
- Automated retraining pipelines
- Docker Compose deployment
- Kubernetes deployment
- Authentication and authorization
- Grafana monitoring dashboard
- Prometheus metrics
- Cloud deployment (AWS, Azure, GCP)

---

# Author

**Abdullah Samara**

Artificial Intelligence Engineer

GitHub:
https://github.com/abdullahsamara99

LinkedIn:
https://www.linkedin.com/in/abdullahsamara99

---

# License

This project is intended for educational, research, and portfolio purposes.

---

## Acknowledgments

This project leverages several open-source technologies:

- Scikit-learn
- SHAP
- FastAPI
- MLflow
- Pandas
- Ollama
- Qwen2.5
- Joblib

Special thanks to the open-source community for providing the tools that made this project possible.