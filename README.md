# Fraud Intelligence System

An end-to-end Explainable AI (XAI) fraud detection platform that combines unsupervised Machine Learning, Feature Engineering, SHAP Explainability, a Business Decision Engine, and a local Large Language Model (LLM) to detect suspicious banking transactions and generate human-readable fraud analysis.

---

# Overview

Financial fraud continues to evolve in sophistication, and labeled fraud data in real banking systems is scarce, delayed, and heavily imbalanced. Traditional supervised approaches struggle in this setting.

The Fraud Intelligence System addresses this by relying on **unsupervised anomaly detection** as its core modeling approach, then layering reasoning and operational logic on top:

- Advanced Feature Engineering
- Isolation Forest Anomaly Detection (unsupervised)
- Fraud Risk Score Normalization
- Business Decision Engine (4 operational actions)
- SHAP Explainable AI with a noise floor
- Local Large Language Model (Qwen2.5 via Ollama) with a grounded rule-based fallback
- FastAPI REST API
- MLflow Experiment Tracking

Unlike systems that only classify transactions as fraudulent or legitimate, this platform explains *why* a transaction is suspicious and recommends an appropriate business action.

---

# Key Features

- Isolation Forest unsupervised fraud detection (no fraud labels used in training)
- Customer behavioral, geographic, temporal, and merchant features
- Batch-independent risk normalization (0–1) suitable for single-transaction scoring
- SHAP feature attribution with a signal floor to suppress noise
- Local LLM executive summaries, with a risk-aware deterministic fallback
- Four-tier business decision engine (Approve / Flag / Block / Escalate)
- FastAPI REST API (`POST /analyze_transaction`)
- MLflow experiment tracking and reproducible training
- Modular, config-driven architecture

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
  Raw Anomaly Score              Prediction (-1 / 1)
          │                               │
          └───────────────┬───────────────┘
                          ▼
              Risk Score Normalization (0–1)
                          │
                          ▼
             Business Decision Engine
                          │
                          ▼
               SHAP Explainability
                          │
                          ▼
          Local LLM (Qwen2.5) / Rule-based Fallback
                          │
                          ▼
                Structured JSON Response
                          │
                          ▼
                  FastAPI REST API
```

---

# Project Structure

```text
Fraud-Intelligence-System/
├── api/
│   ├── app.py
│   ├── routes.py          # POST /analyze_transaction (+ /analyze alias)
│   └── schemas.py         # TransactionRequest + FraudPredictionResponse
├── config/
│   └── config.yaml
├── data/                  # gitignored — provide customers.csv & transactions.csv
│   └── raw/
├── features/
│   ├── preprocess.py
│   └── feature_engineering.py
├── inference/
│   ├── predictor.py
│   ├── explain.py
│   ├── decision_engine.py
│   └── pipeline.py
├── models/                # gitignored — produced by training
├── tests/
│   ├── test_decision_engine.py
│   ├── test_feature_engineering.py
│   └── test_predictor.py
├── training/
│   ├── train.py
│   └── evaluate.py
├── utils/
│   ├── config.py
│   └── logger.py
├── requirements.txt
└── README.md
```

> **Note:** `data/` and `models/` are gitignored. You must supply your own datasets (or a sample) at `data/raw/` before training, running the API, or running the tests, since the pipeline and tests load `data/raw/customers.csv` and `data/raw/transactions.csv`.

---

# Dataset

The project uses two CSV datasets merged on `customer_id`.

## Customers Dataset

- `customer_id`, `account_number`
- `home_country`, `home_city`
- `card_type`
- `avg_monthly_income`
- `account_age_days`

## Transactions Dataset

- `transaction_id`, `customer_id`, `account_number`
- `timestamp`, `transaction_date`, `transaction_time`
- `transaction_amount`, `currency`
- `transaction_country`, `transaction_country_iso`, `transaction_city`, `is_cross_border`
- `merchant_name`, `merchant_category`
- `channel`, `card_type`, `transaction_type`

The datasets are merged in `features/preprocess.py`. Column names must match the fields consumed by `features/feature_engineering.py`; renamed columns will raise a `KeyError`.

An optional `is_fraud` column, if present, is treated strictly as **hidden ground truth for proxy evaluation only**. It is never used during training or inference, preserving the unsupervised requirement.

---

# Feature Engineering

The system creates domain-specific behavioral features beyond raw transaction values.

**Time-based:** transaction hour, day of week, weekend indicator, night-transaction indicator.

**Customer behavioral:** running average transaction amount, transaction count, rolling 24-hour and 7-day transaction counts, amount-to-income ratio.

**Customer aggregates (baseline behavior):** per-customer group-by aggregates that capture typical behavior across four dimensions — daily spending velocity (`cust_daily_total_amount`, `cust_daily_txn_count`), merchant-category behavior (`cust_cat_total_amount`, `cust_cat_txn_count`), geographic exposure (`cust_country_txn_count`), and payment channel (`cust_channel_txn_count`).

**Geographic:** cross-border indicator, home-country match, home-city match.

**Merchant:** merchant category and merchant-category frequency.

---

# Data Preprocessing

All features are preprocessed with Scikit-learn pipelines and the fitted pipeline is serialized for identical inference-time transforms.

- **Numerical:** median imputation + StandardScaler.
- **Categorical:** most-frequent imputation + Ordinal Encoding with unknown-category handling (`unknown_value=-1`).

---

# Machine Learning Model

An **Isolation Forest** performs unsupervised anomaly detection. It requires no fraud labels, which fits real-world datasets where confirmed fraud is rare.

| Parameter | Value |
|-----------|-------|
| Algorithm | Isolation Forest |
| Trees | 200 |
| Contamination | 0.01 |
| Random State | 42 |
| Parallel Jobs | All CPU cores |

---

# Fraud Risk Score

Isolation Forest outputs a raw score via `decision_function()` where **higher = more normal**. The system converts it into a normalized fraud risk score in `[0, 1]` where higher = higher risk.

```
Raw Isolation Forest Score
            │
            ▼
 Clip to Symmetric Bounds  (-scale, +scale)
            │
            ▼
 Absolute (Batch-Independent) Normalization
            │
            ▼
 Fraud Risk Score (0 – 1)
```

Isolation Forest offsets `decision_function` so the inlier/outlier boundary sits at **0** (inliers score above 0, the ~1% outliers below). The calibration therefore normalizes **symmetrically around zero**: `score_min = -scale` and `score_max = +scale`, where `scale` is the 99th percentile of the training scores. This pins `risk = 0.5` exactly on the anomaly boundary, so every normal transaction (score > 0) maps into the Low band and every anomaly (score < 0) maps to Medium or higher. The bounds are persisted to `models/metadata.json` at training time and loaded by `inference/predictor.py`. Because they are absolute rather than per-batch, a single transaction always scores consistently regardless of request size.

| Risk Score | Interpretation |
|-----------|----------------|
| 0.00 | Very Low Risk |
| 0.50 | Medium Risk |
| 1.00 | Very High Risk |

---

# Business Decision Engine

Model output is mapped to one of four operational actions. Thresholds live in `config.yaml` and are configurable without retraining.

| Condition | Risk Level | Recommended Action |
|-----------|-----------|--------------------|
| Prediction = -1 **and** Risk Score ≥ `escalate_threshold` (0.90) | Critical | Escalate to Fraud Team |
| Prediction = -1 **or** Risk Score ≥ `high_threshold` (0.80) | High | Block Transaction |
| Risk Score ≥ `medium_threshold` (0.50) | Medium | Flag for Review |
| Otherwise | Low | Approve |

Because the High tier also triggers on the model's `-1` anomaly flag, a transaction can be blocked even when its normalized score sits mid-range — the model's own judgment is respected alongside the numeric threshold.

---

# Explainable AI

SHAP (SHapley Additive exPlanations) explains every prediction by attributing the decision to individual features.

**Sign convention.** For `IsolationForest.decision_function`, a positive SHAP value pushes the score toward *normal* (lowers risk); a negative value pushes toward *anomalous* (raises risk). Each feature is labeled `Increase Risk`, `Decrease Risk`, or `Negligible`.

**Signal floor.** On near-baseline (normal) transactions, SHAP contributions are tiny and noisy. Any contribution below a magnitude floor (`SIGNAL_FLOOR = 0.03`, tunable) is labeled `Negligible` so trivial attributions are never presented as real risk drivers. This keeps explanations internally consistent with the risk level.

Each explanation entry includes: feature name, absolute impact, and risk direction, ranked by impact.

## SHAP Performance Optimizations

- **K-Means background summarization** reduces the background set for KernelExplainer, lowering memory and latency.
- **DenseData compatibility fix:** the raw NumPy array is extracted from SHAP's `DenseData` wrapper to avoid the `DenseData object is not callable` error.
- **Cached explainer:** the explainer is built once at startup and reused across requests.

---

# LLM Explainability Layer

A locally hosted LLM (`qwen2.5:3b` via Ollama) receives the risk level, recommended action, and the non-negligible SHAP features, then produces a concise two-sentence analyst summary. The prompt constrains the model to the provided features only, with no external facts.

## Rule-Based Fallback

If Ollama is unavailable or times out, a deterministic, **risk-aware** fallback generates the summary from SHAP outputs:

- High / Critical → names the top risk drivers and the recommended action.
- Medium → notes mild deviations and suggests manual review.
- Low (or no feature clears the signal floor) → states the transaction is consistent with the customer's normal profile, with no significant anomalies.

This guarantees the API always returns an explanation that matches the decision, even without the LLM.

---

# Output Specification

```json
{
  "transaction_id": "TEST_ANOMALY_001",
  "prediction": "Anomalous",
  "anomaly_score": -0.0914,
  "risk_score": 0.7401,
  "risk_level": "High",
  "recommended_action": "Block Transaction",
  "top_features": [ ... ],
  "explanation": "..."
}
```

---

# Installation

## 1. Clone

```bash
git clone https://github.com/abdullahsamara99/Fraud-Intelligence-System.git
cd Fraud-Intelligence-System
```

## 2. Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Dependencies

```bash
pip install -r requirements.txt
```

## 4. Provide Data

Place your datasets at:

```
data/raw/customers.csv
data/raw/transactions.csv
```

## 5. (Optional) Ollama for LLM explanations

```bash
ollama pull qwen2.5:3b
ollama serve   # serves http://localhost:11434
```

If Ollama is not running, the rule-based fallback is used automatically.

---

# Training

```bash
python -m training.train
```

The pipeline loads data, engineers features, preprocesses, trains the Isolation Forest, serializes artifacts, and logs the run to MLflow.

Generated artifacts:

```
models/
├── isolation_forest.pkl
├── scaler.pkl
└── metadata.json     # model config + calibrated score_min / score_max
```

---

# Running the API

```bash
uvicorn api.app:app --reload
```

- API root: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Endpoint

### `POST /analyze_transaction`

Accepts a single transaction and returns the prediction, raw anomaly score, normalized risk score, business decision, SHAP feature importance, and an LLM/rule-based explanation. (`POST /analyze` is kept as a backward-compatible alias.)

---

# Example: Normal Transaction

**Request**

```json
{
  "transaction_id": "TEST_NORMAL_001",
  "customer_id": "CUST_00000",
  "account_number": "ACC1000000",
  "timestamp": "2025-03-18 13:20:00",
  "transaction_amount": 45.90,
  "currency": "JOD",
  "transaction_country": "Jordan",
  "transaction_city": "Amman",
  "is_cross_border": 0,
  "merchant_category": "Groceries",
  "channel": "POS",
  "card_type": "Credit",
  "transaction_type": "Purchase",
  "home_country": "Jordan",
  "home_city": "Amman",
  "avg_monthly_income": 900,
  "account_age_days": 800
}
```

**Response**

```json
{
  "transaction_id": "TEST_NORMAL_001",
  "prediction": "Normal",
  "anomaly_score": 0.0636,
  "risk_score": 0.3328,
  "risk_level": "Low",
  "recommended_action": "Approve",
  "top_features": [
    { "feature": "home_country",           "impact": 0.017045, "direction": "Negligible" },
    { "feature": "amount_to_income_ratio", "impact": 0.005749, "direction": "Negligible" },
    { "feature": "transaction_amount",     "impact": 0.005720, "direction": "Negligible" },
    { "feature": "transaction_country",    "impact": 0.004589, "direction": "Negligible" },
    { "feature": "transactions_last_24h",  "impact": 0.003922, "direction": "Negligible" }
  ],
  "explanation": "This transaction is consistent with the customer's established behavior profile, with no individual feature deviating significantly from normal patterns. It is assessed as Low risk and can be approved."
}
```

---

# Example: Anomalous Transaction

**Request** (foreign country, night hours, ~47× the customer's monthly income, gambling, online transfer)

```json
{
  "transaction_id": "TEST_ANOMALY_001",
  "customer_id": "CUST_00000",
  "account_number": "ACC1000000",
  "timestamp": "2025-03-19 03:12:00",
  "transaction_amount": 42000.00,
  "currency": "USD",
  "transaction_country": "UAE",
  "transaction_country_iso": "AE",
  "transaction_city": "Dubai",
  "is_cross_border": 1,
  "merchant_category": "Gambling",
  "channel": "Online",
  "card_type": "Credit",
  "transaction_type": "Transfer",
  "home_country": "Jordan",
  "home_city": "Amman",
  "avg_monthly_income": 900,
  "account_age_days": 800
}
```

**Response**

```json
{
  "transaction_id": "TEST_ANOMALY_001",
  "prediction": "Anomalous",
  "anomaly_score": -0.0914,
  "risk_score": 0.7401,
  "risk_level": "High",
  "recommended_action": "Block Transaction",
  "top_features": [
    { "feature": "transaction_amount",      "impact": 0.037638, "direction": "Increase Risk" },
    { "feature": "cust_daily_total_amount", "impact": 0.024405, "direction": "Negligible" },
    { "feature": "account_age_days",        "impact": 0.024274, "direction": "Negligible" },
    { "feature": "currency",                "impact": 0.018921, "direction": "Negligible" },
    { "feature": "transaction_type",        "impact": 0.015860, "direction": "Negligible" }
  ],
  "explanation": "The anomaly-detection model flagged a transaction with an unusually high amount of 42,000, indicative of potential fraudulent activity. Given this atypical transaction value in conjunction with our high-risk assessment, we recommend blocking this transaction to prevent any unauthorized financial impact. Risk level: High; Recommended action: Block Transaction."
}
```

> The explanation above is live output from the local LLM (Qwen2.5 via Ollama), grounded in the SHAP-attributed `transaction_amount` feature. When Ollama is unavailable, the deterministic rule-based fallback produces an equivalent narrative.

---

# Results & Evaluation

The two verified examples above demonstrate the system across the risk spectrum:

| | Normal transaction | Anomalous transaction |
|---|---|---|
| Prediction | Normal | Anomalous |
| Anomaly score | 0.0636 | -0.0914 |
| Risk score | 0.3328 | 0.7401 |
| Risk level | Low | High |
| Decision | Approve | Block Transaction |
| Explanation | Consistent with normal profile | LLM narrative citing the 42,000 amount |

With the symmetric-around-zero calibration, normal transactions map cleanly into the Low band (0.33) while anomalies rise into the High band (0.74) and are blocked. The risk score crosses 0.5 exactly at the model's anomaly boundary, so the numeric risk level and the model's own `-1` flag agree. The normal transaction shows all attributions as `Negligible`, so its explanation does not fabricate risk drivers, while the anomalous one surfaces `transaction_amount` as the dominant grounded driver.

## Proxy Evaluation (unsupervised)

Since training is unlabeled, model behavior is validated with proxy methods in `training/evaluate.py`:

- **Anomaly score distribution inspection** — histogram of normalized risk scores.
- **Distribution-shift analysis** — a two-sample Kolmogorov–Smirnov test between historical and recent batches, alerting only when the KS effect size exceeds a threshold (avoiding large-sample p-value artifacts).
- **Precision@K** — computed during training against a hidden `is_fraud` label (never used for model fitting). Transactions are ranked by anomaly score and the fraction of true frauds among the top-K is reported, logged to MLflow as `precision_at_100`, `precision_at_500`, `precision_at_1000`, and `precision_at_num_frauds`.

Latest run (from MLflow):

| Metric | Value |
|--------|-------|
| Precision@100 | _<fill from MLflow>_ |
| Precision@500 | _<fill from MLflow>_ |
| Precision@1000 | _<fill from MLflow>_ |
| Precision@(#frauds) | _<fill from MLflow>_ |

Run:

```bash
python -m training.evaluate
mlflow ui   # http://127.0.0.1:5000
```

---

# Known Limitations

- **Single-transaction baseline.** Behavioral features such as the customer's average amount and rolling 24h/7d counts are computed within the request batch. When the API scores one isolated transaction with no history, these features are flat, so SHAP tends to attribute risk to static profile fields (home country, income, account age) rather than behavioral deviation. In production these features would be backed by a feature store keyed on `customer_id`.
- **SHAP background granularity.** The KernelExplainer uses a small K-Means background (20 clusters) for speed, which can dilute the attribution of strong categorical signals (e.g. geography). Increasing the background sample sharpens attribution at the cost of latency.
- **Startup cost.** The SHAP background is built from the dataset at API startup; for large datasets, sample the background (e.g. `df.sample(5000)`) to keep startup fast and memory bounded.

---

# MLflow Experiment Tracking

Each training run logs parameters (algorithm, trees, contamination, seed), metrics (samples, features, training time), and artifacts (model, preprocessing pipeline, metadata JSON), enabling reproducible experiments and version comparison.

---

# Technology Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3 |
| Machine Learning | Scikit-learn (Isolation Forest) |
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

- Feature store for true historical behavioral baselines
- Ensemble anomaly detection (Isolation Forest + LOF)
- Model Registry and automated retraining
- Docker Compose / Kubernetes deployment
- Authentication, monitoring (Prometheus / Grafana), and cloud deployment

---

# Author

**Abdullah Samara** — Artificial Intelligence Engineer
GitHub: https://github.com/abdullahsamara99
LinkedIn: https://www.linkedin.com/in/abdullahsamara99

---

# License

Intended for educational, research, and portfolio purposes.

---

## Acknowledgments

Built with Scikit-learn, SHAP, FastAPI, MLflow, Pandas, Ollama, Qwen2.5, and Joblib. Thanks to the open-source community.