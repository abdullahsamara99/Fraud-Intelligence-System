import json
import os
import time

import joblib
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from features.feature_engineering import create_features
from features.preprocess import load_data
from utils.logger import logger


# ==========================================================
# Paths
# ==========================================================

CUSTOMERS_PATH = "data/raw/customers.csv"
TRANSACTIONS_PATH = "data/raw/transactions.csv"

MODEL_PATH = "models/isolation_forest.pkl"
SCALER_PATH = "models/scaler.pkl"
METADATA_PATH = "models/metadata.json"

# Columns never fed to the model. `is_fraud` is a HIDDEN label used only
# for proxy evaluation (Precision@K) and must never become a feature.
DROP_COLUMNS = [
    "transaction_id",
    "customer_id",
    "account_number",
    "timestamp",
    "transaction_date",
    "transaction_time",
    "merchant_name",
    "transaction_country_iso",
    "is_fraud",
]

CATEGORICAL_COLUMNS = [
    "currency",
    "transaction_country",
    "transaction_city",
    "merchant_category",
    "channel",
    "card_type",
    "transaction_type",
    "home_country",
    "home_city",
]


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """
    Fraction of true frauds among the k most anomalous transactions.

    Lower IsolationForest.decision_function score = more anomalous, so we
    rank ascending and take the first k.
    """
    k = min(k, len(scores))
    if k == 0:
        return 0.0
    order = np.argsort(scores)            # ascending: most anomalous first
    top_k_labels = labels[order[:k]]
    return float(top_k_labels.sum()) / k


def train():
    """Train the Isolation Forest fraud model with MLflow tracking."""

    start_time = time.time()

    try:
        logger.info("=" * 60)
        logger.info("Starting Fraud Model Training")
        logger.info("=" * 60)

        mlflow.set_experiment("Fraud Detection Isolation Forest")

        with mlflow.start_run():
            logger.info("MLflow run started.")

            mlflow.log_param("algorithm", "IsolationForest")
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("contamination", 0.01)
            mlflow.log_param("random_state", 42)

            # ==================================================
            # Load Dataset
            # ==================================================
            logger.info("Loading dataset...")
            df = load_data(CUSTOMERS_PATH, TRANSACTIONS_PATH)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

            # ==================================================
            # Feature Engineering
            # ==================================================
            logger.info("Running feature engineering...")
            df = create_features(df)
            logger.info(f"Feature engineering completed. Shape: {df.shape}")

            # ==================================================
            # Extract hidden label (evaluation ONLY) before dropping
            # ==================================================
            labels = None
            if "is_fraud" in df.columns:
                labels = df["is_fraud"].to_numpy().astype(int)
                logger.info(
                    f"Found 'is_fraud' label ({labels.sum()} positives). "
                    f"Reserved for proxy evaluation; NOT used as a feature."
                )
            else:
                logger.info("No 'is_fraud' column found. Skipping Precision@K.")

            # ==================================================
            # Drop Non-Model Columns  (is_fraud included)
            # ==================================================
            logger.info("Dropping non-model columns...")
            df = df.drop(columns=DROP_COLUMNS, errors="ignore")
            logger.info(f"Model dataframe shape: {df.shape}")

            # ==================================================
            # Feature Types
            # ==================================================
            categorical_columns = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
            numeric_columns = [c for c in df.columns if c not in categorical_columns]

            logger.info(f"Numeric features: {len(numeric_columns)}")
            logger.info(f"Categorical features: {len(categorical_columns)}")
            mlflow.log_metric("num_numeric_features", len(numeric_columns))
            mlflow.log_metric("num_categorical_features", len(categorical_columns))

            # ==================================================
            # Preprocessing Pipeline
            # ==================================================
            logger.info("Building preprocessing pipeline...")

            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )),
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num", numeric_transformer, numeric_columns),
                ("cat", categorical_transformer, categorical_columns),
            ])

            logger.info("Applying preprocessing...")
            X = preprocessor.fit_transform(df)
            logger.info(f"Preprocessing completed. Feature matrix shape: {X.shape}")

            mlflow.log_metric("num_samples", X.shape[0])
            mlflow.log_metric("num_features", X.shape[1])

            # ==================================================
            # Train Model
            # ==================================================
            logger.info("Training Isolation Forest...")
            model = IsolationForest(
                n_estimators=200,
                contamination=0.01,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X)
            logger.info("Isolation Forest training completed successfully.")

            # ==================================================
            # Calibrate normalization bounds (FIX)
            # Predictor reads these from metadata for stable, batch-
            # independent 0-1 risk scores. Use robust percentiles.
            # ==================================================
            logger.info("Calibrating score normalization bounds...")
            train_scores = model.decision_function(X)
            score_min = float(np.percentile(train_scores, 1))
            score_max = float(np.percentile(train_scores, 99))

            # Guard against a degenerate range
            if score_max <= score_min:
                logger.warning("Degenerate score range; using raw min/max.")
                score_min = float(train_scores.min())
                score_max = float(train_scores.max())

            logger.info(f"score_min={score_min:.5f}, score_max={score_max:.5f}")
            mlflow.log_metric("score_min", score_min)
            mlflow.log_metric("score_max", score_max)

            # ==================================================
            # Proxy Evaluation: Precision@K (FIX)
            # Only runs if a hidden is_fraud label is available.
            # ==================================================
            if labels is not None and labels.sum() > 0:
                logger.info("Computing Precision@K...")
                for k in (100, 500, 1000):
                    p = precision_at_k(labels, train_scores, k)
                    logger.info(f"Precision@{k}: {p:.4f}")
                    mlflow.log_metric(f"precision_at_{k}", p)

                # Precision@(#actual frauds) — a natural operating point
                n_frauds = int(labels.sum())
                p_at_n = precision_at_k(labels, train_scores, n_frauds)
                logger.info(f"Precision@{n_frauds} (=#frauds): {p_at_n:.4f}")
                mlflow.log_metric("precision_at_num_frauds", p_at_n)

            # ==================================================
            # Save Artifacts
            # ==================================================
            logger.info("Saving model artifacts...")
            os.makedirs("models", exist_ok=True)

            joblib.dump(model, MODEL_PATH)
            joblib.dump(preprocessor, SCALER_PATH)

            metadata = {
                "algorithm": "IsolationForest",
                "contamination": 0.01,
                "n_estimators": 200,
                "features": list(df.columns),
                "score_min": score_min,   # consumed by inference/predictor.py
                "score_max": score_max,
            }

            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)

            logger.info(f"Model saved to: {MODEL_PATH}")
            logger.info(f"Preprocessor saved to: {SCALER_PATH}")
            logger.info(f"Metadata saved to: {METADATA_PATH}")

            # ==================================================
            # MLflow Logging
            # ==================================================
            logger.info("Logging artifacts to MLflow...")
            mlflow.log_artifact(METADATA_PATH)
            mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
            mlflow.log_artifact(SCALER_PATH)

            elapsed = time.time() - start_time
            mlflow.log_metric("training_time_seconds", elapsed)

            logger.info("=" * 60)
            logger.info("Fraud Model Training Completed Successfully")
            logger.info(f"Training Time: {elapsed:.2f} seconds")
            logger.info("=" * 60)

    except Exception:
        logger.exception("Training pipeline failed.")
        raise


if __name__ == "__main__":
    train()