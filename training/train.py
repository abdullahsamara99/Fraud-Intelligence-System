import json
import os
import time

import joblib
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


def train():
    """
    Train the Isolation Forest fraud detection model
    with MLflow experiment tracking.
    """

    start_time = time.time()

    try:

        logger.info("=" * 60)
        logger.info("Starting Fraud Model Training")
        logger.info("=" * 60)

        # ==================================================
        # MLflow
        # ==================================================

        mlflow.set_experiment("Fraud Detection Isolation Forest")

        with mlflow.start_run():

            logger.info("MLflow run started.")

            # ----------------------------------------------
            # Log Parameters
            # ----------------------------------------------

            mlflow.log_param("algorithm", "IsolationForest")
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("contamination", 0.01)
            mlflow.log_param("random_state", 42)

            # ==================================================
            # Load Dataset
            # ==================================================

            logger.info("Loading dataset...")

            df = load_data(
                CUSTOMERS_PATH,
                TRANSACTIONS_PATH,
            )

            logger.info(
                f"Dataset loaded successfully. Shape: {df.shape}"
            )

            logger.info(f"Columns:\n{list(df.columns)}")

            # ==================================================
            # Feature Engineering
            # ==================================================

            logger.info("Running feature engineering...")

            df = create_features(df)

            logger.info(
                f"Feature engineering completed. Shape: {df.shape}"
            )

            # ==================================================
            # Drop Non-Model Columns
            # ==================================================

            logger.info("Dropping unused columns...")

            drop_columns = [
                "transaction_id",
                "customer_id",
                "account_number",
                "timestamp",
                "transaction_date",
                "transaction_time",
                "merchant_name",
                "transaction_country_iso",
            ]

            df = df.drop(columns=drop_columns)

            logger.info(
                f"Model dataframe shape: {df.shape}"
            )

            # ==================================================
            # Feature Types
            # ==================================================

            categorical_columns = [
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

            numeric_columns = [
                col
                for col in df.columns
                if col not in categorical_columns
            ]

            logger.info(
                f"Numeric features: {len(numeric_columns)}"
            )

            logger.info(
                f"Categorical features: {len(categorical_columns)}"
            )

            mlflow.log_metric(
                "num_numeric_features",
                len(numeric_columns),
            )

            mlflow.log_metric(
                "num_categorical_features",
                len(categorical_columns),
            )

            # ==================================================
            # Preprocessing Pipeline
            # ==================================================

            logger.info("Building preprocessing pipeline...")

            numeric_transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                ]
            )

            categorical_transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        numeric_transformer,
                        numeric_columns,
                    ),
                    (
                        "cat",
                        categorical_transformer,
                        categorical_columns,
                    ),
                ]
            )

            logger.info("Applying preprocessing...")

            X = preprocessor.fit_transform(df)

            logger.info(
                f"Preprocessing completed. Feature matrix shape: {X.shape}"
            )

            mlflow.log_metric("num_samples", X.shape[0])
            mlflow.log_metric("num_features", X.shape[1])

            # ==================================================
            # Train Model
            # ==================================================

            logger.info(
                "Training Isolation Forest..."
            )

            model = IsolationForest(
                n_estimators=200,
                contamination=0.01,
                random_state=42,
                n_jobs=-1,
            )

            model.fit(X)

            logger.info(
                "Isolation Forest training completed successfully."
            )

            # ==================================================
            # Save Artifacts
            # ==================================================

            logger.info("Saving model artifacts...")

            os.makedirs("models", exist_ok=True)

            joblib.dump(
                model,
                MODEL_PATH,
            )

            joblib.dump(
                preprocessor,
                SCALER_PATH,
            )

            metadata = {
                "algorithm": "IsolationForest",
                "contamination": 0.01,
                "n_estimators": 200,
                "features": list(df.columns),
            }

            with open(
                METADATA_PATH,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    metadata,
                    f,
                    indent=4,
                )

            logger.info(
                f"Model saved to: {MODEL_PATH}"
            )

            logger.info(
                f"Preprocessor saved to: {SCALER_PATH}"
            )

            logger.info(
                f"Metadata saved to: {METADATA_PATH}"
            )

            # ==================================================
            # MLflow Logging
            # ==================================================

            logger.info("Logging artifacts to MLflow...")

            mlflow.log_artifact(METADATA_PATH)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
            )

            mlflow.log_artifact(SCALER_PATH)

            # ==================================================
            # Training Metrics
            # ==================================================

            elapsed = time.time() - start_time

            mlflow.log_metric(
                "training_time_seconds",
                elapsed,
            )

            logger.info(
                f"Training metadata: {metadata}"
            )

            logger.info("=" * 60)
            logger.info(
                "Fraud Model Training Completed Successfully"
            )
            logger.info(
                f"Training Time: {elapsed:.2f} seconds"
            )
            logger.info("=" * 60)

    except Exception:

        logger.exception(
            "Training pipeline failed."
        )

        raise


if __name__ == "__main__":
    train()