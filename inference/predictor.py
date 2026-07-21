import joblib
import numpy as np
import pandas as pd

from features.feature_engineering import create_features
from features.preprocess import load_data
from utils.logger import logger


class FraudPredictor:
    """
    Loads the trained Isolation Forest model and performs fraud prediction.

    Input:
        Engineered dataframe.

    Returns:
        - prediction (-1 = anomaly, 1 = normal)
        - anomaly_score (raw Isolation Forest score)
        - risk_score (normalized 0-1, higher = higher fraud risk)
    """

    def __init__(
        self,
        model_path="models/isolation_forest.pkl",
        preprocessor_path="models/scaler.pkl",
        score_min=-0.35,
        score_max=0.15,
    ):
        logger.info("Loading fraud prediction model...")

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.score_min = score_min
        self.score_max = score_max

        logger.info("Fraud prediction model loaded successfully.")

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info(
                f"Starting prediction for {len(feature_df)} transaction(s)."
            )

            # ----------------------------------------
            # Remove columns not used during training
            # ----------------------------------------
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

            model_df = feature_df.drop(columns=drop_columns, errors="ignore")

            logger.info(f"Model input shape: {model_df.shape}")

            # ----------------------------------------
            # Apply preprocessing
            # ----------------------------------------
            logger.info("Applying preprocessing pipeline...")

            X = self.preprocessor.transform(model_df)

            logger.info(
                f"Preprocessing completed. Feature matrix shape: {X.shape}"
            )

            # ----------------------------------------
            # Predict
            # ----------------------------------------
            logger.info("Running Isolation Forest prediction...")

            predictions = self.model.predict(X)
            anomaly_scores = self.model.decision_function(X)

            logger.info("Prediction completed successfully.")

            # ----------------------------------------
            # Absolute Normalization (Batch-Independent)
            # ----------------------------------------
            clipped_scores = np.clip(anomaly_scores, self.score_min, self.score_max)
            risk_scores = 1.0 - (
                (clipped_scores - self.score_min)
                / (self.score_max - self.score_min)
            )

            # ----------------------------------------
            # Build Results
            # ----------------------------------------
            result = model_df.copy()

            result["anomaly_score"] = np.round(anomaly_scores, 4)
            result["risk_score"] = np.round(risk_scores, 4)
            result["prediction"] = predictions

            logger.info("Prediction results prepared.")

            return result

        except Exception:
            logger.exception("Fraud prediction failed.")
            raise


# ----------------------------------------------------
# Standalone Test
# ----------------------------------------------------
if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("Testing Fraud Predictor")
        logger.info("=" * 60)

        logger.info("Loading sample dataset...")

        df = load_data(
            "data/raw/customers.csv",
            "data/raw/transactions.csv",
        )

        logger.info("Creating features...")

        feature_df = create_features(df.head(5))

        predictor = FraudPredictor()

        logger.info("Running prediction...")

        result = predictor.predict(feature_df)

        print("\nPrediction Results\n")

        print(
            result[
                [
                    "transaction_amount",
                    "anomaly_score",
                    "risk_score",
                    "prediction",
                ]
            ]
        )

        logger.info("Standalone prediction completed successfully.")

    except Exception:
        logger.exception("Standalone prediction failed.")