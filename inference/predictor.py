import json
import os

import joblib
import numpy as np
import pandas as pd

from utils.logger import logger

DROP_COLUMNS = [
    "transaction_id", "customer_id", "account_number", "timestamp",
    "transaction_date", "transaction_time", "merchant_name",
    "transaction_country_iso", "is_fraud",   # is_fraud never reaches the model
]


class FraudPredictor:
    """
    Loads the trained Isolation Forest and scores transactions.

    Normalization bounds (score_min/score_max) are read from
    models/metadata.json so single-transaction risk scores are stable and
    batch-independent, instead of relying on hardcoded magic numbers.
    """

    def __init__(
        self,
        model_path="models/isolation_forest.pkl",
        preprocessor_path="models/scaler.pkl",
        metadata_path="models/metadata.json",
    ):
        logger.info("Loading fraud prediction model...")
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

        # Load calibrated score range; fall back to sane defaults
        self.score_min, self.score_max = -0.35, 0.15
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.score_min = float(meta.get("score_min", self.score_min))
                self.score_max = float(meta.get("score_max", self.score_max))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not read score range from metadata: {e}")

        if self.score_max <= self.score_min:  # guard against bad calibration
            self.score_min, self.score_max = -0.35, 0.15

        logger.info(
            f"Predictor ready. score_min={self.score_min}, "
            f"score_max={self.score_max}"
        )

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        try:
            model_df = feature_df.drop(columns=DROP_COLUMNS, errors="ignore")
            X = self.preprocessor.transform(model_df)

            predictions = self.model.predict(X)
            anomaly_scores = self.model.decision_function(X)

            clipped = np.clip(anomaly_scores, self.score_min, self.score_max)
            risk_scores = 1.0 - (
                (clipped - self.score_min) / (self.score_max - self.score_min)
            )

            result = model_df.copy()
            result["anomaly_score"] = np.round(anomaly_scores, 4)
            result["risk_score"] = np.round(risk_scores, 4)
            result["prediction"] = predictions
            return result

        except Exception:
            logger.exception("Fraud prediction failed.")
            raise