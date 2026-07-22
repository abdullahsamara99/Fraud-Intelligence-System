import pandas as pd

from features.preprocess import load_data
from features.feature_engineering import create_features
from inference.predictor import FraudPredictor
from inference.decision_engine import DecisionEngine
from inference.explain import FraudExplainer
from utils.logger import logger


# Columns dropped before the model / explainer. `is_fraud` is a hidden
# evaluation label and must never enter the feature space.
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

# SHAP background size. Kept small: KernelExplainer cost scales with the
# background, and a random sample is statistically representative even
# for very large datasets (e.g. 3.5M rows).
SHAP_BACKGROUND_SIZE = 5000


class FraudPipeline:
    """
    Complete & Production-Ready Fraud Detection Pipeline.

    Steps:
        1. Feature Engineering
        2. Fraud Prediction & Absolute Risk Normalization
        3. Risk Evaluation via Business Decision Engine
        4. Pre-built SHAP Explanation + LLM Business Summary
    """

    def __init__(self):

        logger.info("=" * 60)
        logger.info("Initializing Fraud Detection Pipeline...")

        self.predictor = FraudPredictor()
        self.decision_engine = DecisionEngine()
        self.explainer = FraudExplainer()

        # ----------------------------------------------------
        # Pre-build SHAP Explainer ONCE during server startup,
        # on a SAMPLED background so startup stays fast on large data.
        # ----------------------------------------------------
        try:
            logger.info("Pre-loading background data for SHAP Explainer...")
            raw_bg = load_data("data/raw/customers.csv", "data/raw/transactions.csv")

            # Sample BEFORE feature engineering so we only engineer a small set.
            if len(raw_bg) > SHAP_BACKGROUND_SIZE:
                logger.info(
                    f"Sampling {SHAP_BACKGROUND_SIZE} of {len(raw_bg)} rows "
                    f"for SHAP background."
                )
                raw_bg = raw_bg.sample(
                    n=SHAP_BACKGROUND_SIZE,
                    random_state=42,
                ).reset_index(drop=True)

            engineered_bg = create_features(raw_bg)
            model_bg = engineered_bg.drop(columns=DROP_COLUMNS, errors="ignore")

            self.explainer.build_explainer(model_bg)
            logger.info("SHAP explainer pre-built successfully.")
        except Exception as e:
            logger.error(f"Failed to pre-build SHAP explainer on initialization: {e}")

        logger.info("Fraud Detection Pipeline initialized successfully.")
        logger.info("=" * 60)

    def analyze(self, df: pd.DataFrame):

        try:
            # Reset index to keep iloc safe inside the loop.
            df = df.reset_index(drop=True)

            logger.info(f"Starting analysis for {len(df)} transaction(s).")

            # ----------------------------------------
            # Feature Engineering
            # ----------------------------------------
            logger.info("Running feature engineering...")
            feature_df = create_features(df)
            logger.info(f"Feature engineering completed. Shape: {feature_df.shape}")

            # ----------------------------------------
            # Prepare Model Input
            # ----------------------------------------
            model_df = feature_df.drop(columns=DROP_COLUMNS, errors="ignore")
            logger.info(f"Prepared model dataframe with shape {model_df.shape}")

            # Preserve transaction_id (if provided) to echo back per spec §6.
            txn_ids = (
                feature_df["transaction_id"].tolist()
                if "transaction_id" in feature_df.columns
                else [None] * len(feature_df)
            )

            # ----------------------------------------
            # Prediction
            # ----------------------------------------
            logger.info("Running fraud prediction...")
            prediction_df = self.predictor.predict(feature_df)
            logger.info("Prediction completed successfully.")

            results = []

            # ----------------------------------------
            # Decision + Explanation Loop
            # ----------------------------------------
            logger.info("Generating risk decisions and explanations...")

            for idx, (_, row) in enumerate(prediction_df.iterrows()):

                decision = self.decision_engine.evaluate(
                    prediction=row["prediction"],
                    risk_score=row["risk_score"],
                )

                top_features = self.explainer.explain(
                    transaction=model_df.iloc[[idx]],
                    feature_names=model_df.columns.tolist(),
                )

                explanation_summary = self.explainer.generate_llm_summary(
                    top_features=top_features,
                    risk_level=decision["risk_level"],
                    action=decision["recommended_action"],
                )

                results.append(
                    {
                        "transaction_id": txn_ids[idx],
                        "prediction": (
                            "Anomalous" if row["prediction"] == -1 else "Normal"
                        ),
                        "anomaly_score": round(float(row["anomaly_score"]), 4),
                        "risk_score": round(float(row["risk_score"]), 4),
                        "risk_level": decision["risk_level"],
                        "recommended_action": decision["recommended_action"],
                        "top_features": top_features,
                        "explanation": explanation_summary,
                    }
                )

            logger.info(
                f"Pipeline completed successfully for {len(results)} transaction(s)."
            )
            return results

        except Exception:
            logger.exception("Fraud pipeline failed.")
            raise


# ==========================================================
# Standalone Test
# ==========================================================

if __name__ == "__main__":

    print("Loading sample data...")
    df = load_data(
        "data/raw/customers.csv",
        "data/raw/transactions.csv",
    )
    print(f"Loaded {len(df)} transactions")

    pipeline = FraudPipeline()

    print("Running pipeline...\n")
    results = pipeline.analyze(df.head(3))

    print("\nResults\n")
    for i, result in enumerate(results, start=1):
        print("=" * 60)
        print(f"Transaction {i}")
        print("=" * 60)
        for key, value in result.items():
            print(f"{key}:")
            print(value)
            print()