import pandas as pd

from features.preprocess import load_data
from features.feature_engineering import create_features
from inference.predictor import FraudPredictor
from inference.decision_engine import DecisionEngine
from inference.explain import FraudExplainer
from utils.logger import logger


class FraudPipeline:
    """
    Complete & Production-Ready Fraud Detection Pipeline.

    Steps:
        1. Single Feature Engineering Step
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
        # Pre-build SHAP Explainer ONCE during server startup
        # ----------------------------------------------------
        try:
            logger.info("Pre-loading background data for SHAP Explainer...")
            raw_bg = load_data("data/raw/customers.csv", "data/raw/transactions.csv")
            engineered_bg = create_features(raw_bg)

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
            model_bg = engineered_bg.drop(columns=drop_columns, errors="ignore")

            self.explainer.build_explainer(model_bg)
            logger.info("SHAP explainer pre-built successfully.")
        except Exception as e:
            logger.error(f"Failed to pre-build SHAP explainer on initialization: {e}")

        logger.info("Fraud Detection Pipeline initialized successfully.")
        logger.info("=" * 60)

    def analyze(self, df: pd.DataFrame):

        try:
            # تصفير الـ Index لضمان سلامة الـ iloc في الـ Loops
            df = df.reset_index(drop=True)

            logger.info(
                f"Starting analysis for {len(df)} transaction(s)."
            )

            # ----------------------------------------
            # Feature Engineering
            # ----------------------------------------

            logger.info("Running feature engineering...")

            feature_df = create_features(df)

            logger.info(
                f"Feature engineering completed. Shape: {feature_df.shape}"
            )

            # ----------------------------------------
            # Prepare Model Input
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

            logger.info(
                f"Prepared model dataframe with shape {model_df.shape}"
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

            logger.info(
                "Generating risk decisions and explanations..."
            )

            # استخدام enumerate لضمان الحفاظ على التسلسل الرقمي للـ iloc
            for idx, (_, row) in enumerate(prediction_df.iterrows()):

                decision = self.decision_engine.evaluate(
                    prediction=row["prediction"],
                    risk_score=row["risk_score"],
                )

                # استخراج أوزان الميزات القادمة من SHAP
                top_features = self.explainer.explain(
                    transaction=model_df.iloc[[idx]],
                    feature_names=model_df.columns.tolist(),
                )

                # توليد التقرير المالي الاحترافي عبر الـ LLM Layer
                explanation_summary = self.explainer.generate_llm_summary(
                    top_features=top_features,
                    risk_level=decision["risk_level"],
                    action=decision["recommended_action"],
                )

                results.append(
                    {
                        "prediction": (
                            "Anomalous"
                            if row["prediction"] == -1
                            else "Normal"
                        ),

                        # Raw Isolation Forest score
                        "anomaly_score": round(
                            float(row["anomaly_score"]),
                            4,
                        ),

                        # Normalized score (0-1)
                        "risk_score": round(
                            float(row["risk_score"]),
                            4,
                        ),

                        "risk_level": decision["risk_level"],

                        "recommended_action": decision[
                            "recommended_action"
                        ],

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