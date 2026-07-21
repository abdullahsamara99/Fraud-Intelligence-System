import joblib
import matplotlib.pyplot as plt

from features.preprocess import load_data
from features.feature_engineering import create_features
from utils.logger import logger


def evaluate():
    """
    Evaluate the trained Isolation Forest model and
    convert anomaly scores into normalized fraud risk scores (0-1).
    """

    try:

        logger.info("=" * 60)
        logger.info("Starting Model Evaluation")
        logger.info("=" * 60)

        # ---------------------------------------------------
        # Load Data
        # ---------------------------------------------------

        logger.info("Loading dataset...")

        df = load_data(
            "data/raw/customers.csv",
            "data/raw/transactions.csv",
        )

        logger.info(
            f"Dataset loaded successfully. Shape: {df.shape}"
        )

        # ---------------------------------------------------
        # Feature Engineering
        # ---------------------------------------------------

        logger.info("Running feature engineering...")

        df = create_features(df)

        logger.info(
            f"Feature engineering completed. Shape: {df.shape}"
        )

        # ---------------------------------------------------
        # Drop Non-model Columns
        # ---------------------------------------------------

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

        # ---------------------------------------------------
        # Load Artifacts
        # ---------------------------------------------------

        logger.info("Loading trained model...")

        preprocessor = joblib.load("models/scaler.pkl")
        model = joblib.load("models/isolation_forest.pkl")

        logger.info(
            "Model and preprocessor loaded successfully."
        )

        X = preprocessor.transform(df)

        logger.info(
            f"Feature matrix shape: {X.shape}"
        )

        # ---------------------------------------------------
        # Prediction
        # ---------------------------------------------------

        logger.info("Calculating anomaly scores...")

        scores = model.decision_function(X)

        predictions = model.predict(X)

        # ---------------------------------------------------
        # Normalize Score (0 → 1)
        # Higher = Higher Risk
        # ---------------------------------------------------

        logger.info("Normalizing anomaly scores...")

        score_min = scores.min()
        score_max = scores.max()

        risk_scores = 1 - (
            (scores - score_min)
            / (score_max - score_min)
        )

        df["anomaly_score"] = scores
        df["risk_score"] = risk_scores
        df["prediction"] = predictions

        logger.info("Prediction completed successfully.")

        # ---------------------------------------------------
        # Statistics
        # ---------------------------------------------------

        logger.info("=" * 60)
        logger.info("Evaluation Summary")
        logger.info("=" * 60)

        print("\nOriginal Isolation Forest Scores\n")
        print(df["anomaly_score"].describe())

        print("\nNormalized Risk Scores (0-1)\n")
        print(df["risk_score"].describe())

        anomaly_count = (predictions == -1).sum()

        print(f"\nDetected Anomalies : {anomaly_count}")

        logger.info(
            f"Detected {anomaly_count} anomalous transactions."
        )

        # ---------------------------------------------------
        # Highest Risk Transactions
        # ---------------------------------------------------

        print("\nTop 10 Highest Risk Transactions\n")

        print(
            df[
                [
                    "transaction_amount",
                    "risk_score",
                    "prediction",
                ]
            ]
            .sort_values(
                "risk_score",
                ascending=False,
            )
            .head(10)
        )

        # ---------------------------------------------------
        # Histogram
        # ---------------------------------------------------

        logger.info("Generating risk score distribution plot...")

        plt.figure(figsize=(10, 5))

        plt.hist(
            risk_scores,
            bins=100,
        )

        plt.title("Fraud Risk Score Distribution")

        plt.xlabel("Risk Score (0 = Safe, 1 = High Risk)")

        plt.ylabel("Transactions")

        plt.tight_layout()

        plt.show()

        logger.info("=" * 60)
        logger.info("Evaluation Completed Successfully")
        logger.info("=" * 60)

    except Exception:

        logger.exception("Evaluation failed.")

        raise


if __name__ == "__main__":
    evaluate()