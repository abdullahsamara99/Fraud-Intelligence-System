import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from features.feature_engineering import create_features
from features.preprocess import load_data
from utils.logger import logger


def analyze_distribution_shift(
    baseline_scores: np.ndarray,
    new_scores: np.ndarray,
    ks_threshold: float = 0.05,
    alpha: float = 0.05,
) -> dict:
    """
    Compares the risk score distribution between baseline and new batch data.
    Uses KS-Statistic threshold (Effect Size) as primary alert trigger to avoid
    large-sample size p-value artifacts.

    Args:
        baseline_scores: Array of risk scores from baseline/historical window.
        new_scores: Array of risk scores from recent/live window.
        ks_threshold: Minimum KS statistic threshold for practical drift (default 0.05).
        alpha: Statistical significance threshold (default 0.05).

    Returns:
        dict containing ks_statistic, p_value, shift_detected, and summary message.
    """
    ks_stat, p_value = ks_2samp(baseline_scores, new_scores)

    # Trigger shift ONLY if p-value is significant AND effect size exceeds threshold (D >= 0.05)
    is_shift_detected = bool((p_value < alpha) and (ks_stat >= ks_threshold))

    result = {
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "shift_detected": is_shift_detected,
        "message": (
            f"Practical distribution shift detected! (D={ks_stat:.4f} >= {ks_threshold})"
            if is_shift_detected
            else f"No practical distribution shift (D={ks_stat:.4f} < {ks_threshold})."
        ),
    }

    if is_shift_detected:
        logger.warning(
            f"Distribution Shift Alert: Practical drift detected. KS Stat={ks_stat:.4f}, p-value={p_value:.4e}"
        )
    else:
        logger.info(
            f"Distribution Shift Check Passed: No practical drift. KS Stat={ks_stat:.4f}, p-value={p_value:.4e}"
        )

    return result


def evaluate():
    """
    Evaluate the trained Isolation Forest model,
    convert anomaly scores into normalized fraud risk scores (0-1),
    and check for distribution shifts across transaction batches.
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

        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

        # ---------------------------------------------------
        # Feature Engineering
        # ---------------------------------------------------

        logger.info("Running feature engineering...")

        df = create_features(df)

        logger.info(f"Feature engineering completed. Shape: {df.shape}")

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

        logger.info(f"Model dataframe shape: {df.shape}")

        # ---------------------------------------------------
        # Load Artifacts
        # ---------------------------------------------------

        logger.info("Loading trained model...")

        preprocessor = joblib.load("models/scaler.pkl")
        model = joblib.load("models/isolation_forest.pkl")

        logger.info("Model and preprocessor loaded successfully.")

        X = preprocessor.transform(df)

        logger.info(f"Feature matrix shape: {X.shape}")

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

        risk_scores = 1 - ((scores - score_min) / (score_max - score_min))

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

        logger.info(f"Detected {anomaly_count} anomalous transactions.")

        # ---------------------------------------------------
        # Distribution Shift Analysis (KS-Test)
        # ---------------------------------------------------

        logger.info("Analyzing distribution shift between historical & recent batches...")

        mid_point = len(risk_scores) // 2
        baseline_batch = risk_scores[:mid_point]
        recent_batch = risk_scores[mid_point:]

        shift_results = analyze_distribution_shift(baseline_batch, recent_batch)

        print("\nDistribution Shift Analysis (KS-Test)\n")
        print(f"  • KS Statistic : {shift_results['ks_statistic']:.4f}")
        print(f"  • p-value      : {shift_results['p_value']:.4e}")
        print(
            f"  • Shift Status : {'DETECTED ⚠️' if shift_results['shift_detected'] else 'NO PRACTICAL SHIFT DETECTED ✅'}"
        )

        # ---------------------------------------------------
        # Highest Risk Transactions
        # ---------------------------------------------------

        print("\nTop 10 Highest Risk Transactions\n")

        print(
            df[["transaction_amount", "risk_score", "prediction"]]
            .sort_values("risk_score", ascending=False)
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