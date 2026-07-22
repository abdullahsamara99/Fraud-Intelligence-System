from utils.config import config
from utils.logger import logger


class DecisionEngine:
    """
    Determines the fraud risk level and recommended action
    based on the normalized fraud risk score and model predictions.

    Risk Score:
        0.0 -> Lowest Risk
        1.0 -> Highest Risk
    """

    def __init__(self):
        # Read thresholds directly using the Config class's .get(*keys) method
        try:
            self.high_threshold = config.get("decision_engine", "high_threshold")
            self.medium_threshold = config.get("decision_engine", "medium_threshold")
        except Exception as e:
            logger.warning(
                f"Could not parse thresholds from config ({e}). "
                f"Falling back to default values (0.80 / 0.50)."
            )
            self.high_threshold = 0.80
            self.medium_threshold = 0.50

        logger.info(
            f"DecisionEngine initialized with thresholds: "
            f"High={self.high_threshold}, Medium={self.medium_threshold}"
        )

    def evaluate(self, prediction: int, risk_score: float) -> dict:
        """
        Evaluate the model output using thresholds configured in config.yaml.

        Args:
            prediction: Isolation Forest prediction (-1 = Anomaly, 1 = Normal)
            risk_score: Normalized fraud risk score (0-1)

        Returns:
            Dictionary containing:
                - risk_level ("High", "Medium", "Low")
                - recommended_action ("Block Transaction", "Manual Review", "Approve")
        """
        try:
            logger.info(
                f"Evaluating transaction "
                f"(prediction={prediction}, risk_score={risk_score:.4f})"
            )

            # -------------------------------------
            # High Risk
            # -------------------------------------
            if prediction == -1 or risk_score >= self.high_threshold:
                decision = {
                    "risk_level": "High",
                    "recommended_action": "Block Transaction",
                }

            # -------------------------------------
            # Medium Risk
            # -------------------------------------
            elif risk_score >= self.medium_threshold:
                decision = {
                    "risk_level": "Medium",
                    "recommended_action": "Manual Review",
                }

            # -------------------------------------
            # Low Risk
            # -------------------------------------
            else:
                decision = {
                    "risk_level": "Low",
                    "recommended_action": "Approve",
                }

            logger.info(
                f"Decision completed: "
                f"Risk={decision['risk_level']}, "
                f"Action={decision['recommended_action']}"
            )

            return decision

        except Exception:
            logger.exception("Decision engine failed.")
            raise


# ==========================================================
# Demo
# ==========================================================

if __name__ == "__main__":

    engine = DecisionEngine()

    samples = [
        (-1, 0.95),
        (1, 0.85),
        (1, 0.72),
        (1, 0.55),
        (1, 0.41),
        (1, 0.18),
        (1, 0.03),
    ]

    print("=" * 60)
    print("Decision Engine Demo")
    print("=" * 60)

    for prediction, risk_score in samples:

        result = engine.evaluate(
            prediction,
            risk_score,
        )

        print(
            f"Prediction={prediction}, "
            f"Risk Score={risk_score:.2f} "
            f"-> {result['risk_level']} | "
            f"{result['recommended_action']}"
        )