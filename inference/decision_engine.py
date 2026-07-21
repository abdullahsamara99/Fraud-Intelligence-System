from utils.logger import logger


class DecisionEngine:
    """
    Determines the fraud risk level and recommended action
    based on the normalized fraud risk score.

    Risk Score:
        0.0 -> Lowest Risk
        1.0 -> Highest Risk
    """

    def evaluate(self, prediction: int, risk_score: float):
        """
        Evaluate the model output.

        Args:
            prediction: Isolation Forest prediction
                        (-1 = Anomaly, 1 = Normal)

            risk_score: Normalized fraud risk score (0-1)

        Returns:
            Dictionary containing:
                - risk_level
                - recommended_action
        """

        try:

            logger.info(
                f"Evaluating transaction "
                f"(prediction={prediction}, risk_score={risk_score:.4f})"
            )

            # -------------------------------------
            # High Risk
            # -------------------------------------

            if prediction == -1 or risk_score >= 0.80:

                decision = {
                    "risk_level": "High",
                    "recommended_action": "Block Transaction",
                }

            # -------------------------------------
            # Medium Risk
            # -------------------------------------

            elif risk_score >= 0.50:

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