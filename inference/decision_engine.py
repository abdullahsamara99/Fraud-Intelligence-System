from utils.config import config
from utils.logger import logger


class DecisionEngine:
    """
    Maps model output -> risk level -> operational action.

    Actions (spec §5.4): Approve, Manual Review (Flag), Block, Escalate.
    """

    def __init__(self):
        try:
            self.escalate_threshold = config.get("decision_engine", "escalate_threshold")
            self.high_threshold = config.get("decision_engine", "high_threshold")
            self.medium_threshold = config.get("decision_engine", "medium_threshold")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Using default thresholds ({e}).")
            self.escalate_threshold = 0.90
            self.high_threshold = 0.80
            self.medium_threshold = 0.50

    def evaluate(self, prediction: int, risk_score: float) -> dict:
        # Highest tier: model flagged anomaly AND very high score -> humans
        if prediction == -1 and risk_score >= self.escalate_threshold:
            return {"risk_level": "Critical",
                    "recommended_action": "Escalate to Fraud Team"}

        if prediction == -1 or risk_score >= self.high_threshold:
            return {"risk_level": "High",
                    "recommended_action": "Block Transaction"}

        if risk_score >= self.medium_threshold:
            return {"risk_level": "Medium",
                    "recommended_action": "Flag for Review"}

        return {"risk_level": "Low", "recommended_action": "Approve"}