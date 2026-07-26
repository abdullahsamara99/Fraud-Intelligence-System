from utils.config import config
from utils.logger import logger


class DecisionEngine:
    """
    Maps the normalized risk score into a risk level and operational action.

    With the symmetric-around-zero calibration, risk_score monotonically
    encodes anomaly severity (0.5 = the model's inlier/outlier boundary),
    so tiers are driven purely by risk bands. All four actions are reachable:
    Approve / Flag for Review / Block / Escalate.

        risk >= escalate_threshold (0.77)  -> Critical -> Escalate to Fraud Team
        risk >= high_threshold     (0.68)  -> High     -> Block Transaction
        risk >= medium_threshold   (0.50)  -> Medium   -> Flag for Review
        otherwise                          -> Low      -> Approve
    """

    def __init__(self):
        try:
            self.escalate_threshold = config.get("decision_engine", "escalate_threshold")
            self.high_threshold = config.get("decision_engine", "high_threshold")
            self.medium_threshold = config.get("decision_engine", "medium_threshold")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Using default thresholds ({e}).")
            self.escalate_threshold = 0.77
            self.high_threshold = 0.68
            self.medium_threshold = 0.50

    def evaluate(self, prediction: int, risk_score: float) -> dict:
        if risk_score >= self.escalate_threshold:
            return {"risk_level": "Critical",
                    "recommended_action": "Escalate to Fraud Team"}

        if risk_score >= self.high_threshold:
            return {"risk_level": "High",
                    "recommended_action": "Block Transaction"}

        if risk_score >= self.medium_threshold:
            return {"risk_level": "Medium",
                    "recommended_action": "Flag for Review"}

        # Safety floor: a model-flagged anomaly should never be auto-approved.
        if prediction == -1:
            return {"risk_level": "Medium",
                    "recommended_action": "Flag for Review"}

        return {"risk_level": "Low", "recommended_action": "Approve"}