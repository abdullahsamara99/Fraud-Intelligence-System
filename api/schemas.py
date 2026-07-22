from pydantic import BaseModel

class FraudPredictionResponse(BaseModel):
    prediction: str            # e.g., "Anomalous" or "Normal"
    anomaly_score: float       # raw Isolation Forest score required by spec
    risk_score: float          # normalized (0-1) score
    risk_level: str            # High, Medium, Low
    recommended_action: str    # Block Transaction, Manual Review, Approve
    top_features: list = []    # Extended XAI feature (defaults to empty list if unparsed)
    explanation: str