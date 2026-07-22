from typing import List, Optional

from pydantic import BaseModel, Field


# ==========================================================
# Request Schema  (THE MISSING PIECE)
# ==========================================================
class TransactionRequest(BaseModel):
    """
    Incoming transaction payload for POST /analyze_transaction.

    Field names MUST match the columns your feature_engineering /
    preprocess pipeline expects. Optional fields default to None so a
    partial payload still validates and is imputed downstream.
    """

    transaction_id: str = Field(..., examples=["TX100001"])
    customer_id: str = Field(..., examples=["C1001"])
    account_number: Optional[str] = Field(None, examples=["ACC0001"])

    # Timestamps
    timestamp: str = Field(..., examples=["2025-07-15 22:45:00"])
    transaction_date: Optional[str] = Field(None, examples=["2025-07-15"])
    transaction_time: Optional[str] = Field(None, examples=["22:45:00"])

    # Money
    transaction_amount: float = Field(..., ge=0, examples=[9500.0])
    currency: str = Field("USD", examples=["USD"])

    # Geography
    transaction_country: str = Field(..., examples=["United States"])
    transaction_country_iso: Optional[str] = Field(None, examples=["US"])
    transaction_city: str = Field(..., examples=["New York"])
    is_cross_border: int = Field(0, ge=0, le=1)

    # Merchant / channel
    merchant_name: Optional[str] = Field(None, examples=["Apple Store"])
    merchant_category: str = Field(..., examples=["Electronics"])
    channel: str = Field("POS", examples=["POS"])
    card_type: str = Field("Credit", examples=["Credit"])
    transaction_type: str = Field("Purchase", examples=["Purchase"])

    # Customer profile
    home_country: str = Field(..., examples=["United States"])
    home_city: str = Field(..., examples=["New York"])
    avg_monthly_income: float = Field(..., ge=0, examples=[4500.0])
    account_age_days: int = Field(..., ge=0, examples=[850])


# ==========================================================
# Response Schemas
# ==========================================================
class FeatureImpact(BaseModel):
    feature: str
    impact: float
    direction: str  # "Increase Risk" | "Decrease Risk"


class FraudPredictionResponse(BaseModel):
    transaction_id: str            # now echoed back per spec §6
    prediction: str                # "Anomalous" | "Normal"
    anomaly_score: float           # raw Isolation Forest score
    risk_score: float              # normalized 0-1
    risk_level: str                # High | Medium | Low
    recommended_action: str        # Approve | Manual Review | Block | Escalate
    top_features: List[FeatureImpact] = []
    explanation: str