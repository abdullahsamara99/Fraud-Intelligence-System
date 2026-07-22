import pandas as pd

from fastapi import APIRouter, HTTPException

from api.schemas import TransactionRequest, FraudPredictionResponse
from inference.pipeline import FraudPipeline
from utils.logger import logger

router = APIRouter()

# --------------------------------------------------
# Load Pipeline Once (heavy: model + SHAP background)
# --------------------------------------------------
logger.info("=" * 60)
logger.info("Initializing Fraud Detection API...")
pipeline = FraudPipeline()
logger.info("Fraud Pipeline loaded successfully.")
logger.info("=" * 60)


@router.get("/")
def home():
    """Health check endpoint."""
    return {"message": "Fraud Intelligence System API", "status": "Running"}


@router.post(
    "/analyze_transaction",           # spec §10 endpoint name
    response_model=FraudPredictionResponse,
)
def analyze_transaction(transaction: TransactionRequest):
    """Analyze a single transaction for fraud."""
    try:
        logger.info(f"Analyzing transaction {transaction.transaction_id}")

        df = pd.DataFrame([transaction.model_dump()])
        result = pipeline.analyze(df)[0]

        # Echo transaction_id back per spec §6 output contract
        result["transaction_id"] = transaction.transaction_id

        logger.info(
            f"Result: {result['prediction']} | {result['risk_level']} | "
            f"{result['recommended_action']}"
        )
        return result

    except Exception as e:
        logger.exception("Fraud analysis request failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Fraud analysis failed: {str(e)}",
        )


# Backwards-compatible alias so the old /analyze path keeps working
@router.post("/analyze", response_model=FraudPredictionResponse)
def analyze(transaction: TransactionRequest):
    return analyze_transaction(transaction)