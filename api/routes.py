import pandas as pd

from fastapi import APIRouter, HTTPException

from api.schemas import TransactionRequest
from inference.pipeline import FraudPipeline
from utils.logger import logger


router = APIRouter()


# --------------------------------------------------
# Load Pipeline Once
# --------------------------------------------------

logger.info("=" * 60)
logger.info("Initializing Fraud Detection API...")
logger.info("Loading Fraud Pipeline...")

pipeline = FraudPipeline()

logger.info("Fraud Pipeline loaded successfully.")
logger.info("=" * 60)


# --------------------------------------------------
# Health Endpoint
# --------------------------------------------------

@router.get("/")
def home():
    """
    Health check endpoint.
    """

    logger.info("Health check requested.")

    return {
        "message": "Fraud Intelligence System API",
        "status": "Running",
    }


# --------------------------------------------------
# Analyze Transaction
# --------------------------------------------------

@router.post("/analyze")
def analyze(transaction: TransactionRequest):
    """
    Analyze a single transaction for fraud.
    """

    try:

        logger.info("=" * 60)
        logger.info("Received fraud analysis request.")

        logger.info(
            f"Transaction ID: {transaction.transaction_id}"
        )

        logger.info(
            f"Customer ID: {transaction.customer_id}"
        )

        # ----------------------------------------
        # Convert request to DataFrame
        # ----------------------------------------

        logger.info("Creating DataFrame from request...")

        df = pd.DataFrame(
            [transaction.model_dump()]
        )

        logger.info(
            f"Input dataframe shape: {df.shape}"
        )

        # ----------------------------------------
        # Run Pipeline
        # ----------------------------------------

        logger.info("Running fraud detection pipeline...")

        result = pipeline.analyze(df)

        logger.info("Fraud analysis completed successfully.")

        logger.info(
            f"Prediction: {result[0]['prediction']}"
        )

        logger.info(
            f"Risk Level: {result[0]['risk_level']}"
        )

        logger.info(
            f"Recommended Action: {result[0]['recommended_action']}"
        )

        logger.info("=" * 60)

        return result[0]

    except Exception as e:

        logger.exception("Fraud analysis request failed.")

        raise HTTPException(
            status_code=500,
            detail=f"Fraud analysis failed: {str(e)}",
        )