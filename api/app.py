from fastapi import FastAPI

from api.routes import router
from utils.logger import logger


logger.info("=" * 60)
logger.info("Starting Fraud Intelligence System API...")
logger.info("=" * 60)

app = FastAPI(
    title="Fraud Intelligence System",
    version="1.0.0",
    description="AI-powered fraud detection using Isolation Forest and SHAP explanations.",
)

app.include_router(router)

logger.info("API routes registered successfully.")
logger.info("Fraud Intelligence System API is ready.")