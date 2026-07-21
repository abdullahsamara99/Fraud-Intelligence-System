import os
import logging
from logging.handlers import RotatingFileHandler

# --------------------------------------------------
# Create logs directory
# --------------------------------------------------

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "fraud_system.log")

# --------------------------------------------------
# Logger
# --------------------------------------------------

logger = logging.getLogger("FraudSystem")
logger.setLevel(logging.INFO)

# Prevent messages from being propagated to the root logger
logger.propagate = False

# Prevent duplicate handlers
if not logger.handlers:

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # ----------------------------------------
    # Console Handler
    # ----------------------------------------

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ----------------------------------------
    # Rotating File Handler
    # ----------------------------------------

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # ----------------------------------------
    # Register Handlers
    # ----------------------------------------

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)