import logging
import os
from src.config import LOGS_DIR

def setup_logging():
    """Set up the root logger to output to console and a file."""
    # log_file = os.path.join(LOGS_DIR, "pipeline.log")

    # Ensure the logs directory exists
    os.makedirs(LOGS_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        handlers=[
            # logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    return logger

logger = setup_logging()
