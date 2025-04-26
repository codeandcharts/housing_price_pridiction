import logging
import os
from datetime import datetime
from typing import Optional

# Define the log file name with a timestamp
LOG_FILE_NAME = f"app_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Define the logs directory
LOGS_DIR = os.path.join(os.getcwd(), "logs", LOG_FILE_NAME)
# Create the logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)
# Define the full log file path
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s ] - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
