import logging
import os
from datetime import datetime

# Define the log directory path
log_dir = os.path.join(os.getcwd(), "logs")

# Ensure the directory exists
os.makedirs(log_dir, exist_ok=True)

# Define the log file name and path
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

# Configure logging to log both to a file and to the console
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)
