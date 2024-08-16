import sys
import logging
import os
from datetime import datetime

# Setup logging
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(message)s",
    level=logging.INFO,
)

def error_message_detail(error, error_detail: sys):
    """
    Function to capture error details including script name, line number, and error message.

    Parameters:
    - error: Exception object containing the error message.
    - error_detail: sys module to capture error details using sys.exc_info().

    Returns:
    - error_message: Formatted error message string.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    return error_message

class CustomException(Exception):
    """
    Custom exception class for handling exceptions and logging detailed error messages.

    Attributes:
    - error_message_detail: Formatted error message with script name, line number, and error message.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message_detail = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message_detail
