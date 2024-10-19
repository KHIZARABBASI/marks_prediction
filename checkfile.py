from src.exception import CustomException
from src.logger import logging
import sys
try:
    a = 1/0
    
except Exception as e:
    raise CustomException(e, sys)

# logging.info("thus the error not show")

