import logging 
import os 
from datetime import datetime 

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%M_%S')}.log"

file_path = os.path.join(os.getcwd(), "log", LOG_FILE)
os.makedirs(os.path.dirname(file_path), exist_ok= True)

LOG_FILE_PATH =  os.path.join(file_path)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

