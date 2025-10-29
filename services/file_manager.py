
import os
from config import UPLOAD_DIR, RESULT_DIR

def save_upload(filename: str) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    return os.path.join(UPLOAD_DIR, filename)

def result_path(filename: str) -> str:
    os.makedirs(RESULT_DIR, exist_ok=True)
    return os.path.join(RESULT_DIR, filename)
