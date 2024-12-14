import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploaded_pdfs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
