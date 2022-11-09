import os
from dotenv import load_dotenv
from pathlib import Path

from utils import createLogHandler

logger = createLogHandler(__name__, 'logs.log')

load_dotenv()
API_KEY = os.getenv('FB_API_KEY')
DATABASE_URL = os.getenv('FB_DATABASE_URL')
STORAGE_URL = os.getenv('FB_STORAGE_URL')
HF_TOKEN = os.getenv('HF_TOKEN')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

TMPSTORAGE = f"{os.getcwd()}/tempimages"
logger.info(f"Temp storage {TMPSTORAGE}")
Path(TMPSTORAGE).mkdir(parents=True, exist_ok=True)
os.environ["TMPSTORAGE"] = TMPSTORAGE