from dotenv import load_dotenv
import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

DEEPSEEK_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-MiniLM-L3-v2"
)

print("ENV_PATH =", ENV_PATH)
print("DEEPSEEK_API_KEY exists =", bool(DEEPSEEK_API_KEY))
print("OPENAI_API_KEY exists =", bool(OPENAI_API_KEY))
print("当前 EMBEDDING_MODEL_NAME =", EMBEDDING_MODEL_NAME)
