import os
from pathlib import Path
from dotenv import load_dotenv

# =========================
# 路径配置
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

# =========================
# 加载本地环境变量（云端不会用这个）
# =========================
load_dotenv(dotenv_path=ENV_PATH)

# =========================
# API 配置
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DEEPSEEK_BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com"
)

DEEPSEEK_MODEL = os.getenv(
    "DEEPSEEK_MODEL",
    "deepseek-chat"
)

# =========================
# 向量库 & Embedding 配置
# =========================
CHROMA_DIR = os.getenv(
    "CHROMA_DIR",
    str(BASE_DIR / "chroma_db")
)

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-MiniLM-L3-v2"
)

# =========================
# 调试信息（可选）
# =========================
if __name__ == "__main__":
    print("ENV_PATH =", ENV_PATH)
    print("OPENAI_API_KEY exists =", bool(OPENAI_API_KEY))
    print("DEEPSEEK_BASE_URL =", DEEPSEEK_BASE_URL)
    print("DEEPSEEK_MODEL =", DEEPSEEK_MODEL)
    print("EMBEDDING_MODEL_NAME =", EMBEDDING_MODEL_NAME)
