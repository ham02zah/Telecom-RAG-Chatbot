import os
from dotenv import load_dotenv


load_dotenv()


DATA_PATH = "data/telecom_faqs.csv"
INDEX_PATH = "models/rag_index.joblib"

REPORTS_DIR = "reports"
VISUALIZATIONS_DIR = "visualizations"

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")