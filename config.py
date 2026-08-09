import os

MODEL="gemma4:e4b"  # swap depending on llama variant

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Set to True to enable MLflow logging, False to disable
MLFLOW_ENABLED = False

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://103.231.240.155:11434")