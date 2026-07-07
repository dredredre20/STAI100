import os

MODEL = "llama3.2:3b "  # swap depending on llama variant

# Ollama's default local server
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")