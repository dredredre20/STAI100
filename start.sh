#!/bin/sh
set -e

# Start MLflow server
mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db &

python -m uvicorn app.backend_api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit
exec streamlit run app/streamlit_app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false