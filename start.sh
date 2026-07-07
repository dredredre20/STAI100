#!/bin/sh
set -e

# Start the Backend API in the background
python -m uvicorn app.backend_api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in the foreground
exec streamlit run app/streamlit_app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false