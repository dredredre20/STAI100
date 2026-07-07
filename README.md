# Running the Webapp

Open 3 terminal tabs. Run these in order.

## 1. MLflow 

```bash
mlflow server --host 0.0.0.0 --port 5001
```

Leave this running. View it at: http://localhost:5001

## 2. Backend

```bash
uvicorn app.backend_api:app --reload --port 8000
```

## 3. Frontend

```bash
streamlit run app/streamlit_app.py
```

---

Make sure Ollama is running too (`ollama list` to check).

MLflow must be started before the backend, or the backend will fail to start.