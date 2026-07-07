# Running the Webapp

You can run this either locally or via Docker.

---

## Option A — Local 

Open 2 terminal tabs. Run these in order.

### 1. Backend

```bash
uvicorn app.backend_api:app --reload --port 8000
```

### 2. Frontend

```bash
streamlit run app/streamlit_app.py
```

Make sure Ollama is running too (`ollama list` to check).

MLflow must be started before the backend, or the backend will fail to start.

---

## Option B — Docker

### Prerequisites

- Docker Desktop installed and running (check for the whale icon in your
  menu bar — wait until it's fully started before building)
- Ollama running locally on your host machine (`ollama list` to check)

### 1. Build the image

```bash
docker build -t stai100-app .
```

### 2. Run the container

```bash
docker run -p 8000:8000 -p 8501:8501 stai100-app
```

### 3. Open the app

- Frontend: http://localhost:8501
- Backend health check: http://localhost:8000/health