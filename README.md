## Agent8 - Project Summary

Agent 8 is a career-readiness web application that ingests job postings, processes resumes, runs model-driven matching and generation tasks (resume/cover letters), and exposes a backend API and a Streamlit frontend for interaction. The project includes tooling for evaluation, guardrails for validating model outputs, and an agentic workflow layer powered by a ReAct agent.

## Deployed Application (Proxmox)

If you want to access the live hosted environment directly without running it locally:

- Web App: http://103.231.240.130:2151

## Quick Start (Local Deployment)

- Start MLflow (if you use it for tracking):

```bash
mlflow server --host 0.0.0.0 --port 5001
```

- Start the backend API (development):

```bash
uvicorn app.backend_api:app --reload --port 8000
```

- Start the frontend (Streamlit):

```bash
streamlit run app/streamlit_app.py
```

- Or build and run the Docker image:

```bash
docker build -t stai100-app .
docker run -p 8000:8000 -p 8501:8501 -p 5001:5001 stai100-app
```

Frontend: http://localhost:8501 — Backend health: http://localhost:8000/health

## High-level Architecture & Data Flow

1. Data ingestion: job postings are collected/ingested by
   [ds_integration/ingest_job_postings.py](ds_integration/ingest_job_postings.py#L1)
   and stored in the local Chroma DB under [ds_integration/chroma_db](ds_integration/chroma_db/).

2. Resume processing: resumes are parsed and normalized by the
   `resume_processing` pipeline. Key modules:
   - [resume_processing/pipeline.py](resume_processing/pipeline.py#L1) — orchestrates the extraction and validation flow
   - [resume_processing/field_extraction.py](resume_processing/field_extraction.py#L1) — extracts structured resume fields
   - [resume_processing/disambiguation.py](resume_processing/disambiguation.py#L1) — resolves ambiguous fields
   - [resume_processing/validate_output.py](resume_processing/validate_output.py#L1) — validates output schema

3. Matching & prediction: `ds_integration/job_fit_prediction.py` and
   `ds_integration/job_search.py` perform candidate-job matching, ranking, and skill-gap analysis.

4. Agentic reasoning & orchestration: the `react/` package contains the
   ReAct agent in [react/react_agent.py](react/react_agent.py#L1), which follows a
   thought → action → observation loop to choose tools, reason over intermediate
   results, and coordinate multi-step workflows. The agent uses the following tools
   during execution:
   - `get_user_profile`: returns the most recently saved resume profile for the active session.
   - `update_skills`: adds new skills and certifications to the saved profile.
   - `get_top_matches`: ranks stored job postings against the user's resume and returns the best-fit roles.
   - `get_skill_gap`: compares a user-specified company/title against the resume and highlights missing skills.
   - `search_posting`: semantically searches stored job postings for a fuzzy query.
   - `generate_cover_letter`: creates a customized cover letter and DOCX output; only for explicit cover-letter requests.
   - `generate_targeted_resume`: creates a targeted resume DOCX; only for explicit resume-generation requests.

   This allows the agent to retrieve user profile context, search and rank jobs,
   analyze skill gaps, persist updates, and generate tailored resume/cover-letter
   content as part of a single reasoning flow.

5. Resume & cover letter generation: the `resume_cover_generation/` package
   contains generation helpers for tailored resumes and cover letters:
   - [resume_cover_generation/resume_generation.py](resume_cover_generation/resume_generation.py#L1)
   - [resume_cover_generation/cover_letter_generation.py](resume_cover_generation/cover_letter_generation.py#L1)

6. LLM interactions: `llm_utils.py` (root) and modules under `llmops/`
   provide standardized LLM call helpers and logging. See
   [llm_utils.py](llm_utils.py#L1) and [llmops/llmops_logger.py](llmops/llmops_logger.py#L1).

7. Guardrails & safety: modules under [guardrails](guardrails/) validate
   and sanitize model outputs (format checks, prompt safety, and PII redaction):
   - [guardrails/format_verification.py](guardrails/format_verification.py#L1)
   - [guardrails/input_guardrail.py](guardrails/input_guardrail.py#L1)
   - [guardrails/pii_redaction.py](guardrails/pii_redaction.py#L1)

8. Memory & persistence: the `memory/` package handles conversational and
   DB-backed memory via [memory/conversation_memory.py](memory/conversation_memory.py#L1)
   and [memory/db_memory.py](memory/db_memory.py#L1).

9. API & UI: The backend API is in [app/backend_api.py](app/backend_api.py#L1)
   and the Streamlit frontend in [app/streamlit_app.py](app/streamlit_app.py#L1).


## Top-level Files and Directories

- [config.py](config.py#L1) — central configuration (ports, external service endpoints, feature flags).
- [start.sh](start.sh#L1) — helper script to start services locally (if present).
- [requirements.txt](requirements.txt#L1) — Python dependencies.
- [Dockerfile](Dockerfile#L1) — image build for container deployment.

- `app/` — application entrypoints
  - [app/backend_api.py](app/backend_api.py#L1) — FastAPI backend exposing endpoints
  - [app/streamlit_app.py](app/streamlit_app.py#L1) — Streamlit UI

- `ds_integration/` — data ingestion and model integration
  - [ds_integration/ingest_job_postings.py](ds_integration/ingest_job_postings.py#L1)
  - [ds_integration/job_fit_prediction.py](ds_integration/job_fit_prediction.py#L1)
  - [ds_integration/job_search.py](ds_integration/job_search.py#L1)
  - [ds_integration/skill_gap.py](ds_integration/skill_gap.py#L1)

- `resume_processing/` — resume parsing and normalization
  - [resume_processing/pipeline.py](resume_processing/pipeline.py#L1)
  - [resume_processing/field_extraction.py](resume_processing/field_extraction.py#L1)
  - [resume_processing/disambiguation.py](resume_processing/disambiguation.py#L1)
  - [resume_processing/validate_output.py](resume_processing/validate_output.py#L1)
  - [resume_processing/resume_schema.py](resume_processing/resume_schema.py#L1)

- `resume_cover_generation/` — generation helpers for resumes and cover letters
  - [resume_cover_generation/resume_generation.py](resume_cover_generation/resume_generation.py#L1)
  - [resume_cover_generation/cover_letter_generation.py](resume_cover_generation/cover_letter_generation.py#L1)

- `react/` — ReAct agent implementation
  - [react/react_agent.py](react/react_agent.py#L1) — executes the ReAct reasoning pattern:
    thought → action → observation → next step

- `guardrails/` — verification & redaction utilities

- `memory/` — memory and persistence utilities

- `llmops/` — LLM operation helpers and logging

- `evaluation/` — tests and metrics
  - Unit, trajectory, and end-to-end test suites under `evaluation/`

## How the pieces work together (example flow)

1. A user submits a resume via the frontend (Streamlit) or through the API.
2. The backend forwards the resume to the `resume_processing.pipeline`, which
   extracts structured fields and validates them against `resume_schema`.
3. Processed resume data is stored or used to query the job index created by
   `ds_integration` (Chroma DB). Matching logic in `job_search.py` and
   `job_fit_prediction.py` returns ranked job recommendations.
4. If the user asks for resume/cover-letter generation, the app uses the
   generation modules in `resume_cover_generation/` to produce personalized output.
5. The ReAct agent can orchestrate the higher-level workflow, selecting tools,
   reasoning over intermediate results, and continuing until a final answer or
   generated artifact is ready.
6. When textual generation is required, the app calls `llm_utils` and records
   events via `llmops/llmops_logger.py` while applying guardrails to the LLM output.
7. Evaluation tests and metrics live in `evaluation/` and are designed to be
   run with `pytest` for automated checks.

## Running tests

Run unit and integration tests with pytest from the repository root:

```bash
pytest evaluation/unit_tests
pytest evaluation/trajectory_tests
pytest evaluation/e2e_tests
```

## Notes & Tips

- Ensure any LLM runtime (Ollama or other) is running and reachable if the
  project is configured to use it.
- MLflow is optional but used here for experiment tracking; start it before
  the backend if configured in `config.py`.